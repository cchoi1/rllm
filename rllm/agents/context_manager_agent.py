from __future__ import annotations
import copy
import re
from typing import Any, Dict, List, Optional

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


def _format_verifier_results(verifier_results: Dict[str, Any]) -> str:
    print(f"\n\nVerifier Results: {verifier_results}")
    if not isinstance(verifier_results, dict):
        return str(verifier_results)
    all_passed = verifier_results.get('all_passed', False)
    total_tests = verifier_results.get('total_tests', 0)
    passed_tests = verifier_results.get('passed_tests', 0)
    test_results = verifier_results.get('test_results', [])
    
    if all_passed:
        return "Congratulations! You've successfully passed all test cases. Please carefully review your solution one more time to ensure it handles all edge cases properly. If you're confident your code is optimal, you can proceed with outputting your final solution."
    
    formatted_test_results = ""
    n_failed = 0
    for i, test in enumerate(test_results):
        if not test.get('passed', True):
            formatted_test_results += f"### Test {i + 1} failed\n"
            formatted_test_results += f"  Input: {truncatefn(test['input'])}\n"
            formatted_test_results += f"  Expected: {truncatefn(test['expected'])}\n"
            if 'output' in test and test['output'] is not None:
                formatted_test_results += f"  Actual: {truncatefn(test['output'])}\n\n"
            if 'error_message' in test and test['error_message'] is not None:
                formatted_test_results += f"  Error message: {truncatefn(test['error_message'])}\n"
            n_failed += 1
    
    if n_failed > 0:
        return f"Here are the results on the public test cases:\n{formatted_test_results}\nSome test cases are still failing. Please carefully analyze the error patterns, revise your code to address these issues, and ensure your solution handles all the test cases correctly. Then, output your final code."
    else:
        return "Congratulations! You've successfully passed all test cases. Please carefully review your solution one more time to ensure it handles all edge cases properly. If you're confident your code is optimal, you can proceed with outputting your final solution."


class ContextManagerAgent(BaseAgent):
    """
    Feedback-only agent: observes (problem, latest solver output, verifier results),
    produces concise actionable feedback. Supports stateless prompts per step.
    """

    def __init__(
        self,
        remove_cm_thinking: bool,
        system_instruction: str,
        use_memory: bool,
        use_solver_cot: bool,
        keep_history: bool = False,
        memory_max_chars: int = 4000,   # soft cap for running summary
    ):
        self._trajectory = Trajectory()
        self._messages: List[Dict[str, str]] = []
        self._current_obs_str: Optional[str] = None
        self._current_obs = None

        self.remove_cm_thinking = remove_cm_thinking
        self.system_instruction = system_instruction
        self.use_memory = use_memory
        self.use_solver_cot = use_solver_cot

        # history policy
        self.keep_history = keep_history

        # memory / summary
        self._attempt_history: List[Dict[str, Any]] = []
        self._problem_summary: str = ""
        self._current_summary: str = ""
        self._memory_max_chars = int(memory_max_chars)

        self._initialized = False

    # ---------- BaseAgent API ----------

    def reset(self):
        self._trajectory = Trajectory()
        self._messages = []
        self._current_obs_str = None
        self._current_obs = None
        self._initialized = False
        self._attempt_history = []
        self._problem_summary = ""
        self._current_summary = ""

    def _build_fresh_messages(self, obs_text: str) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if self.system_instruction:
            msgs.append({"role": "system", "content": self.system_instruction})
        msgs.append({"role": "user", "content": obs_text})
        return msgs

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Build prompt for the current step only (unless keep_history=True)."""
        if done:
            return

        # Lazy system message if we *do* keep history
        if self.keep_history and not self._initialized and self.system_instruction:
            self._messages.append({"role": "system", "content": self.system_instruction})
            self._initialized = True

        # # Update running memory from observation (not the model output)
        # if self.use_memory:
        #     self._update_memory(observation)

        # Text for the user turn (stateless description of current state)
        obs_text = self._format_observation(observation)

        if self.keep_history:
            # Append to existing chat
            self._messages.append({"role": "user", "content": obs_text})
        else:
            # Rebuild a fresh prompt each step: [system?, user(obs)]
            self._messages = self._build_fresh_messages(obs_text)

        self._current_obs_str = obs_text
        self._current_obs = observation

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Convert model response to action. If remove_cm_thinking=True,
        strip a single </think> block from the *returned action* only.
        """
        action = response
        thought = ""

        # Default safety: if the model started a <think> block but did not finish it,
        # treat feedback to the solver as empty by default.
        if "<think>" in response and "</think>" not in response:
            # Preserve thought if memory is enabled; blank actionable feedback
            if self.use_memory:
                thought = response.strip()
            action = ""
        elif self.remove_cm_thinking:
            # Robust: tolerate missing tags / cutoffs
            if "</think>" in response:
                pre, post = response.split("</think>", 1)
                # Capture inner content if <think> exists, otherwise just treat pre as thought
                if "<think>" in pre:
                    thought_inner = pre.split("<think>", 1)[1].strip()
                else:
                    thought_inner = pre.strip()
                thought = (thought_inner + "</think>").strip()
                action = post.strip()
            else:
                if self.use_memory:
                    thought = response
                    action = ""
            # else: treat full response as action

        # Only echo assistant turn if explicitly requested
        if self.keep_history:
            self._messages.append({"role": "assistant", "content": action})

        # # Update memory from full response if desired (keeps running summary small)
        # if self.use_memory:
        #     self._extract_and_store_summary(response)

        step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=action,
            model_response=response,
            observation=self._current_obs_str,
        )
        try:
            setattr(step, "thought", thought)
        except Exception:
            pass
        
        # Store raw observation dict in step.info for easier extraction later
        if self._current_obs and isinstance(self._current_obs, dict):
            step.info['raw_observation'] = self._current_obs

        self._trajectory.steps.append(step)
        return Action(action=action)

    # ---------- Memory / Prompt Formatting ----------

    def _extract_and_store_summary(self, response: str):
        """Extract an updated summary from the agent's response."""
        m = re.search(
            r"##?\s*Summary[:\s]*\n(.*?)(?=\n##?\s|\Z)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            candidate = m.group(1).strip()
        else:
            # Heuristic: take leading lines before any section headers
            lines = response.splitlines()
            buf = []
            for line in lines:
                low = line.strip().lower()
                if low.startswith("##") or "feedback" in low or "analysis" in low:
                    break
                if line.strip():
                    buf.append(line.strip())
            candidate = "\n".join(buf).strip()

        if candidate:
            # Soft clamp summary length
            # if len(candidate) > self._memory_max_chars:
            #     candidate = _truncate(candidate, self._memory_max_chars)
            self._current_summary = candidate

    # def _update_memory(self, observation: Dict[str, Any]):
    #     """
    #     Maintain compact running memory for the episode.
    #     This NEVER affects chat history directly; it only feeds _format_observation.
    #     """
    #     round_idx = int(observation.get("round_idx", 0))
    #     problem = observation.get("problem", "") or ""
    #     solver_output = observation.get("solver_output", "") or ""
    #     solver_full_output = observation.get("solver_full_output", "") or ""
    #     verifier_results = observation.get("verifier_results", {}) or {}
    #     feedback = observation.get("feedback", "") or ""

    #     # Record a compact attempt snapshot
    #     attempt = {
    #         "round_idx": round_idx,
    #         "solver_output": solver_output,
    #         "solver_full_output": solver_full_output,
    #         "verifier_results": verifier_results,
    #         "feedback": feedback,
    #         "passed_tests": verifier_results.get("passed_tests", observation.get("passed_tests", 0)),
    #         "total_tests": verifier_results.get("total_tests", observation.get("total_tests", 0)),
    #         "solved": bool(observation.get("solved", False)),
    #     }

    #     if round_idx < len(self._attempt_history):
    #         self._attempt_history[round_idx] = attempt
    #     else:
    #         self._attempt_history.append(attempt)

    #     if round_idx == 0:
    #         self._problem_summary = problem or self._problem_summary

    #     # Keep memory bounded (drop oldest if huge)
    #     if len(self._attempt_history) > 16:  # arbitrary safety cap
    #         self._attempt_history = self._attempt_history[-16:]

    def _format_observation(self, observation: Dict[str, Any]) -> str:
        """
        Stateless view of the CURRENT problem state.
        If use_memory=True, includes a compact running summary (not raw history).
        """
        problem = observation.get("problem", "") or ""
        round_idx = int(observation.get("round_idx", 0))

        last_solver_output = observation.get(
            "solver_full_output" if self.use_solver_cot else "solver_output",
            ""
        ) or ""
        print(f"\n\nLast Solver Output: {last_solver_output}")

        last_verifier_results = _format_verifier_results(
            observation.get("verifier_results", {}) or {}
        )

        last_feedback = observation.get("feedback", "") or ""

        if not self.use_memory:
            parts = [
                "Analyze a solver's previous attempt at a code problem and its unit test results. Provide ONLY feedback and analysis; DO NOT provide code, pseudocode, or code-like snippets.",
                f"\nProblem:\n{problem}",
                f"\nRound {round_idx}:",
                f"\n**Latest Attempt (high-level):**\n{truncatefn(last_solver_output, 2000)}",
                f"\n**Latest Test Results:**\n{last_verifier_results}",
                "\n**Your Response Should Include ONLY:**",
                "- Clear explanation of mistakes or misunderstandings.",
                "- Guidance on what needs to change conceptually.",
                "- High-level reasoning about why the solution failed and what to improve.",
            ]
        else:
            # current_summary_text = self._current_summary or "No previous summary available."
            # Bound the summary injected into the prompt
            # current_summary_text = _truncate(current_summary_text, self._memory_max_chars)

            parts = [
                "Analyze a solver's previous attempt at a code problem and its unit test results. Provide ONLY feedback and analysis; DO NOT provide code, pseudocode, or code-like snippets.",
                f"\nProblem:\n{problem}",
                f"\nRound {round_idx}:",
                f"\n**Latest Feedback (running log, no code):**\n{last_feedback}",
                f"\n**Latest Attempt (high-level):**\n{truncatefn(last_solver_output, 2000)}",
                f"\n**Latest Test Results:**\n{last_verifier_results}",
                "\n**Your Response Should Include ONLY:**",
                "- Clear explanation of mistakes or misunderstandings.",
                "- Guidance on what needs to change conceptually.",
                "- High-level reasoning about why the solution failed and what to improve.",
            ]

        cm_prompt = "\n\n".join([p for p in parts if p])
        print(f"\n\nCM Prompt: {cm_prompt}")
        return cm_prompt

    # ---------- Introspection ----------

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        return self._messages

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def get_current_state(self) -> Optional[Step]:
        if not self._trajectory.steps:
            return None
        return self._trajectory.steps[-1]


if __name__ == "__main__":
    import argparse
    import torch

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        AutoTokenizer = None
        AutoModelForCausalLM = None

    # Optional vLLM (only if --backend vllm)
    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except Exception:
        LLM = None
        SamplingParams = None

    def build_prompt_from_messages(tokenizer, messages: list[dict[str, str]]) -> str:
        """
        Use the tokenizer's chat template to convert messages into a single prompt string.
        Works for Qwen/DeepSeek-style chat models.
        """
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback: naive concatenation
        parts = []
        for m in messages:
            role = m.get("role", "user")
            parts.append(f"<{role}>\n{m.get('content','')}\n</{role}>")
        parts.append("<assistant>\n")
        return "\n".join(parts)

    def generate_vllm(model_name: str, messages, max_new_tokens: int, temperature: float, top_p: float):
        assert LLM is not None and SamplingParams is not None, "vLLM not installed."
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        prompt = build_prompt_from_messages(tok, messages)
        engine = LLM(model=model_name, trust_remote_code=True, tensor_parallel_size=1, gpu_memory_utilization=0.9)
        samp = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        outs = engine.generate([prompt], samp)
        return outs[0].outputs[0].text.strip() if outs and outs[0].outputs else ""

    parser = argparse.ArgumentParser(description="Standalone smoke test for ContextManagerAgent with a real LLM.")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--remove_cm_thinking", type=int, default=1)
    parser.add_argument("--use_memory", type=int, default=1)
    args = parser.parse_args()

    # 1) Instantiate the agent
    agent = ContextManagerAgent(
        remove_cm_thinking=bool(args.remove_cm_thinking),
        system_instruction="You are a feedback-only agent that provides concise, actionable feedback to help the Solver fix their code.",
        use_memory=bool(args.use_memory),
        use_solver_cot=bool(args.use_solver_cot),
    )

    # 2) Turn 0: initial observation (problem + starter), produces first feedback
    obs0 = {
        "problem": "Implement function add(a, b) that returns the sum of a and b.",
        "solver_output": "def add(a, b):\n    return a - b\n",
        "round_idx": 0,
        "verifier_results": {"passed_tests": 0, "total_tests": 3, "all_passed": False},
        "passed_tests": 0,
        "total_tests": 3,
        "solved": False,
        "feedback": ""
    }
    agent.update_from_env(obs0, reward=0.0, done=False, info={})

    messages = agent.chat_completions
    response0 = generate_vllm(args.model, messages, args.max_new_tokens, args.temperature, args.top_p)
    act0 = agent.update_from_model(response0)

    print("\n=== TURN 0 ===")
    print("Prompted messages:\n", messages)
    print("\nModel feedback:\n", response0)

    # 3) Turn 1: follow-up observation with latest solver output + error/test results
    # (We simulate test_results here to exercise the formatting; in full training this comes from the env's verifier.)
    obs1 = {
        "problem": obs0["problem"],
        "round_idx": 1,
        "solver_output": "def add(a, b):\n    return a - b\n",
        "verifier_results": {
            "passed_tests": 0, 
            "total_tests": 3, 
            "all_passed": False,
            "test_results": [
                {"input": (1, 2), "expected": 3, "output": -1, "passed": False, "error_message": "assert 3 == -1"},
                {"input": (0, 5), "expected": 5, "output": -5, "passed": False, "error_message": "assert 5 == -5"},
                {"input": (-2, 4), "expected": 2, "output": -6, "passed": False, "error_message": "assert 2 == -6"},
            ]
        },
        "passed_tests": 0,
        "total_tests": 3,
        "solved": False,
        "feedback": act0.action
    }
    agent.update_from_env(obs1, reward=0.0, done=False, info={})

    messages = agent.chat_completions
    response1 = generate_vllm(args.model, messages, args.max_new_tokens, args.temperature, args.top_p)
    _ = agent.update_from_model(response1)

    print("\n=== TURN 1 ===")
    print("Prompted messages:\n", messages)
    print("\nModel feedback:\n", response1)

    print("\nDone. If you see reasonable feedback (and optional </think> stripped), the agent wiring is good.")
