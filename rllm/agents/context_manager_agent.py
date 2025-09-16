from __future__ import annotations
import copy
import re
from typing import Any, Dict, List, Optional

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

def _truncate(s: Any, length: int = 1200) -> str:
    s = s if isinstance(s, str) else str(s)
    if len(s) <= length:
        return s
    half = length // 2
    return s[:half] + "...(truncated)..." + s[-half:]

def _format_verifier_results(verifier_results: Dict[str, Any]) -> str:
    if not isinstance(verifier_results, dict):
        return str(verifier_results)
    all_passed = verifier_results.get('all_passed', False)
    total_tests = verifier_results.get('total_tests', 0)
    passed_tests = verifier_results.get('passed_tests', 0)
    test_results = verifier_results.get('test_results', [])
    summary = f"Tests: {passed_tests}/{total_tests} passed"
    if all_passed:
        return f"{summary} - All tests passed!"
    failed_tests = []
    for i, test in enumerate(test_results):
        if not test.get('passed', True):
            input_data = test.get('input', 'N/A')
            expected = test.get('expected', 'N/A')
            output = test.get('output', 'N/A')
            error_msg = test.get('error_message', '')
            input_str = _truncate(str(input_data), 100)
            expected_str = _truncate(str(expected), 100)
            output_str = _truncate(str(output), 100)
            msg = f"Test {i+1}: input={input_str}, expected={expected_str}, got={output_str}"
            if error_msg:
                msg += f" ({error_msg})"
            failed_tests.append(msg)
    return f"{summary} \nFailed tests:\n" + "\n".join(failed_tests) if failed_tests else f"{summary} "


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

        # Update running memory from observation (not the model output)
        if self.use_memory:
            self._update_memory(observation)

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

        if self.remove_cm_thinking:
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
            # else: treat full response as action

        # Only echo assistant turn if explicitly requested
        if self.keep_history:
            self._messages.append({"role": "assistant", "content": action})

        # Update memory from full response if desired (keeps running summary small)
        if self.use_memory:
            self._extract_and_store_summary(response)

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

    def _update_memory(self, observation: Dict[str, Any]):
        """
        Maintain compact running memory for the episode.
        This NEVER affects chat history directly; it only feeds _format_observation.
        """
        round_idx = int(observation.get("round_idx", 0))
        problem = observation.get("problem", "") or ""
        solver_output = observation.get("solver_output", "") or ""
        solver_full_output = observation.get("solver_full_output", "") or ""
        verifier_results = observation.get("verifier_results", {}) or {}
        feedback = observation.get("feedback", "") or ""

        # Record a compact attempt snapshot
        attempt = {
            "round_idx": round_idx,
            "solver_output": solver_output,
            "solver_full_output": solver_full_output,
            "verifier_results": verifier_results,
            "feedback": feedback,
            "passed_tests": verifier_results.get("passed_tests", observation.get("passed_tests", 0)),
            "total_tests": verifier_results.get("total_tests", observation.get("total_tests", 0)),
            "solved": bool(observation.get("solved", False)),
        }

        if round_idx < len(self._attempt_history):
            self._attempt_history[round_idx] = attempt
        else:
            self._attempt_history.append(attempt)

        if round_idx == 0:
            self._problem_summary = problem or self._problem_summary

        # Keep memory bounded (drop oldest if huge)
        if len(self._attempt_history) > 16:  # arbitrary safety cap
            self._attempt_history = self._attempt_history[-16:]

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

        last_verifier_results = _format_verifier_results(
            observation.get("verifier_results", {}) or {}
        )

        if not self.use_memory:
            parts = [
                "Analyze a solver's previous attempt at a code problem and its unit test results. Provide ONLY feedback and analysis; DO NOT provide code, pseudocode, or code-like snippets.",
                f"\nProblem:\n{problem}",
                f"\nRound {round_idx}:",
                f"\n**Latest Attempt (high-level):**\n{_truncate(last_solver_output, 2000)}",
                f"\n**Latest Test Results:**\n{last_verifier_results}",
                "\n**Your Response Should Include ONLY:**",
                "- Clear explanation of mistakes or misunderstandings.",
                "- Guidance on what needs to change conceptually.",
                "- High-level reasoning about why the solution failed and what to improve.",
            ]
        else:
            current_summary_text = self._current_summary or "No previous summary available."
            # Bound the summary injected into the prompt
            # current_summary_text = _truncate(current_summary_text, self._memory_max_chars)

            parts = [
                "You are a Context Manager that provides feedback and maintains a structured running summary.",
                "Analyze a solver's previous attempt at a code problem and its unit test results. Provide ONLY feedback and analysis — absolutely DO NOT provide code, pseudocode, or code-like snippets.",
                f"\nProblem:\n{problem}",
                f"\nRound {round_idx}:",
                f"\n**Current Summary (running log, no code):**\n{current_summary_text}",
                f"\n**Latest Attempt (high-level):**\n{_truncate(last_solver_output, 2000)}",
                f"\n**Latest Test Results:**\n{last_verifier_results}",
                "\n**Your Response MUST Include ONLY:**",
                "1. **Summary**: Update the running memory by adding this round (attempt, outcome, causal insight). Keep it concise and cumulative.",
                "2. **Feedback**: Specific, actionable guidance (no code).",
                "3. **Analysis**: Patterns/root causes across attempts so far (no code).",
                "\n**Format your response with clear sections:**",
                "## Summary\n[Updated running summary — no code]",
                "## Analysis\n[Patterns/root causes — no code]",
                "## Feedback\n[Actionable guidance — no code]",
            ]

        return "\n\n".join([p for p in parts if p])

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


# OLD
# from __future__ import annotations
# import copy
# import re
# from typing import Any, Dict, List, Optional

# from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


# def _truncate(s: Any, length: int = 1200) -> str:
#     s = s if isinstance(s, str) else str(s)
#     if len(s) <= length:
#         return s
#     half = length // 2
#     return s[:half] + "...(truncated)..." + s[-half:]


# def _format_verifier_results(verifier_results: Dict[str, Any]) -> str:
#     """
#     Format verifier results concisely into text.
    
#     Args:
#         verifier_results: Dict with keys 'all_passed', 'test_results', 'total_tests', 'passed_tests'
    
#     Returns:
#         Concise text summary of test results
#     """
#     if not isinstance(verifier_results, dict):
#         return str(verifier_results)
    
#     all_passed = verifier_results.get('all_passed', False)
#     total_tests = verifier_results.get('total_tests', 0)
#     passed_tests = verifier_results.get('passed_tests', 0)
#     test_results = verifier_results.get('test_results', [])
    
#     # Summary line
#     summary = f"Tests: {passed_tests}/{total_tests} passed"
#     if all_passed:
#         return f"{summary} - All tests passed!"
    
#     # Format failed tests
#     failed_tests = []
#     for i, test in enumerate(test_results):  # Limit to first 3 failed tests
#         if not test.get('passed', True):
#             input_data = test.get('input', 'N/A')
#             expected = test.get('expected', 'N/A')
#             output = test.get('output', 'N/A')
#             error_msg = test.get('error_message', '')
            
#             # Truncate long inputs/outputs
#             input_str = _truncate(str(input_data), 100)
#             expected_str = _truncate(str(expected), 100)
#             output_str = _truncate(str(output), 100)
            
#             failed_test = f"Test {i+1}: input={input_str}, expected={expected_str}, got={output_str}"
#             if error_msg:
#                 failed_test += f" ({error_msg})"
#             failed_tests.append(failed_test)
    
#     if failed_tests:
#         return f"{summary} \nFailed tests:\n" + "\n".join(failed_tests)
#     else:
#         return f"{summary} "


# class ContextManagerAgent(BaseAgent):
#     """
#     A feedback-only agent: given (problem, last_solver_output, last_error_trace, optional test_results),
#     produce concise, actionable feedback that helps the Solver fix the code.
#     """

#     def __init__(
#         self,
#         remove_cm_thinking: bool,
#         system_instruction: str,
#         use_memory: bool,
#         use_solver_cot: bool
#     ):
#         self._trajectory = Trajectory()
#         self._messages: List[Dict[str, str]] = []
#         self._current_obs_str: Optional[str] = None
#         self._current_obs = None

#         self.remove_cm_thinking = remove_cm_thinking
#         self.system_instruction = system_instruction
#         self.use_memory = use_memory
#         self._initialized = False

#         # Memory for tracking previous attempts and generated summaries
#         self._attempt_history: List[Dict[str, Any]] = []
#         self._problem_summary: str = ""
#         self._current_summary: str = ""

#         # Solver CoT
#         self.use_solver_cot = use_solver_cot

#     # ---------- BaseAgent API ----------

#     def reset(self):
#         self._trajectory = Trajectory()
#         self._messages = []
#         self._current_obs_str = None
#         self._current_obs = None
#         self._initialized = False
#         # Reset memory
#         self._attempt_history = []
#         self._problem_summary = ""
#         self._current_summary = ""

#     def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
#         """
#         Accepts observations produced by ContextManagerEnv.reset() or ContextManagerEnv.step():
#           {"round_idx", "problem", "feedback", "solver_output", "verifier_results"}
#         """
#         # End the trajectory if the Solver's solution is correct
#         if done:
#             return

#         # Lazily add a 'system' rule once per episode
#         if not self._initialized and self.system_instruction:
#             self._messages.append({"role": "system", "content": self.system_instruction})
#             self._initialized = True

#         # Update memory with current observation
#         if self.use_memory:
#             self._update_memory(observation)

#         obs_text = self._format_observation(observation)
#         self._messages.append({"role": "user", "content": obs_text})
#         self._current_obs_str = obs_text
#         self._current_obs = observation

#     def update_from_model(self, response: str, **kwargs) -> Action:
#         """
#         If remove_cm_thinking=True, strip a single </think> block from what we
#         append to _messages (so training prompt doesn't include thoughts).
#         """
#         action = response
#         thought = ""
#         if self.remove_cm_thinking and "</think>" in response:
#             parts = response.split("</think>", 1)
#             thought = parts[0] + "</think>"
#             action = parts[1].strip()

#         # APPEND ONLY ACTION to chat history
#         self._messages.append({"role": "assistant", "content": action})

#         # Keep summary extraction based on the full response (optional)
#         if self.use_memory:
#             self._extract_and_store_summary(response)

#         step = Step(
#             chat_completions=copy.deepcopy(self.chat_completions),
#             action=action,
#             model_response=response,
#             observation=self._current_obs_str,
#         )
#         # (optional) save the thought on the Step for analysis
#         try:
#             step.thought = thought
#         except Exception:
#             pass

#         self._trajectory.steps.append(step)
#         return Action(action=action)


#     def _extract_and_store_summary(self, response: str):
#         """Extract the updated summary from the agent's response and store it"""
#         # Look for summary section in the response
#         # The agent should format responses with clear sections
#         summary_match = re.search(r"##? Summary[:\s]*\n(.*?)(?=\n##? |\n##? Feedback|\n##? Analysis|\n$)", response, re.DOTALL | re.IGNORECASE)
#         if summary_match:
#             self._current_summary = summary_match.group(1).strip()
#         else:
#             # If no explicit summary section, try to extract from the beginning
#             # Look for content before any clear section markers
#             lines = response.split('\n')
#             summary_lines = []
#             for line in lines:
#                 if line.strip().startswith('##') or line.strip().startswith('**') or 'feedback' in line.lower() or 'analysis' in line.lower():
#                     break
#                 if line.strip():
#                     summary_lines.append(line.strip())
            
#             if summary_lines:
#                 self._current_summary = '\n'.join(summary_lines).strip()

#     def _update_memory(self, observation: Dict[str, Any]):
#         """Update memory with current observation"""
#         round_idx = observation.get("round_idx", 0)
#         problem = observation.get("problem", "")
#         solver_output = observation.get("solver_output", "")
#         solver_full_output = observation.get("solver_full_output", "")
#         verifier_results = observation.get("verifier_results", {})
#         feedback = observation.get("feedback", "")
        
#         # Store attempt in history
#         attempt = {
#             "round_idx": round_idx,
#             "solver_output": solver_output,
#             "solver_full_output": solver_full_output,
#             "verifier_results": verifier_results,
#             "feedback": feedback,
#             "passed_tests": observation.get("passed_tests", 0),
#             "total_tests": observation.get("total_tests", 0),
#             "solved": observation.get("solved", False)
#         }
        
#         # Update or add to history
#         if round_idx < len(self._attempt_history):
#             self._attempt_history[round_idx] = attempt
#         else:
#             self._attempt_history.append(attempt)
        
#         # Update problem summary if this is the first round
#         if round_idx == 0:
#             self._problem_summary = problem

#     def _format_observation(self, observation: Dict[str, Any]) -> str:
#         """
#         Builds a feedback prompt that asks for both feedback and an updated summary.
#           {"round_idx", "problem", "feedback", "solver_output", "verifier_results"}
#         """
#         problem = observation["problem"]
#         round_idx = observation["round_idx"]
#         if self.use_solver_cot:
#             last_solver_output = observation["solver_full_output"]
#         else:
#             last_solver_output = observation["solver_output"]
#         last_verifier_results = _format_verifier_results(observation["verifier_results"])
#         last_feedback = observation["feedback"]

#         if not self.use_memory:
#             parts = [
#                 f"Analyze a solver's previous attempt at a code problem and its unit test results. Provide ONLY feedback and analysis on how the solver can correct their solution. Absolutely DO NOT provide any code, pseudocode, or code-like snippets.",
#                 f"\nProblem:\n{problem}",
#                 f"\nRound {round_idx}:",
#                 f"\n**Latest Attempt:**\n{last_solver_output}",
#                 f"\n**Latest Test Results:**\n{last_verifier_results}",
#                 f"\n**Your Response Should Include ONLY:**",
#                 f"- Clear explanation of mistakes or misunderstandings.",
#                 f"- Guidance on what needs to change conceptually.", 
#                 f"- High-level reasoning about why the solution failed and what to improve.",
#             ]
#         else:
#             # Include current summary and ask for updated summary + feedback
#             parts = [
#                 f"Analyze a solver's previous attempt at a code problem and its unit test results. Provide ONLY feedback and analysis on how the solver can correct their solution. Absolutely DO NOT provide any code, pseudocode, or code-like snippets.",
#                 f"\nProblem:\n{problem}",
#                 f"\nRound {round_idx}:",
#                 f"\n**Latest Attempt:**\n{last_solver_output}",
#                 f"\n**Latest Test Results:**\n{last_verifier_results}",
#                 f"\n**Latest Feedback:**\n{last_feedback}",
#                 f"\n**Your Response Should Include ONLY:**",
#                 f"- Clear explanation of mistakes or misunderstandings.",
#                 f"- Guidance on what needs to change conceptually.", 
#                 f"- High-level reasoning about why the solution failed and what to improve.",
#             ]
#             # current_summary_text = self._current_summary if self._current_summary else "No previous summary available."
#             # parts = [
#             #     "You are a Context Manager that provides both feedback and maintains a structured running summary.",
#             #     "Analyze a solver's previous attempt at a code problem and its unit test results. Provide ONLY feedback and analysis — absolutely DO NOT provide any code, pseudocode, or code-like snippets.",

#             #     f"\nProblem:\n{problem}",

#             #     f"\nRound {round_idx}:",

#             #     f"\n**Current Summary (running log of all attempts so far):**\n{current_summary_text}",

#             #     f"\n**Latest Attempt:**\n{last_solver_output}",

#             #     f"\n**Latest Test Results:**\n{last_verifier_results}",

#             #     "\n**Your Response MUST Include ONLY:**",
#             #     "1. **Summary**: Update the running memory of the solver's attempts by adding a new entry for this round. Each entry should include:",
#             #     "   - Attempt number",
#             #     "   - What the solver tried (at a high level, no code)",
#             #     "   - Outcome of the attempt (errors fixed or new issues)", 
#             #     "   - Any causal insight (e.g., 'fix resolved X but revealed Y').",
#             #     "   Keep the summary concise but cumulative, so it reflects the progression of all attempts so far.",
#             #     "2. **Feedback**: Provide specific, actionable feedback on how the solver can improve, without writing or suggesting code.",
#             #     "3. **Analysis**: Identify patterns or root causes across all attempts so far (e.g., recurring mistakes, signs of progress, shifts in error type).",

#             #     "\n**Format your response with clear sections:**",
#             #     "## Summary\n[Updated running summary including this attempt — no code]",
#             #     "## Analysis\n[Pattern analysis and root cause identification — no code]", 
#             #     "## Feedback\n[Specific, actionable guidance for the solver — no code]",
#             # ]

#         obs = "\n\n".join([p for p in parts if p])
#         return obs

#     @property
#     def chat_completions(self) -> List[Dict[str, str]]:
#         return self._messages

#     @property
#     def trajectory(self) -> Trajectory:
#         return self._trajectory

#     def get_current_state(self) -> Optional[Step]:
#         if not self._trajectory.steps:
#             return None
#         return self._trajectory.steps[-1]
