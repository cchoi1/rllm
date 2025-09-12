import json
import re
import time

import hashlib
import asyncio
from typing import Any, Optional, Dict, Tuple

from rllm.rewards.reward_types import RewardConfig, RewardOutput, RewardType
from rllm.rewards.code_reward import (
    extract_code_from_model,
    clean_code_main_block,
    lcb_check_correctness_v2,
    taco_to_lcb_format,
    leetcode_check_correctness,
    kodcode_check_correctness,
    humanevalplus_check_correctness,
    codetool_check_correctness,
)
from rllm.tools.code_tools.together_tool import TogetherCodeTool  # optional

from rllm.client.llm_client import LLMClient, LLMClientType
from rllm.data.utils import fetch_live_code_bench_system_prompt
from rllm.agents.context_manager_agent import _format_verifier_results


class Solver:
    """
    Wraps a code generation model using LLMClient for remote vLLM server calls.
    Must implement: generate(problem, feedback, extra)
    """
    def __init__(self,
                 model_name: str,
                 max_tokens: int = 1024,
                 temperature: float = 0.2,
                 remote_url: str = "http://localhost:12345/v1",
                 remote_api_key: str = "None",
                 timeout: float = 600.0,
                 max_retries: int = 3):
        
        self.model_name = model_name
        self.remote_url = remote_url.rstrip("/")
        self.remote_api_key = remote_api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_client = None
    
    def _get_client(self):
        if self.llm_client is None:
            self.llm_client = LLMClient(
                llm_client_type=LLMClientType.OpenAI,
                llm_name=self.model_name,
                llm_server_url=self.remote_url,
                llm_server_api_key=self.remote_api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
        return self.llm_client
    
    @staticmethod
    def _build_prompt(problem: str,
                     feedback: Optional[str] = None,
                     prev_attempts: list[dict[str, Any]] = None) -> str:
        """prev_attempts is a list of dictionaries with the following keys:
            - round_idx: the round number
            - feedback: the feedback string
            - solver_code: the solver code
            - results: unit test results of solver_code
        """
        parts = []
        parts.append(problem)
        
        if prev_attempts:
            last = prev_attempts[-1]
            formatted_results = _format_verifier_results(last.get("verifier_results", {}))
            # Prefer raw model output (reasoning + code), fall back to code only
            last_attempt_txt = last.get("solver_full_output") or last.get("solver_output") or ""
            parts.append("\nPrevious attempt (for reference; DO NOT copy):\n")
            parts.append(last_attempt_txt.strip())
            parts.append("\nUnit test results summary:\n")
            parts.append(formatted_results.strip())

        if feedback:
            parts.append(
                "\nYour previous attempt was incorrect. For your next solution, apply the following guidance:\n"
            )
            parts.append(feedback.strip())
        
        prompt = "\n".join(parts)
        return prompt

    async def generate_async(self, problem: str, feedback: Optional[str], prev_attempts: list[dict[str, Any]] = None) -> tuple[str, str]:
        prompt = self._build_prompt(problem, feedback, prev_attempts)
        messages = [{"role": "user", "content": prompt}]
        last_err = None
        client = self._get_client()
        for attempt in range(self.max_retries):
            try:
                text, finish_reason = await client.generate(messages)
                return text, prompt
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1.0 * (2 ** attempt))  # Exponential backoff
                else:
                    break
        print(f"Warning: Failed to generate from remote server after {self.max_retries} attempts: {last_err}")
        return "", prompt
    
    def generate(self, problem: str, feedback: Optional[str], prev_attempts: list[dict[str, Any]] = None) -> tuple[str, str]:
        return asyncio.run(self.generate_async(problem, feedback, prev_attempts))


class RewardContextAssistFn:
    """
    Train the ContextManager (action = feedback string) by checking if a separate Solver,
    when given this feedback, can now produce correct code.

    Reward modes:
      - binary: 1.0 if Solver passes, 0.0 otherwise
      - marginal: +1 only if feedback improves over baseline (no-feedback)
      - fractional: pass ratio (e.g., passed_tests/total_tests) as shaping
    """

    def __init__(self,
                 config: RewardConfig,
                 solver_args: dict,
                 use_marginal_improvement: bool = True,
                 use_together_code_interpreter: bool = False):
        self.config = config
        self.solver = Solver(**solver_args)
        self.use_marginal = use_marginal_improvement
        self.use_tci = use_together_code_interpreter

    # ---- helpers to evaluate solver code (direct evaluators; Option A) ----
    def _eval_code(self, dataset_name: str, tests: Any, code: str) -> Tuple[bool, Dict[str, Any]]:
        if code is None:
            return False, {"error": "no code extracted from solver output"}

        if dataset_name in ["taco", "apps", "code_contests"]:
            if self.use_tci:
                codetool = TogetherCodeTool()
                return codetool_check_correctness(tests, code, codetool, is_taco_format=True)
            else:
                lcb_tests = taco_to_lcb_format(tests)
                return lcb_check_correctness_v2(lcb_tests, code, debug=False)

        if dataset_name == "leetcode":
            return leetcode_check_correctness(tests, code)

        if dataset_name in ["livecodebench", "codeforces", "primeintellect"]:
            if isinstance(tests, str):
                tests = json.loads(tests)
            return lcb_check_correctness_v2(tests, code, debug=False)

        if dataset_name == "kodcode":
            return kodcode_check_correctness(tests, code)

        if dataset_name == "humanevalplus":
            return humanevalplus_check_correctness(tests, code)

        return False, {"error": f"Dataset {dataset_name} not implemented in context assist reward"}


    def _extract_solver_code(self, solver_text: str) -> Optional[str]:
        code = extract_code_from_model(solver_text)
        if code:
            code = clean_code_main_block(code)
        return code

    # ---- main entrypoint (RewardFunction protocol) ----
    def __call__(self, task_info: dict, action: str, prev_attempts: list[dict[str, Any]] = None) -> RewardOutput:
        """
        task_info:
          {
            "problem": <str>,
            "problem_type": RewardType.CODE,
            "data_source": <dataset_name>,
          }

        action: the ContextManager feedback string
        """
        dataset_name = task_info.get("data_source", "")
        problem = task_info.get("problem", "")
        tests = task_info.get("ground_truth", None)

        if not tests or not isinstance(problem, str) or not problem.strip():
            return RewardOutput(
                reward=self.config.format_error_reward,
                is_correct=False,
                metadata={"error": "Missing question/tests for context assist reward."}
            )

        # Extract baseline from previous attempts (first attempt)
        baseline_passed = None
        baseline_results = None
        baseline_code = None
        baseline_solver_output = None
        
        if prev_attempts and len(prev_attempts) > 0:
            first_attempt = prev_attempts[0]
            baseline_passed = first_attempt.get("passed", False)
            baseline_results = first_attempt.get("results", {})
            baseline_code = first_attempt.get("code", "")
            baseline_solver_output = first_attempt.get("solver_output", "")

        if not action or not action.strip():
            # No feedback provided, compute baseline if not available
            if baseline_passed is None:
                solver_output, solver_prompt = self.solver.generate(problem=problem, feedback=None)
                code = self._extract_solver_code(solver_output)
                passed, results = self._eval_code(dataset_name, tests, code)
                baseline_passed, baseline_results = passed, results
                baseline_code, baseline_solver_output = code, solver_output
            else:
                passed, results = baseline_passed, baseline_results
                code = baseline_code
                solver_output = baseline_solver_output
                # For baseline, we need to generate the prompt that was used
                solver_prompt = self.solver._build_prompt(problem=problem, feedback=None, prev_attempts=prev_attempts)
                
            final_reward = 1.0 if passed else 0.0
            meta = {
                "initial_passed": passed,
                "initial_results": results,
                "initial_solver_code": code,
                "initial_solver_output": solver_output,
                "initial_solver_prompt": solver_prompt,
                "retry_passed": None,
                "retry_results": None,
                "retry_solver_code": None,
                "retry_solver_output": None,
                "retry_solver_prompt": None
            }
        else:
            # Feedback provided, compute retry with feedback
            solver_output, solver_prompt = self.solver.generate(problem=problem, feedback=action, prev_attempts=prev_attempts)
            code = self._extract_solver_code(solver_output)
            passed, results = self._eval_code(dataset_name, tests, code)

            # ---------- reward shaping ----------
            reward_binary = 1.0 if passed else 0.0

            frac_reward = None

            if self.use_marginal and baseline_passed is not None:
                if passed and not baseline_passed:
                    final_reward = 1.0
                elif passed and baseline_passed:
                    final_reward = 1.0
                else:
                    final_reward = 0.0
            else:
                final_reward = reward_binary

            meta = {
                "initial_passed": baseline_passed,
                "initial_results": baseline_results,
                "initial_solver_code": baseline_code,
                "initial_solver_output": baseline_solver_output,
                "initial_solver_prompt": self.solver._build_prompt(problem=problem, feedback=None, prev_attempts=prev_attempts),
                "retry_passed": passed,
                "retry_results": results,
                "retry_solver_code": code,
                "retry_solver_output": solver_output,
                "retry_solver_prompt": solver_prompt
            }

        return RewardOutput(
            reward=float(final_reward if passed else self.config.incorrect_reward if not self.use_marginal else final_reward),
            is_correct=bool(passed),
            metadata=meta,
        )


def rllm_reward_fn_context_assist(data_source: str,
                                  feedback: str,
                                  ground_truth: dict,
                                  problem: str,
                                  prev_attempts: list[dict[str, Any]] = None,
                                  **kwargs):
    """
    Convenience wrapper if you want a function-style entry (mirrors rllm_reward_fn_code).
    """
    reward_config = RewardConfig()
    
    # Extract solver configuration from kwargs
    solver_model_path = kwargs.get("solver_model_path", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    gen_params = kwargs.get("gen", {})
    remote_url = kwargs.get("remote_url", "http://localhost:12345/v1")
    remote_api_key = kwargs.get("remote_api_key", "None")
    use_marginal_improvement = kwargs.get("use_marginal_improvement", True)
    use_together_code_interpreter = kwargs.get("use_together_code_interpreter", False)
    
    # Create solver arguments dictionary
    solver_args = {
        "model_name": solver_model_path,
        "max_tokens": gen_params.get("max_new_tokens", 1024),
        "temperature": gen_params.get("temperature", 0.2),
        "remote_url": remote_url,
        "remote_api_key": remote_api_key,
    }
    
    reward_fn = RewardContextAssistFn(
        reward_config,
        solver_args=solver_args,
        use_marginal_improvement=use_marginal_improvement,
        use_together_code_interpreter=use_together_code_interpreter,
    )
    task_info = {
        "problem": problem,
        "problem_type": RewardType.CODE,
        "data_source": data_source,
        "ground_truth": ground_truth,
        "previous_solution": kwargs.get("previous_solution"),
    }
    return reward_fn(task_info, feedback, prev_attempts)


if __name__ == "__main__":
    import json
    import argparse
    from rllm.data.dataset import DatasetRegistry
    
    print("Loading dataset...")
    test_dataset = DatasetRegistry.load_dataset("lcb", "test")
    if test_dataset is None:
        print("Dataset not found, preparing dataset...")
        from prepare_deepcoder_data import prepare_deepcoder_data
        _, test_dataset = prepare_deepcoder_data()
    tasks = test_dataset.get_data()

    # Test on first task only
    if tasks:
        task = tasks[0]
        print(f"Testing on task: {task.get('question', task.get('problem', ''))[:100]}...")
        
        # Create task_info for the reward function
        task_info = {
            "problem": task.get("question", task.get("problem", "")),
            "data_source": "livecodebench",
            "ground_truth": task.get("ground_truth", ""),
        }
        
        feedback = "Consider the edge cases more carefully."
        
        try:
            # Test rllm_reward_fn_context_assist with solver arguments
            reward = rllm_reward_fn_context_assist(
                data_source="livecodebench",
                feedback=feedback,
                ground_truth=task.get("ground_truth", ""),
                problem=task.get("question", task.get("problem", "")),
                prev_attempts=[],  # No previous attempts for this test
                solver_model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                remote_url="http://localhost:12345/v1",
                remote_api_key="None",
                gen={
                    "max_new_tokens": 32768,
                    "temperature": 0.2
                }
            )
            print(f"Reward: {reward.reward}")
            print(f"Correct: {reward.is_correct}")
            print(f"Metadata: {reward.metadata}")
        except Exception as e:
            print(f"Error testing reward function: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No tasks found in dataset")
