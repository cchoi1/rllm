from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import os
import time
import requests

from rllm.environments.base.multi_turn_env import MultiTurnEnvironment

# Types for dependency injection
SolverGenerateFn = Callable[[str | List[str]],
                            Tuple[str | List[str], Dict[str, Any] | List[Dict[str, Any]]]]
RewardFn = Callable[..., Tuple[Union[float, bool], Dict[str, Any]]]


class ContextManagerEnv(MultiTurnEnvironment):
    """
    Multi-turn environment for training a Context Manager (CM).
    The CM outputs feedback; the env calls a fixed, already-running vLLM server
    (OpenAI-compatible) and computes reward via the provided reward_fn.
    """
    def __init__(
        self,
        task: Optional[Dict[str, Any]] = None,
        max_turns: int = 3,
        # Dependencies
        solver_generate_fn: Optional[SolverGenerateFn] = None,   # optional direct callable
        reward_fn: Optional[RewardFn] = None,
        reward_kwargs: Optional[Dict[str, Any]] = None,
        # Remote vLLM config (preferred)
        solver_remote: Optional[Dict[str, Any]] = None,
        # Legacy local field kept for compat but NOT used
        solver_model_name: Optional[str] = None,
        # Behavior
        system_prompt: str = "",
        use_shaped_reward: bool = False,
        reward_bonus_coeff: float = 0.0,
        truncate_trace_chars: int = 2000,
        **kwargs,
    ):
        super().__init__(task=task, max_turns=max_turns, **kwargs)
        self.solver_generate_fn = solver_generate_fn
        self.solver_remote = solver_remote or {}
        self.solver_model_name = solver_model_name  # ignored on purpose
        self.reward_fn = reward_fn
        self.reward_kwargs = (reward_kwargs or {}).copy()

        self.system_prompt = system_prompt
        self.use_shaped_reward = use_shaped_reward
        self.reward_bonus_coeff = reward_bonus_coeff
        self.truncate_trace_chars = truncate_trace_chars

        self.prev_reward_raw: Optional[float] = None
        self.history: List[Dict[str, Any]] = []

        # Defaults for remote vLLM (accept base_url or base_urls)
        base_url = (self.solver_remote.get("base_url") or "http://localhost:12345/v1").rstrip("/")
        base_urls = self.solver_remote.get("base_urls")
        if base_urls and isinstance(base_urls, list) and len(base_urls) > 0:
            self._remote_base_urls = [u.rstrip("/") for u in base_urls]
        else:
            self._remote_base_urls = [base_url]

        self._remote_model    = self.solver_remote.get("model") or os.environ.get("VLLM_MODEL_NAME") or "default"
        self._remote_api_key  = self.solver_remote.get("api_key") or os.environ.get("VLLM_API_KEY", "EMPTY")
        self._timeout_s       = float(self.solver_remote.get("timeout_s", 30.0))
        self._max_retries     = int(self.solver_remote.get("max_retries", 3))
        self._base_delay_s    = float(self.solver_remote.get("base_delay_s", 1.0))
        self._temperature     = float(self.solver_remote.get("temperature", 0.0))
        self._top_p           = float(self.solver_remote.get("top_p", 0.95))
        self._max_tokens      = int(self.solver_remote.get("max_tokens", 512))
        self._extra_headers   = self.solver_remote.get("extra_headers") or {}

        # Push connection/model defaults into reward_kwargs so reward_fn/Solver sees them
        # Keep legacy keys for backward compatibility
        self.reward_kwargs.setdefault("remote_urls", self._remote_base_urls)
        self.reward_kwargs.setdefault("remote_url", self._remote_base_urls[0])
        self.reward_kwargs.setdefault("remote_api_key", self._remote_api_key)
        self.reward_kwargs.setdefault("solver_model_path", self._remote_model)
        self.reward_kwargs.setdefault("gen", {})
        self.reward_kwargs["gen"].setdefault("temperature", self._temperature)
        self.reward_kwargs["gen"].setdefault("max_new_tokens", self._max_tokens)
        self.reward_kwargs.setdefault("timeout_s", self._timeout_s)
        self.reward_kwargs.setdefault("max_retries", self._max_retries)

        self._last_passed_tests = None
        self._last_total_tests = None
        self._last_solved = False

    # ---------- rllm env API ----------

    def reset(self, task: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        if task is not None:
            self.task = task
        assert self.task is not None, "Task must be set before reset."

        self.done = False
        self.current_turn = 0
        self.history = []
        self.prev_reward_raw = None

        # Generate baseline solver information for the initial observation
        assert self.reward_fn is not None, "Provide reward_fn."
        reward_output = self.reward_fn(
            data_source=self.task["data_source"],
            feedback="",  # No feedback for baseline
            ground_truth=self.task["ground_truth"],
            problem=self.task["prompt"],
            prev_attempts=[],
            **self.reward_kwargs
        )
        metadata = reward_output.metadata or {}

        # Extract baseline solver information
        solver_output = metadata.get("initial_solver_code", "")
        verifier_results = metadata.get("initial_results", {})
        solver_prompt = metadata.get("initial_solver_prompt", "")
        solver_full_output = metadata.get("initial_solver_output", "")

        # Extract test statistics
        passed_tests = verifier_results.get("passed_tests", 0) if isinstance(verifier_results, dict) else 0
        total_tests = verifier_results.get("total_tests", 0) if isinstance(verifier_results, dict) else 0
        solved = bool(metadata.get("initial_passed", False))

        obs = {
            "round_idx": self.current_turn,
            "problem": self.task.get("prompt"),
            "feedback": "",
            "solver_output": solver_output,
            "solver_full_output": solver_full_output,
            "solver_prompt": solver_prompt,
            "verifier_results": verifier_results,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "solved": solved,
        }

        # Add the baseline attempt to history so it can be referenced in subsequent attempts
        baseline_attempt = {
            "round_idx": 0,
            "problem": self.task.get("prompt"),
            "feedback": "",
            "solver_output": solver_output,
            "solver_full_output": solver_full_output,
            "solver_prompt": solver_prompt,
            "verifier_results": verifier_results,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "passed": solved,
        }
        self.history.append(baseline_attempt)

        # If the solver succeeded on the first try, terminate immediately
        if solved:
            self.done = True

        return obs, {}

    def step(self, action: str):
        assert self.task is not None, "Task is not set"
        if self.done:
            return {}, 1.0, True, self.task

        raw_reward, next_obs = self.get_reward_and_next_obs(self.task, action)

        if self.prev_reward_raw is None or self.reward_bonus_coeff == 0.0:
            reward = raw_reward
        else:
            reward = raw_reward + self.reward_bonus_coeff * (raw_reward - self.prev_reward_raw)
        self.prev_reward_raw = raw_reward

        self.current_turn += 1
        if next_obs.get("solved", False) or self.current_turn >= self.max_turns:
            self.done = True
            return {}, reward, self.done, self.task

        return next_obs, reward, self.done, self.task

    def get_reward_and_next_obs(self, task: Dict[str, Any], feedback: str) -> Tuple[float, Dict[str, Any]]:
        assert self.reward_fn is not None, "Provide reward_fn."
        if "</think>" in feedback:
            feedback = feedback.split("</think>")[1].strip()

        reward_output = self.reward_fn(
            data_source=task["data_source"],
            feedback=feedback,
            ground_truth=task["ground_truth"],
            problem=task["prompt"],
            prev_attempts=self.history,
            **self.reward_kwargs
        )
        reward = reward_output.reward
        metadata = reward_output.metadata or {}

        # Always use retry metadata since we're generating a new attempt based on feedback
        solver_output = metadata.get("retry_solver_code", "")
        verifier_results = metadata.get("retry_results", {})
        solver_prompt = metadata.get("retry_solver_prompt", "")
        solver_full_output = metadata.get("retry_solver_output", "")
        passed_tests = verifier_results.get("passed_tests", 0) if isinstance(verifier_results, dict) else 0
        total_tests = verifier_results.get("total_tests", 0) if isinstance(verifier_results, dict) else 0
        solved = bool(metadata.get("retry_passed", False))

        next_obs = {
            "round_idx": self.current_turn + 1,
            "problem": task["prompt"],
            "feedback": feedback,
            "solver_output": solver_output,
            "solver_full_output": solver_full_output,
            "solver_prompt": solver_prompt,
            "verifier_results": verifier_results,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "solved": solved,
        }
        self.history.append(next_obs)
        return reward, next_obs

    # ---------- helpers ----------

    @staticmethod
    def from_dict(env_args: Dict[str, Any]) -> "ContextManagerEnv":
        # Extract task from env_args - handle both cases:
        if "task" in env_args:
            task = env_args["task"]
        else:
            task = {
                "prompt": env_args.get("question", ""),
                "data_source": env_args.get("data_source", "livecodebench"),
                "ground_truth": env_args.get("ground_truth", ""),
            }

        # Ensure solver_remote has proper defaults
        solver_remote = env_args.get("solver_remote", {}) or {}
        if ("base_url" not in solver_remote) and ("base_urls" not in solver_remote):
            solver_remote["base_url"] = "http://localhost:12345/v1"
        if "model" not in solver_remote:
            solver_remote["model"] = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

        return ContextManagerEnv(
            task=task,
            max_turns=env_args.get("max_turns", 3),
            solver_generate_fn=env_args.get("solver_generate_fn"),
            reward_fn=env_args.get("reward_fn"),
            reward_kwargs=env_args.get("reward_kwargs"),
            solver_remote=solver_remote,
            solver_model_name=env_args.get("solver_model_name"),
            system_prompt=env_args.get("system_prompt", ""),
            use_shaped_reward=env_args.get("use_shaped_reward", False),
            reward_bonus_coeff=env_args.get("reward_bonus_coeff", 0.0),
            truncate_trace_chars=env_args.get("truncate_trace_chars", 2000),
        )
