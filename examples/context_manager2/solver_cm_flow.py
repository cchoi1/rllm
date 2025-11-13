from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import Workflow

###############################################################################
# Minimal interfaces you must provide from your training harness
###############################################################################

def truncatefn(s, length=300):
    """Truncate a string to a maximum length, showing middle ellipsis."""
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s
    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


def format_test_results(test_results: List[Dict], max_tests: int = 2) -> str:
    """Format test results to match code_agent.py format."""
    if not test_results:
        return "No test cases found. Please review your solution once more for correctness and efficiency, then output your final code if you're confident it's optimal."

    formatted_test_results = ""
    n_failed = 0
    for i, test in enumerate(test_results):
        if not test.get("passed", True):
            formatted_test_results += f"### Test {i + 1} failed\n"
            formatted_test_results += f"  Input: {truncatefn(test.get('input', ''))}\n"
            formatted_test_results += f"  Expected: {truncatefn(test.get('expected', ''))}\n"
            if "output" in test and test["output"] is not None:
                formatted_test_results += f"  Actual: {truncatefn(test['output'])}\n\n"
            if "error_message" in test and test["error_message"] is not None:
                formatted_test_results += f"  Error message: {truncatefn(test['error_message'])}\n"

            n_failed += 1
            if n_failed >= max_tests:
                break

    if n_failed > 0:
        return f"Here are the results on the public test cases:\n{formatted_test_results}\nSome test cases are still failing. Please carefully analyze the error patterns, revise your code to address these issues, and ensure your solution handles all the test cases correctly. Then, output your final code."
    else:
        return "Congratulations! You've successfully passed all test cases. Please carefully review your solution one more time to ensure it handles all edge cases properly. If you're confident your code is optimal, you can proceed with outputting your final solution."


@dataclass
class UnitTestResult:
    """Result returned by the DeepCoder verifier.

    Attributes
    ----------
    passed : int
        Number of tests passed by the candidate solution.
    total : int
        Total number of tests.
    feedback : str
        Textual feedback from the unit-test harness (stdout/stderr, failing
        cases, traceback, etc.). This is surfaced to the ContextManager.
    test_results : Optional[List[Dict]]
        Detailed test results list with input, expected, output, passed, error_message.
    """

    passed: int
    total: int
    feedback: str
    test_results: Optional[List[Dict]] = None

    @property
    def ratio(self) -> float:
        return 0.0 if self.total == 0 else self.passed / self.total


# Type alias: given a task dict and a solver "action" (candidate program as str),
# return a UnitTestResult.
VerifierFn = Callable[[Dict, str], UnitTestResult]


###############################################################################
# Fixed Solver (frozen weights)
###############################################################################

class DeepCoderSolver:
    """Frozen solver that proposes or refines a program.

    This class *does not* train. It uses a separate RolloutEngine that points to
    a fixed (pretrained) model. It can be called multiple times in an episode.
    """

    def __init__(self, rollout_engine: RolloutEngine, *, language: str = "python"):
        self.rollout_engine = rollout_engine
        self.language = language

    async def initial_attempt(self, problem: str, io_spec: Optional[str] = None) -> Trajectory:
        messages = [
            {
                "role": "user",
                "content": self._build_initial_solver_prompt(problem, io_spec),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        code = self._parse_code_from_response(output.content)
        return Trajectory(
            name="solver",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": getattr(output, "reasoning", None)}],
                    thought=getattr(output, "reasoning", None),
                    action=code,
                    model_response=output.content,  # Store full model response for verifier
                    model_output=output,
                    # mark non-trainable downstream (most trainers ignore this, but we add it to info just in case)
                    info={"trainable": False},
                )
            ],
        )

    async def refine_with_feedback(
        self,
        problem: str,
        previous_solution: Optional[str],
        unit_test_feedback: str,
        cm_feedback: Optional[str],
        io_spec: Optional[str] = None,
        truncation_info: Optional[str] = None,
    ) -> Trajectory:
        messages = [
            {
                "role": "user",
                "content": self._build_refine_solver_prompt(
                    problem=problem,
                    io_spec=io_spec,
                    previous_solution=previous_solution,
                    unit_test_feedback=unit_test_feedback,
                    cm_feedback=cm_feedback,
                    truncation_info=truncation_info,
                ),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        code = self._parse_code_from_response(output.content)
        return Trajectory(
            name="solver",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": getattr(output, "reasoning", None)}],
                    thought=getattr(output, "reasoning", None),
                    action=code,
                    model_response=output.content,  # Store full model response for verifier
                    model_output=output,
                    info={"trainable": False},
                )
            ],
        )

    # ---------------------- prompt & parse helpers ---------------------- #

    def _build_initial_solver_prompt(self, problem: str, io_spec: Optional[str]) -> str:
        # Match code_agent.py format: just the problem/question
        return problem

    def _build_refine_solver_prompt(
        self,
        problem: str,
        io_spec: Optional[str],
        previous_solution: Optional[str],
        unit_test_feedback: str,
        cm_feedback: Optional[str],
        truncation_info: Optional[str] = None,
    ) -> str:
        prompt = (
            "You are refining a program to pass DeepCoder unit tests. Consider the prior attempt, "
            "the failing test feedback, and the ContextManager's advice. Produce an improved *complete* program.\n\n"
            f"Problem:\n{problem}\n"
        )
        if io_spec:
            prompt += f"\nI/O Specification:\n{io_spec}\n"
        if previous_solution:
            prompt += f"\nPrevious attempt (code):\n```{self.language}\n{previous_solution}\n```\n"
        else:
            # previous_solution would only be None if we're excluding truncated, but this shouldn't happen here
            # since we're always passing the previous solution from the workflow
            prompt += "\n[Previous solution excluded due to truncation]\n"
        prompt += f"\nUnit test feedback:\n{unit_test_feedback}\n\n"
        if cm_feedback:
            prompt += f"ContextManager feedback:\n{cm_feedback}\n\n"
        else:
            if truncation_info:
                prompt += f"[ContextManager feedback excluded: {truncation_info}]\n\n"
            else:
                prompt += "[ContextManager feedback excluded due to truncation]\n\n"
        prompt += (
            "Return ONLY a single fenced code block with the full corrected program.\n"
            f"```{self.language}\n<program>\n```\n"
        )
        return prompt

    def _parse_code_from_response(self, response: str) -> str:
        # Prefer fenced code blocks
        fence = re.search(r"```[a-zA-Z0-9_\-]*\n(.*?)```", response, re.DOTALL)
        if fence:
            return fence.group(1).strip()
        # return response.strip()
        return ""


###############################################################################
# Trainable ContextManager (CM)
###############################################################################

class ContextManager:
    """Trainable agent that provides actionable feedback to the Solver.

    This class *is* trained. It uses the actor/rollout engine passed to the
    Workflow via super().__init__(rollout_engine). The CM should output
    feedback inside <feedback>...</feedback> tags.
    """

    def __init__(self, rollout_engine: RolloutEngine):
        self.rollout_engine = rollout_engine

    async def give_feedback(
        self,
        problem: str,
        previous_solution: Optional[str],
        unit_test_feedback: str,
        turn_index: int,
        io_spec: Optional[str] = None,
        truncation_info: Optional[str] = None,
    ) -> Trajectory:
        messages = [
            {
                "role": "user",
                "content": self._build_cm_prompt(
                    problem=problem,
                    io_spec=io_spec,
                    previous_solution=previous_solution,
                    unit_test_feedback=unit_test_feedback,
                    turn_index=turn_index,
                    truncation_info=truncation_info,
                ),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        # Default: if CM response is truncated or unfinished thinking, do not pass feedback
        if output.finish_reason == "length" or ("<think>" in (output.text or output.content) and "</think>" not in (output.text or output.content)):
            feedback = ""
        else:
            feedback = self._parse_feedback(output.content)
        return Trajectory(
            name="context_manager",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": getattr(output, "reasoning", None)}],
                    thought=getattr(output, "reasoning", None),
                    action=feedback,
                    model_output=output,
                )
            ],
        )

    # ---------------------- prompt & parse helpers ---------------------- #

    def _build_cm_prompt(
        self,
        problem: str,
        io_spec: Optional[str],
        previous_solution: Optional[str],
        unit_test_feedback: str,
        turn_index: int,
        truncation_info: Optional[str] = None,
    ) -> str:
        prompt = (
            "You are the ContextManager guiding a program synthesis agent. "
            "Study the problem, the solver's last attempt, and unit-test feedback. "
            "Write precise, *actionable* guidance that will most improve the next attempt.\n\n"
            "Provide guidance only; do not provide code, pseudocode, or code-like snippets.\n\n"
            f"Turn: {turn_index}\n"
            f"Problem:\n{problem}\n"
        )
        if io_spec:
            prompt += f"\nI/O Specification:\n{io_spec}\n"
        if previous_solution:
            prompt += f"\nSolver's previous program:\n```python\n{previous_solution}\n```\n\n"
        else:
            if truncation_info:
                prompt += f"\n[Solver's previous program excluded: {truncation_info}]\n\n"
            else:
                prompt += "\n[Solver's previous program excluded due to truncation]\n\n"
        prompt += f"Unit-test feedback (failures, traces, diffs, etc.):\n{unit_test_feedback}\n\n"
        prompt += (
            "Constraints:\n"
            "- Be concise (<= 10 bullet points).\n"
            "- Focus on *root causes* and concrete fixes (variable names, loops, boundary conditions, types, I/O).\n"
            "- If tests reveal specific failing cases, propose exact code changes.\n"
            "- Avoid restating the problem.\n\n"
            "Return your advice inside <feedback>...</feedback> tags only."
        )
        return prompt

    def _parse_feedback(self, response: str) -> str:
        m = re.search(r"<feedback>(.*?)</feedback>", response, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else response.strip()


###############################################################################
# Workflow: train CM only, Solver fixed, 4 CM↔Solver turns on DeepCoder
###############################################################################

class DeepCoderCMWorkflow(Workflow):
    """Workflow that *trains only the ContextManager* on DeepCoder.

    - The Solver uses a fixed RolloutEngine (frozen weights).\n
    - The ContextManager uses the trainable engine provided to Workflow.__init__.\n
    - Each episode runs up to `n_turns` CM↔Solver cycles:\n
        1) Solver attempts a program.\n        2) Verifier runs tests and produces textual feedback.\n        3) CM gives feedback based on (problem, previous program, test feedback).\n        4) Solver refines using CM advice.\n
      The reward for each CM step is the *delta in unit-test pass ratio* between
      the solver attempt after CM advice and the solver attempt before CM advice.\n
    Notes
    -----
    * Ensure your trainer only updates the actor tied to the CM (e.g., filter by
      trajectory name == 'context_manager' or ignore trajectories with
      metadata.trainable == False).\n
    * Initialize the CM from the DeepCoder-1.5b checkpoint in your config. The
      Solver should also point to the pretrained checkpoint but its engine must
      be excluded from training.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine = None,
        *,
        solver_engine: RolloutEngine = None,
        verifier: VerifierFn,
        reward_function: Optional[RewardFunction] = None,
        n_turns: int = 4,
        language: str = "python",
        cm_engine: RolloutEngine = None,
        exclude_truncated: bool = False,
        **kwargs,
    ) -> None:
        # Handle both rollout_engine (from AgentWorkflowEngine) and cm_engine (from workflow_args)
        cm_engine = cm_engine or rollout_engine
        if cm_engine is None:
            raise ValueError("Either rollout_engine or cm_engine must be provided")
        
        # During training, if solver_engine is not provided, use rollout_engine
        # The solver will be marked as non-trainable via info={"trainable": False}
        if solver_engine is None:
            solver_engine = rollout_engine
        
        super().__init__(cm_engine, **kwargs)  # this is the *trainable* engine
        self.n_turns = n_turns
        self.verifier = verifier
        self.reward_function = reward_function  # optional override
        self.solver = DeepCoderSolver(solver_engine, language=language)
        self.cm = ContextManager(cm_engine)
        self.exclude_truncated = exclude_truncated
        # Store max_response_length from engines for truncation messages
        self.solver_max_response_length = getattr(solver_engine, 'max_response_length', None)
        self.cm_max_response_length = getattr(cm_engine, 'max_response_length', None)

    # ---------------------- core episode logic ---------------------- #

    async def run(self, task: Dict, uid: str, **kwargs) -> Episode:
        self.reset(task, uid)

        # Expect these keys in DeepCoder tasks (adjust if your loader differs)
        problem: str = task.get("question") or task.get("problem") or task["prompt"]
        io_spec: Optional[str] = task.get("io_spec")  # optional (depends on dataset view)

        trajectories: List[Trajectory] = []
        metrics: Dict[str, float] = {}

        # Initial solver attempt
        solver_traj = await self.solver.initial_attempt(problem, io_spec)
        trajectories.append(solver_traj)

        # Evaluate - pass the full model response, not just extracted code
        # code_reward_fn expects the full model response to extract code from markdown
        model_response = solver_traj.steps[0].model_response if hasattr(solver_traj.steps[0], 'model_response') and solver_traj.steps[0].model_response else solver_traj.steps[0].action
        result_prev = self.verifier(task, model_response)
        metrics[f"pass_ratio_t0"] = result_prev.ratio
        metrics[f"passed_t0"] = float(result_prev.passed)
        metrics[f"total_tests"] = float(result_prev.total)

        # Early stop if already perfect
        if result_prev.ratio >= 1.0:
            return Episode(
                id=uid,
                task=task,
                trajectories=trajectories,  # no CM steps; nothing to train on here
                is_correct=True,
                metrics=metrics,
            )

        # Interleave CM feedback and solver refinements
        for t in range(1, self.n_turns + 1):
            # Format test results like code_agent.py
            formatted_test_feedback = format_test_results(result_prev.test_results) if result_prev.test_results else result_prev.feedback
            
            # Get the most recent solution (from previous iteration)
            previous_solution = solver_traj.steps[-1].action
            # Check if previous solution was truncated and exclude if flag is set
            truncation_info_solver = None
            if self.exclude_truncated:
                if hasattr(solver_traj.steps[-1], 'model_output') and solver_traj.steps[-1].model_output:
                    model_output = solver_traj.steps[-1].model_output
                    if model_output.finish_reason == "length":
                        # Response was truncated - exclude from CM prompt but provide truncation info
                        tokens_generated = model_output.completion_length
                        max_token_limit = self.solver_max_response_length
                        truncation_info_solver = f"Incomplete response (truncated at {tokens_generated} tokens, limit: {max_token_limit})"
                        previous_solution = None  # Will be handled in CM prompt building
            
            # CM feedback (trainable step)
            cm_traj = await self.cm.give_feedback(
                problem=problem,
                previous_solution=previous_solution,
                unit_test_feedback=formatted_test_feedback,
                turn_index=t,
                io_spec=io_spec,
                truncation_info=truncation_info_solver,  # Pass truncation info if available
            )
            trajectories.append(cm_traj)

            # Solver refines using CM advice
            # Note: Solver uses formatted test feedback (matching code_agent.py), CM feedback is ignored for prompt matching
            cm_feedback = cm_traj.steps[-1].action
            # Check if CM feedback was truncated and exclude if flag is set
            truncation_info_cm = None
            if self.exclude_truncated:
                if hasattr(cm_traj.steps[-1], 'model_output') and cm_traj.steps[-1].model_output:
                    model_output = cm_traj.steps[-1].model_output
                    if model_output.finish_reason == "length":
                        # Response was truncated - exclude from Solver prompt but provide truncation info
                        tokens_generated = model_output.completion_length
                        max_token_limit = self.cm_max_response_length
                        truncation_info_cm = f"Incomplete response (truncated at {tokens_generated} tokens, limit: {max_token_limit})"
                        cm_feedback = None  # Will be handled in Solver prompt building
            
            solver_traj_next = await self.solver.refine_with_feedback(
                problem=problem,
                previous_solution=previous_solution,
                unit_test_feedback=formatted_test_feedback,
                cm_feedback=cm_feedback,
                io_spec=io_spec,
                truncation_info=truncation_info_cm,  # Pass truncation info if available
            )
            trajectories.append(solver_traj_next)

            # Evaluate the refined program - pass the full model response, not just extracted code
            model_response = solver_traj_next.steps[-1].model_response if hasattr(solver_traj_next.steps[-1], 'model_response') and solver_traj_next.steps[-1].model_response else solver_traj_next.steps[-1].action
            result_next = self.verifier(task, model_response)

            # Reward: delta pass ratio (after CM vs before CM)
            delta = result_next.ratio - result_prev.ratio
            cm_traj.steps[-1].reward = float(delta)

            # Update rolling state & metrics
            metrics[f"delta_t{t}"] = float(delta)
            metrics[f"pass_ratio_t{t}"] = result_next.ratio
            metrics[f"passed_t{t}"] = float(result_next.passed)

            solver_traj = solver_traj_next
            result_prev = result_next

            # Optional early stop on full pass
            if result_prev.ratio >= 1.0:
                break

        is_correct = result_prev.ratio >= 1.0
        return Episode(
            id=uid,
            task=task,
            trajectories=trajectories,
            is_correct=is_correct,
            metrics=metrics,
        )