from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.rewards.math_reward import rllm_reward_fn_math
from rllm.workflows.workflow import Workflow

logger = logging.getLogger(__name__)

###############################################################################
# Minimal interfaces you must provide from your training harness
###############################################################################


@dataclass
class MathResult:
    """Result returned by the math verifier.

    Attributes
    ----------
    is_correct : bool
        Whether the solution is correct.
    reward : float
        Reward value from the math reward function.
    feedback : str
        Textual feedback from the verifier (e.g., correctness status, hints).
    metadata : dict
        Additional metadata from the reward function.
    """

    is_correct: bool
    reward: float
    feedback: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# Type alias: given a task dict and a solver "action" (candidate solution as str),
# return a MathResult.
VerifierFn = Callable[[Dict, str], MathResult]


def format_math_feedback(result: MathResult) -> str:
    """Format math result into feedback string for the ContextManager."""
    if result.is_correct:
        return "Congratulations! Your solution is correct. The answer matches the expected result."
    else:
        feedback = "Your solution is incorrect. "
        
        # Add hints if available in metadata
        if result.metadata:
            error_hint = result.metadata.get("error_hint", "")
            if error_hint:
                feedback += f"\nHint: {error_hint}"
        
        feedback += "\nPlease carefully review your calculations and reasoning. Check for:\n"
        feedback += "- Calculation errors\n"
        feedback += "- Logic errors in your step-by-step reasoning\n"
        feedback += "- Missing steps or incorrect intermediate values\n"
        feedback += "- Answer format issues\n"
        
        return feedback


###############################################################################
# Fixed Solver (frozen weights)
###############################################################################


class MathSolver:
    """Frozen solver that proposes or refines a math solution.

    This class *does not* train. It uses a separate RolloutEngine that points to
    a fixed (pretrained) model. It can be called multiple times in an episode.
    """

    def __init__(self, rollout_engine: RolloutEngine, *, use_tools: bool = True, max_previous_solution_length: int = 2000):
        self.rollout_engine = rollout_engine
        self.use_tools = use_tools
        self.max_previous_solution_length = max_previous_solution_length

    def _get_prompt_length(self, messages: list[dict]) -> tuple[int, int]:
        """Get prompt length in characters and tokens.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
        
        Returns:
            (char_count, token_count) tuple. token_count is -1 if tokenizer unavailable.
        """
        # Calculate character length from messages
        char_count = sum(len(str(msg.get("content", ""))) for msg in messages)
        
        # Try to get accurate token count using chat_parser (same as what's actually used)
        token_count = -1
        try:
            if hasattr(self.rollout_engine, 'chat_parser') and self.rollout_engine.chat_parser is not None:
                # Use the same parsing logic as the engine
                prompt = self.rollout_engine.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
                if hasattr(self.rollout_engine, 'tokenizer') and self.rollout_engine.tokenizer is not None:
                    tokens = self.rollout_engine.tokenizer.encode(prompt, add_special_tokens=False)
                    token_count = len(tokens)
                    char_count = len(prompt)  # Use actual parsed prompt length
        except Exception:
            # Fallback to simple encoding if chat_parser fails
            try:
                if hasattr(self.rollout_engine, 'tokenizer') and self.rollout_engine.tokenizer is not None:
                    # Simple approximation: encode the content strings
                    content_str = " ".join(str(msg.get("content", "")) for msg in messages)
                    tokens = self.rollout_engine.tokenizer.encode(content_str, add_special_tokens=False)
                    token_count = len(tokens)
            except Exception:
                pass
        return char_count, token_count

    async def initial_attempt(self, problem: str, uid: str = None, ground_truth: Optional[str | float] = None) -> Trajectory:
        prompt = self._build_initial_solver_prompt(problem)
        messages = [{"role": "user", "content": prompt}]
        char_len, token_len = self._get_prompt_length(messages)
        uid_str = f"uid={uid} " if uid else ""
        logger.info(f"[Solver Initial] {uid_str}chars={char_len}, tokens={token_len if token_len >= 0 else 'N/A'}")
        
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        solution = self._parse_solution_from_response(output.content)
        
        # Extract numeric answer and check correctness
        numeric_answer = self._extract_numeric_answer(solution)
        is_correct = self._check_correctness(numeric_answer, ground_truth)
        
        info = {"trainable": False}
        if numeric_answer is not None:
            info["numeric_answer"] = numeric_answer
        if is_correct is not None:
            info["is_correct"] = is_correct
        
        return Trajectory(
            name="solver",
            steps=[
                Step(
                    chat_completions=messages + [{"role": "assistant", "content": output.content, "reasoning": getattr(output, "reasoning", None)}],
                    thought=getattr(output, "reasoning", None),
                    action=solution,
                    model_output=output,
                    info=info,
                )
            ],
        )

    async def refine_with_feedback(
        self,
        problem: str,
        previous_solution: str,
        math_feedback: str,
        cm_feedback: str,
        turn_index: int = None,
        uid: str = None,
        ground_truth: Optional[str | float] = None,
    ) -> Trajectory:
        prompt = self._build_refine_solver_prompt(
            problem=problem,
            previous_solution=previous_solution,
            math_feedback=math_feedback,
            cm_feedback=cm_feedback,
            max_previous_solution_length=self.max_previous_solution_length,
        )
        messages = [{"role": "user", "content": prompt}]
        char_len, token_len = self._get_prompt_length(messages)
        uid_str = f"uid={uid} " if uid else ""
        turn_str = f"turn={turn_index} " if turn_index is not None else ""
        prev_sol_len = len(previous_solution)
        logger.info(f"[Solver Refine] {uid_str}{turn_str}chars={char_len}, tokens={token_len if token_len >= 0 else 'N/A'}, prev_solution_len={prev_sol_len}")
        
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        solution = self._parse_solution_from_response(output.content)
        
        # Extract numeric answer and check correctness
        numeric_answer = self._extract_numeric_answer(solution)
        is_correct = self._check_correctness(numeric_answer, ground_truth)
        
        info = {"trainable": False}
        if numeric_answer is not None:
            info["numeric_answer"] = numeric_answer
        if is_correct is not None:
            info["is_correct"] = is_correct
        
        return Trajectory(
            name="solver",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": getattr(output, "reasoning", None)}],
                    thought=getattr(output, "reasoning", None),
                    action=solution,
                    model_output=output,
                    info=info,
                )
            ],
        )

    # ---------------------- prompt & parse helpers ---------------------- #

    def _build_initial_solver_prompt(self, problem: str) -> str:
        prompt = problem
        if self.use_tools:
            prompt += "\n\nYou can use Python tools to help with calculations. Show your reasoning step by step and provide the final answer."
        return prompt

    def _build_refine_solver_prompt(
        self,
        problem: str,
        previous_solution: str,
        math_feedback: str,
        cm_feedback: str,
        max_previous_solution_length: int = 2000,
    ) -> str:
        # Truncate previous solution to prevent prompt length explosion
        # Keep the end of the solution (most recent reasoning) if it's too long
        if len(previous_solution) > max_previous_solution_length:
            truncated_solution = "..." + previous_solution[-max_previous_solution_length:]
        else:
            truncated_solution = previous_solution
            
        return (
            "You are refining your solution to a math problem. Consider the previous attempt, "
            "the correctness feedback, and the ContextManager's advice. Produce an improved solution.\n\n"
            f"Problem:\n{problem}\n\n"
            f"Previous attempt:\n{truncated_solution}\n\n"
            f"Correctness feedback:\n{math_feedback}\n\n"
            f"ContextManager feedback:\n{cm_feedback}\n\n"
            "Please provide a complete, corrected solution with step-by-step reasoning."
            + (" You can use Python tools to help with calculations." if self.use_tools else "")
        )

    def _parse_solution_from_response(self, response: str) -> str:
        # Return the full response as the solution
        # The verifier will extract the answer from it
        return response.strip()
    
    def _extract_numeric_answer(self, solution: str) -> Optional[float]:
        """Extract numeric answer from solution string."""
        num_match = re.search(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", solution)
        if num_match:
            try:
                return float(num_match.group(0))
            except Exception:
                pass
        return None
    
    def _check_correctness(self, predicted_answer: Optional[float], ground_truth: Optional[str | float]) -> Optional[bool]:
        """Check if predicted answer matches ground truth (within tolerance)."""
        if predicted_answer is None or ground_truth is None:
            return None
        
        # Convert ground_truth to float
        try:
            if isinstance(ground_truth, (int, float)):
                true_answer = float(ground_truth)
            else:
                num_match = re.search(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", str(ground_truth))
                if not num_match:
                    return None
                true_answer = float(num_match.group(0))
        except Exception:
            return None
        
        # Check if answers match within tolerance
        tol = 1e-6
        if predicted_answer != predicted_answer or true_answer != true_answer:  # Check for NaN
            return None
        return abs(predicted_answer - true_answer) <= tol


###############################################################################
# Trainable ContextManager (CM)
###############################################################################


class ContextManager:
    """Trainable agent that provides actionable feedback to the Solver.

    This class *is* trained. It uses the actor/rollout engine passed to the
    Workflow via super().__init__(rollout_engine). The CM should output
    feedback inside <feedback>...</feedback> tags.
    """

    def __init__(self, rollout_engine: RolloutEngine, max_previous_solution_length: int = 2000):
        self.rollout_engine = rollout_engine
        self.max_previous_solution_length = max_previous_solution_length

    def _get_prompt_length(self, messages: list[dict]) -> tuple[int, int]:
        """Get prompt length in characters and tokens.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
        
        Returns:
            (char_count, token_count) tuple. token_count is -1 if tokenizer unavailable.
        """
        # Calculate character length from messages
        char_count = sum(len(str(msg.get("content", ""))) for msg in messages)
        
        # Try to get accurate token count using chat_parser (same as what's actually used)
        token_count = -1
        try:
            if hasattr(self.rollout_engine, 'chat_parser') and self.rollout_engine.chat_parser is not None:
                # Use the same parsing logic as the engine
                prompt = self.rollout_engine.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
                if hasattr(self.rollout_engine, 'tokenizer') and self.rollout_engine.tokenizer is not None:
                    tokens = self.rollout_engine.tokenizer.encode(prompt, add_special_tokens=False)
                    token_count = len(tokens)
                    char_count = len(prompt)  # Use actual parsed prompt length
        except Exception:
            # Fallback to simple encoding if chat_parser fails
            try:
                if hasattr(self.rollout_engine, 'tokenizer') and self.rollout_engine.tokenizer is not None:
                    # Simple approximation: encode the content strings
                    content_str = " ".join(str(msg.get("content", "")) for msg in messages)
                    tokens = self.rollout_engine.tokenizer.encode(content_str, add_special_tokens=False)
                    token_count = len(tokens)
            except Exception:
                pass
        return char_count, token_count

    async def give_feedback(
        self,
        problem: str,
        previous_solution: str,
        math_feedback: str,
        turn_index: int,
        uid: str = None,
    ) -> Trajectory:
        prompt = self._build_cm_prompt(
            problem=problem,
            previous_solution=previous_solution,
            math_feedback=math_feedback,
            turn_index=turn_index,
            max_previous_solution_length=self.max_previous_solution_length,
        )
        messages = [{"role": "user", "content": prompt}]
        char_len, token_len = self._get_prompt_length(messages)
        uid_str = f"uid={uid} " if uid else ""
        prev_sol_len = len(previous_solution)
        logger.info(f"[ContextManager] {uid_str}turn={turn_index} chars={char_len}, tokens={token_len if token_len >= 0 else 'N/A'}, prev_solution_len={prev_sol_len}")
        
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        # Default: if CM response is truncated or unfinished thinking, do not pass feedback
        if output.finish_reason == "length" or ("<think>" in (output.text or output.content) and "</think>" not in (output.text or output.content)):
            feedback = ""
        else:
            feedback = self._parse_feedback(output.content)
        return Trajectory(
            # name="context_manager",
            name="agent",   # trainer-friendly name
            steps=[
                Step(
                    chat_completions=messages + [{"role": "assistant", "content": output.content, "reasoning": getattr(output, "reasoning", None)}],
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
        previous_solution: str,
        math_feedback: str,
        turn_index: int,
        max_previous_solution_length: int = 2000,
    ) -> str:
        # Truncate previous solution to prevent prompt length explosion
        if len(previous_solution) > max_previous_solution_length:
            truncated_solution = "..." + previous_solution[-max_previous_solution_length:]
        else:
            truncated_solution = previous_solution
            
        return (
            "You are the ContextManager guiding a math problem-solving agent. "
            "Study the problem, the solver's last attempt, and correctness feedback. "
            "Write precise, *actionable* guidance that will most improve the next attempt.\n\n"
            f"Turn: {turn_index}\n"
            f"Problem:\n{problem}\n\n"
            f"Solver's previous solution:\n{truncated_solution}\n\n"
            f"Correctness feedback:\n{math_feedback}\n\n"
            + (
                "Constraints:\n"
                "- Be concise (<= 10 bullet points).\n"
                "- Focus on *root causes* and concrete fixes (calculation errors, logic issues, missing steps, format issues).\n"
                "- If the solution is incorrect, identify specific errors and suggest corrections.\n"
                "- Avoid restating the problem.\n\n"
                "Return your advice inside <feedback>...</feedback> tags only."
            )
        )

    def _parse_feedback(self, response: str) -> str:
        m = re.search(r"<feedback>(.*?)</feedback>", response, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else response.strip()


###############################################################################
# Workflow: train CM only, Solver fixed, N CM↔Solver turns on Math
###############################################################################


class MathCMWorkflow(Workflow):
    """Workflow that *trains only the ContextManager* on Math problems.

    - The Solver uses a fixed RolloutEngine (frozen weights).\n
    - The ContextManager uses the trainable engine provided to Workflow.__init__.\n
    - Each episode runs up to `n_turns` CM↔Solver cycles:\n
        1) Solver attempts a solution.\n
        2) Verifier evaluates the solution.\n
        3) CM gives feedback based on (problem, previous solution, correctness feedback).\n
        4) Solver refines using CM advice.\n
      The reward for each CM step is the *delta in correctness* (0->1 or 0->0) between
      the solver attempt after CM advice and the solver attempt before CM advice.\n
    Notes
    -----
    * Ensure your trainer only updates the actor tied to the CM (e.g., filter by
      trajectory name == 'context_manager' or ignore trajectories with
      metadata.trainable == False).\n
    * Initialize the CM from a pretrained checkpoint. The
      Solver should also point to a pretrained checkpoint but its engine must
      be excluded from training.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine = None,
        *,
        solver_engine: RolloutEngine = None,
        verifier: VerifierFn = None,
        reward_function: Optional[RewardFunction] = None,
        n_turns: int = 4,
        use_tools: bool = True,
        cm_engine: RolloutEngine = None,
        **kwargs,
    ) -> None:
        cm_engine = rollout_engine
        if cm_engine is None:
            raise ValueError("Need a trainable CM engine")

        if solver_engine is None:
            solver_engine = cm_engine 
        
        # For training, if verifier is not provided, create a default one
        if verifier is None:
            def default_verifier(task: dict, solution: str) -> MathResult:
                # Simple default verifier - just check if reward is positive
                data_source = task.get("data_source", "math")
                ground_truth = task.get("ground_truth")
                if reward_function is None:
                    reward_output = rllm_reward_fn_math(data_source, solution, ground_truth)
                else:
                    task_info = {"ground_truth": ground_truth}
                    reward_output = reward_function(task_info, solution)
                return MathResult(
                    is_correct=reward_output.is_correct,
                    reward=reward_output.reward,
                    feedback=reward_output.metadata.get("feedback", "") if reward_output.metadata else "",
                    metadata=reward_output.metadata or {},
                )
            verifier = default_verifier
        
        super().__init__(cm_engine, **kwargs)  # this is the *trainable* engine
        self.n_turns = n_turns
        self.verifier = verifier
        self.reward_function = reward_function  # optional override
        # Control prompt length by limiting previous solution length (in characters)
        # Default 2000 chars ~= 500-700 tokens, adjust based on your model's context window
        max_previous_solution_length = kwargs.get("max_previous_solution_length", 2000)
        self.solver = MathSolver(solver_engine, use_tools=use_tools, max_previous_solution_length=max_previous_solution_length)
        self.cm = ContextManager(cm_engine, max_previous_solution_length=max_previous_solution_length)

    # ---------------------- core episode logic ---------------------- #

    async def run(self, task: Dict, uid: str, **kwargs) -> Episode:
        self.reset(task, uid)

        # Expect these keys in Math tasks
        problem: str = task.get("question") or task.get("problem") or task.get("prompt", "")
        ground_truth = task.get("ground_truth")

        trajectories: List[Trajectory] = []
        metrics: Dict[str, float] = {}

        # Initial solver attempt
        solver_traj = await self.solver.initial_attempt(problem, uid=uid, ground_truth=ground_truth)
        # Explicitly ensure initial solver trajectory steps have 0 reward (solver is not trained)
        for step in solver_traj.steps:
            step.reward = 0.0
        trajectories.append(solver_traj)

        # Evaluate
        result_prev = self.verifier(task, solver_traj.steps[0].action)
        # Store verifier result metadata in step info
        if solver_traj.steps:
            solver_traj.steps[0].info = solver_traj.steps[0].info or {}
            if result_prev.metadata:
                solver_traj.steps[0].info["verifier_metadata"] = result_prev.metadata

        # Track correctness at turn 0 (initial attempt)
        metrics["is_correct_t0"] = 1.0 if result_prev.is_correct else 0.0

        # Early stop if already correct
        if result_prev.is_correct:
            metrics["pass@1"] = 1.0
            # Fill in remaining turns as correct (since we're done)
            for t in range(1, self.n_turns + 1):
                metrics[f"is_correct_t{t}"] = 1.0
            episode = Episode(
                id=uid,
                task=task,
                trajectories=trajectories,
                is_correct=True,
                metrics=metrics,
            )
            # self.collect_metrics(episode)
            return episode

        # Interleave CM feedback and solver refinements
        for t in range(1, self.n_turns + 1):
            # Format feedback
            formatted_feedback = format_math_feedback(result_prev)
            
            # Get the most recent solution (from previous iteration)
            previous_solution = solver_traj.steps[-1].action
            
            # CM feedback (trainable step)
            cm_traj = await self.cm.give_feedback(
                problem=problem,
                previous_solution=previous_solution,
                math_feedback=formatted_feedback,
                turn_index=t,
                uid=uid,
            )
            trajectories.append(cm_traj)

            # Solver refines using CM advice
            solver_traj_next = await self.solver.refine_with_feedback(
                problem=problem,
                previous_solution=previous_solution,
                math_feedback=formatted_feedback,
                cm_feedback=cm_traj.steps[-1].action,
                turn_index=t,
                uid=uid,
                ground_truth=ground_truth,
            )
            trajectories.append(solver_traj_next)

            # Evaluate the refined solution
            result_next = self.verifier(task, solver_traj_next.steps[-1].action)
            # Store verifier result metadata in step info
            if solver_traj_next.steps:
                solver_traj_next.steps[-1].info = solver_traj_next.steps[-1].info or {}
                if result_next.metadata:
                    solver_traj_next.steps[-1].info["verifier_metadata"] = result_next.metadata

            # Track correctness at this turn (after refinement)
            metrics[f"is_correct_t{t}"] = 1.0 if result_next.is_correct else 0.0

            # Reward: delta correctness (after CM vs before CM)
            # If result_prev was incorrect (0) and result_next is correct (1), delta = 1
            # If result_prev was incorrect (0) and result_next is still incorrect (0), delta = 0
            # If result_prev was correct, we wouldn't be in this loop
            delta_reward = result_next.reward - result_prev.reward
            
            # Set reward on CM trajectory step (trainable step)
            cm_traj.steps[-1].reward = float(delta_reward)
            
            # Explicitly ensure solver trajectory steps have 0 reward (solver is not trained)
            for step in solver_traj_next.steps:
                if not hasattr(step, 'reward') or step.reward is None:
                    step.reward = 0.0
                else:
                    step.reward = 0.0  # Ensure solver steps have 0 reward

            solver_traj = solver_traj_next
            result_prev = result_next

            # Optional early stop on correct solution
            if result_prev.is_correct:
                # Fill in remaining turns as correct (since we're done)
                for future_turn in range(t + 1, self.n_turns + 1):
                    metrics[f"is_correct_t{future_turn}"] = 1.0
                break

        is_correct = result_prev.is_correct
        # Populate metrics before passing to Episode
        metrics["pass@1"] = 1.0 if is_correct else 0.0
        
        # Fill in any missing turn metrics (if we didn't complete all turns and didn't break early)
        # This handles the case where we completed all turns without becoming correct
        for t in range(1, self.n_turns + 1):
            if f"is_correct_t{t}" not in metrics:
                metrics[f"is_correct_t{t}"] = 0.0
        
        episode = Episode(
            id=uid,
            task=task,
            trajectories=trajectories,
            is_correct=is_correct,
            metrics=metrics,
        )
        # self.collect_metrics(episode)
        return episode

###############################################################################
# New Workflow: CM sums correct answers over N turns (with Solver)
###############################################################################


class SumContextManager:
    """Trainable CM that maintains a running sum across N problems shown one-by-one.
    
    At each turn t, CM sees:
      - the current problem P_t
      - the Solver's solution and whether it's correct
      - the running sum from previous turns S_{t-1}
    If the Solver was correct, CM just adds that answer to the sum.
    If the Solver was incorrect/incomplete, CM may reattempt the problem, then add the correct answer.
    It must output the updated sum S_t inside <sum>...</sum>.
    """

    SUM_RE = re.compile(r"<sum>(.*?)</sum>", re.IGNORECASE | re.DOTALL)

    def __init__(self, rollout_engine: RolloutEngine, *, use_tools: bool = True):
        self.rollout_engine = rollout_engine
        self.use_tools = use_tools

    def _get_prompt_length(self, messages: list[dict]) -> tuple[int, int]:
        char_count = sum(len(str(m.get("content", ""))) for m in messages)
        token_count = -1
        try:
            if hasattr(self.rollout_engine, "chat_parser") and self.rollout_engine.chat_parser is not None:
                prompt = self.rollout_engine.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
                if hasattr(self.rollout_engine, "tokenizer") and self.rollout_engine.tokenizer is not None:
                    tokens = self.rollout_engine.tokenizer.encode(prompt, add_special_tokens=False)
                    token_count = len(tokens)
                    char_count = len(prompt)
        except Exception:
            try:
                if hasattr(self.rollout_engine, "tokenizer") and self.rollout_engine.tokenizer is not None:
                    content_str = " ".join(str(msg.get("content", "")) for msg in messages)
                    tokens = self.rollout_engine.tokenizer.encode(content_str, add_special_tokens=False)
                    token_count = len(tokens)
            except Exception:
                pass
        return char_count, token_count

    def _build_cm_sum_prompt(
        self,
        *,
        turn_index: int,
        problem: str,
        solver_solution: str,
        solver_is_correct: bool,
        prev_sum: float | int,
    ) -> str:
        prompt = (
            f"Your task is to provide the updated cumulative sum of all correct answers inside <sum>...</sum>."
            f"Problem {turn_index}:\n{problem}\n\n"
            # f"Solver's solution is correct: {solver_is_correct}\n\n"
            f"Solver's solution:\n{solver_solution}\n\n"
            f"Cumulative sum so far: <sum>{prev_sum}</sum>.\n\n"
            f"Given the Solver's solution, provide the updated cumulative sum of all correct answers inside <sum>...</sum>.\n\n"
            f"Example:\n"
            f"Problem: 1 + 1 = ?\n"
            f"Solver's solution: 2\n"
            # f"Solver's solution is correct: True\n"
            f"Cumulative sum so far: <sum>10</sum>.\n"
            f"Updated cumulative sum: <sum>12</sum>.\n\n"
        )
        return prompt

    def _parse_sum(self, response: str) -> Optional[float]:
        m = self.SUM_RE.search(response)
        text = m.group(1).strip() if m else response.strip()
        num_match = re.search(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", text)
        if not num_match:
            return None
        try:
            return float(num_match.group(0))
        except Exception:
            return None

    async def update_sum(
        self,
        *,
        turn_index: int,
        problem: str,
        solver_solution: str,
        solver_is_correct: bool,
        prev_sum: float | int,
        uid: str = None,
    ) -> Trajectory:
        prompt = self._build_cm_sum_prompt(
            turn_index=turn_index,
            problem=problem,
            solver_solution=solver_solution,
            solver_is_correct=solver_is_correct,
            prev_sum=prev_sum,
        )
        print(f"\n\nCM Prompt: {prompt}")
        messages = [{"role": "user", "content": prompt}]
        char_len, token_len = self._get_prompt_length(messages)
        uid_str = f"uid={uid} " if uid else ""
        solver_status = "correct" if solver_is_correct else "incorrect"
        logger.info(
            f"[CM-Sum] {uid_str}turn={turn_index} chars={char_len}, tokens={token_len if token_len >= 0 else 'N/A'}, prev_sum={prev_sum}, solver={solver_status}"
        )
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        parsed = self._parse_sum(output.content)
        new_sum = parsed if parsed is not None and output.finish_reason != "length" else prev_sum
        return Trajectory(
            name="agent",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": getattr(output, "reasoning", None)}],
                    thought=getattr(output, "reasoning", None),
                    action=f"<sum>{new_sum}</sum>",
                    model_output=output,
                )
            ],
        )


class SumOfProblemsWorkflow(Workflow):
    """Workflow where the CM must return the sum of correct answers over N problems shown sequentially.
    
    Expected task format:
      - task['problems']: List[str]         length N
      - task['ground_truths']: List[float]  length N (numeric answers)
    
    Behavior:
      - On each turn t in 1..N:
          1. Solver attempts problem t
          2. Verifier checks if Solver's solution is correct
          3. CM sees: problem t, Solver's solution, correctness status, and running total
          4. If Solver was correct: CM just adds the answer to the sum
          5. If Solver was incorrect/incomplete: CM reattempts the problem, then adds the correct answer
          6. CM outputs the updated running total in <sum>...</sum>
      - Final reward assigned to the last CM step:
          1.0 if final running total equals true sum (within tolerance), else 0.0.
      - All earlier steps have reward 0.0 (sparse reward).
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        *,
        solver_engine: RolloutEngine = None,
        verifier: VerifierFn = None,
        n_turns: int | None = None,
        use_tools: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(rollout_engine, **kwargs)
        if solver_engine is None:
            solver_engine = rollout_engine
        if verifier is None:
            # Create a default verifier that extracts numeric answer from solution
            def default_verifier(task: dict, solution: str) -> MathResult:
                # Extract numeric answer from solution
                num_match = re.search(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", solution)
                if num_match:
                    try:
                        pred_answer = float(num_match.group(0))
                        # For sum workflow, we need to check against ground_truths
                        # This is a simplified verifier - the workflow will handle the actual checking
                        return MathResult(
                            is_correct=True,  # Will be checked in workflow
                            reward=1.0,
                            feedback="",
                            metadata={"predicted_answer": pred_answer},
                        )
                    except Exception:
                        pass
                return MathResult(is_correct=False, reward=0.0, feedback="Could not extract numeric answer", metadata={})
            verifier = default_verifier
        
        self.solver = MathSolver(solver_engine, use_tools=use_tools)
        self.cm = SumContextManager(rollout_engine, use_tools=use_tools)
        self.verifier = verifier
        self.n_turns_override = n_turns
    
    async def run(self, task: Dict, uid: str, **kwargs) -> Episode:
        self.reset(task, uid)

        problems: List[str] = task.get("problems") or []
        ground_truths: List = task.get("ground_truths") or []
        if not problems or not ground_truths or len(problems) != len(ground_truths):
            raise ValueError("Task must include 'problems' and 'ground_truths' lists of equal non-zero length.")

        N = self.n_turns_override or len(problems)
        N = min(N, len(problems))

        trajectories: List[Trajectory] = []
        running_sum: float = 0.0

        def _to_float(x) -> float:
            """Convert various formats to float."""
            try:
                if isinstance(x, (int, float)):
                    return float(x)
                m = re.search(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", str(x))
                return float(m.group(0)) if m else float("nan")
            except Exception:
                return float("nan")

        for t in range(1, N + 1):
            problem = problems[t - 1]
            ground_truth = ground_truths[t - 1]
            true_answer = _to_float(ground_truth)
            
            # 1. Solver attempts the problem
            solver_traj = await self.solver.initial_attempt(problem, uid=uid, ground_truth=ground_truth)
            solver_solution = solver_traj.steps[0].action
            # Mark solver steps as non-trainable with 0 reward, but preserve existing info
            for step in solver_traj.steps:
                step.reward = 0.0
                step.info = step.info or {}
                step.info["trainable"] = False
            trajectories.append(solver_traj)
            
            # 2. Verify solver solution
            # Create a task dict for verifier (it expects 'ground_truth' key)
            verifier_task = {"ground_truth": ground_truth, "question": problem}
            verifier_result = self.verifier(verifier_task, solver_solution)
            # Store verifier result metadata in step info
            if solver_traj.steps:
                solver_traj.steps[0].info = solver_traj.steps[0].info or {}
                if verifier_result.metadata:
                    solver_traj.steps[0].info["verifier_metadata"] = verifier_result.metadata
            
            # Check if solver's answer matches ground truth (within tolerance)
            solver_pred_answer = _to_float(solver_solution)
            tol = 1e-6
            solver_is_correct = (
                abs(solver_pred_answer - true_answer) <= tol
                if not (solver_pred_answer != solver_pred_answer or true_answer != true_answer)
                else False
            )
            # Also use verifier result if available
            if hasattr(verifier_result, "is_correct"):
                solver_is_correct = solver_is_correct or verifier_result.is_correct
            
            # 3. CM updates sum based on solver solution
            cm_traj = await self.cm.update_sum(
                turn_index=t,
                problem=problem,
                solver_solution=solver_solution,
                solver_is_correct=solver_is_correct,
                prev_sum=running_sum,
                uid=uid,
            )
            trajectories.append(cm_traj)
            
            # 4. Extract updated sum from CM output
            action_text = cm_traj.steps[-1].action or ""
            m = re.search(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", action_text)
            if m:
                try:
                    running_sum = float(m.group(0))
                except Exception:
                    pass
            cm_traj.steps[-1].reward = 0.0  # Will be set to final reward on last turn

        # Calculate true sum and final reward
        true_sum = sum(_to_float(gt) for gt in ground_truths[:N])
        tol = 1e-6
        is_correct = (abs(running_sum - true_sum) <= tol) if (not (true_sum != true_sum or running_sum != running_sum)) else False
        final_reward = 1.0 if is_correct else 0.0
        if trajectories:
            trajectories[-1].steps[-1].reward = float(final_reward)

        metrics = {
            "pred_final_sum": float(running_sum),
            "true_final_sum": float(true_sum) if true_sum == true_sum else 0.0,
            "pass@1": 1.0 if is_correct else 0.0,
            "n_turns": float(N),
        }

        episode = Episode(
            id=uid,
            task=task,
            trajectories=trajectories,
            is_correct=is_correct,
            metrics=metrics,
        )
        return episode

