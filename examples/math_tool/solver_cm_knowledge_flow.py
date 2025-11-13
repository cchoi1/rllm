from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import RewardFunction, math_reward_fn
from rllm.workflows.workflow import Workflow

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

    def __init__(self, rollout_engine: RolloutEngine, *, use_tools: bool = True):
        self.rollout_engine = rollout_engine
        self.use_tools = use_tools

    async def initial_attempt(self, problem: str, lessons: Optional[List[str]] = None) -> Trajectory:
        messages = [
            {
                "role": "user",
                "content": self._build_initial_solver_prompt(problem, lessons),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        solution = self._parse_solution_from_response(output.content)
        return Trajectory(
            name="solver",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": getattr(output, "reasoning", None)}],
                    thought=getattr(output, "reasoning", None),
                    action=solution,
                    model_output=output,
                    # mark non-trainable downstream (most trainers ignore this, but we add it to info just in case)
                    info={"trainable": False},
                )
            ],
        )

    async def refine_with_feedback(
        self,
        problem: str,
        previous_solution: str,
        math_feedback: str,
        cm_feedback: str,
        lessons: List[str],
    ) -> Trajectory:
        messages = [
            {
                "role": "user",
                "content": self._build_refine_solver_prompt(
                    problem=problem,
                    previous_solution=previous_solution,
                    math_feedback=math_feedback,
                    cm_feedback=cm_feedback,
                    lessons=lessons,
                ),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        solution = self._parse_solution_from_response(output.content)
        return Trajectory(
            name="solver",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": getattr(output, "reasoning", None)}],
                    thought=getattr(output, "reasoning", None),
                    action=solution,
                    model_output=output,
                    info={"trainable": False},
                )
            ],
        )

    # ---------------------- prompt & parse helpers ---------------------- #

    def _build_initial_solver_prompt(self, problem: str, lessons: Optional[List[str]] = None) -> str:
        prompt = problem
        if lessons and len(lessons) > 0:
            prompt += "\n\nLessons learned from previous attempts:\n"
            for i, lesson in enumerate(lessons, 1):
                prompt += f"{i}. {lesson}\n"
        if self.use_tools:
            prompt += "\n\nYou can use Python tools to help with calculations. Show your reasoning step by step and provide the final answer."
        return prompt

    def _build_refine_solver_prompt(
        self,
        problem: str,
        previous_solution: str,
        math_feedback: str,
        cm_feedback: str,
        lessons: List[str],
    ) -> str:
        prompt = (
            "You are refining your solution to a math problem. Consider the previous attempt, "
            "the correctness feedback, the ContextManager's advice, and the lessons learned. Produce an improved solution.\n\n"
            f"Problem:\n{problem}\n\n"
        )
        
        # Include lessons learned
        if lessons and len(lessons) > 0:
            prompt += "Lessons learned from previous attempts:\n"
            for i, lesson in enumerate(lessons, 1):
                prompt += f"{i}. {lesson}\n"
            prompt += "\n"
        
        prompt += (
            f"Previous attempt:\n{previous_solution}\n\n"
            f"Correctness feedback:\n{math_feedback}\n\n"
            f"ContextManager feedback:\n{cm_feedback}\n\n"
            "Please provide a complete, corrected solution with step-by-step reasoning."
            + (" You can use Python tools to help with calculations." if self.use_tools else "")
        )
        return prompt

    def _parse_solution_from_response(self, response: str) -> str:
        # Return the full response as the solution
        # The verifier will extract the answer from it
        return response.strip()


###############################################################################
# Trainable ContextManager (CM) that also extracts lessons
###############################################################################


class ContextManager:
    """Trainable agent that provides actionable feedback to the Solver and extracts lessons.

    This class *is* trained. It uses the actor/rollout engine passed to the
    Workflow via super().__init__(rollout_engine). The CM should output
    feedback inside <feedback>...</feedback> tags and lessons inside <lessons>...</lessons> tags.
    """

    def __init__(self, rollout_engine: RolloutEngine):
        self.rollout_engine = rollout_engine

    async def give_feedback_and_extract_lessons(
        self,
        problem: str,
        previous_solution: str,
        math_feedback: str,
        current_lessons: List[str],
        turn_index: int,
    ) -> Tuple[Trajectory, List[str]]:
        messages = [
            {
                "role": "user",
                "content": self._build_cm_prompt(
                    problem=problem,
                    previous_solution=previous_solution,
                    math_feedback=math_feedback,
                    current_lessons=current_lessons,
                    turn_index=turn_index,
                ),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        # Default: if CM response is truncated or unfinished thinking, do not pass feedback
        if output.finish_reason == "length" or ("<think>" in (output.text or output.content) and "</think>" not in (output.text or output.content)):
            feedback = ""
            updated_lessons = current_lessons
        else:
            feedback = self._parse_feedback(output.content)
            updated_lessons = self._parse_lessons(output.content, current_lessons)
        
        return Trajectory(
            name="context_manager",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": getattr(output, "reasoning", None)}],
                    thought=getattr(output, "reasoning", None),
                    action=feedback,  # Store feedback as action
                    model_output=output,
                )
            ],
        ), updated_lessons

    # ---------------------- prompt & parse helpers ---------------------- #

    def _build_cm_prompt(
        self,
        problem: str,
        previous_solution: str,
        math_feedback: str,
        current_lessons: List[str],
        turn_index: int,
    ) -> str:
        prompt = (
            "You are the ContextManager guiding a math problem-solving agent. "
            "Study the problem, the solver's last attempt, correctness feedback, and current lessons. "
            "Write precise, *actionable* guidance that will most improve the next attempt. "
            "Also reflect on what went wrong and extract key lessons to add to the knowledge base.\n\n"
            f"Turn: {turn_index}\n"
            f"Problem:\n{problem}\n\n"
        )
        
        # Show current lessons
        if current_lessons and len(current_lessons) > 0:
            prompt += "Current lessons learned:\n"
            for i, lesson in enumerate(current_lessons, 1):
                prompt += f"{i}. {lesson}\n"
            prompt += "\n"
        else:
            prompt += "Current lessons learned: (empty)\n\n"
        
        prompt += (
            f"Solver's previous solution:\n{previous_solution}\n\n"
            f"Correctness feedback:\n{math_feedback}\n\n"
            "Your tasks:\n"
            "1. Provide actionable feedback to help the solver improve (inside <feedback>...</feedback> tags).\n"
            "2. Extract and update lessons learned from this failed attempt (inside <lessons>...</lessons> tags).\n\n"
            "For feedback:\n"
            "- Be concise (<= 10 bullet points).\n"
            "- Focus on *root causes* and concrete fixes (calculation errors, logic issues, missing steps, format issues).\n"
            "- Identify specific errors and suggest corrections.\n"
            "- Avoid restating the problem.\n\n"
            "For lessons:\n"
            "- Extract key insights, patterns, and mistakes to avoid.\n"
            "- Update the lessons list: add new insights, remove outdated items, refine existing ones.\n"
            "- Keep lessons concise and actionable (max 10-15 items total).\n"
            "- Each lesson should be a clear, standalone insight that will help in future attempts.\n\n"
            "Format:\n"
            "<feedback>\n"
            "Your feedback here...\n"
            "</feedback>\n\n"
            "<lessons>\n"
            "1. Lesson one\n"
            "2. Lesson two\n"
            "...\n"
            "</lessons>"
        )
        return prompt

    def _parse_feedback(self, response: str) -> str:
        m = re.search(r"<feedback>(.*?)</feedback>", response, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else response.strip()

    def _parse_lessons(self, response: str, current_lessons: List[str]) -> List[str]:
        """Parse lessons list from response, or return current lessons if parsing fails."""
        m = re.search(r"<lessons>(.*?)</lessons>", response, re.DOTALL | re.IGNORECASE)
        if m:
            lessons_text = m.group(1).strip()
            # Parse numbered list items
            lesson_items = []
            for line in lessons_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Remove leading number (e.g., "1. ", "2. ", etc.)
                item = re.sub(r'^\d+\.\s*', '', line)
                if item:
                    lesson_items.append(item)
            if lesson_items:
                return lesson_items
        
        # Fallback: return current lessons if parsing fails
        return current_lessons


###############################################################################
# Workflow: train CM only, Solver fixed, N CM↔Solver turns on Math with lessons
###############################################################################


class MathCMKnowledgeWorkflow(Workflow):
    """Workflow that *trains only the ContextManager* on Math problems with lesson extraction.

    - The Solver uses a fixed RolloutEngine (frozen weights).\n
    - The ContextManager uses the trainable engine provided to Workflow.__init__.\n
    - Each episode runs up to `n_turns` CM↔Solver cycles:\n
        1) Solver attempts a solution (with current lessons).\n
        2) Verifier evaluates the solution.\n
        3) CM gives feedback and extracts/updates lessons based on (problem, previous solution, correctness feedback, current lessons).\n
        4) Solver refines using CM advice and updated lessons.\n
      The reward for each CM step is the *delta in correctness* (0->1 or 0->0) between
      the solver attempt after CM advice and the solver attempt before CM advice.\n
    Notes
    -----
    * Ensure your trainer only updates the actor tied to the CM (e.g., filter by
      trajectory name == 'context_manager' or ignore trajectories with
      metadata.trainable == False).\n
    * Initialize the CM from a pretrained checkpoint. The
      Solver should also point to a pretrained checkpoint but its engine must
      be excluded from training.\n
    * Lessons accumulate across turns within an episode, helping the solver learn from past mistakes.
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
                task_info = {"ground_truth": task.get("ground_truth")}
                reward_output = math_reward_fn(task_info, solution) if reward_function is None else reward_function(task_info, solution)
                return MathResult(
                    is_correct=reward_output.is_correct,
                    reward=reward_output.reward,
                    feedback=reward_output.metadata.get("feedback", ""),
                    metadata=reward_output.metadata or {},
                )
            verifier = default_verifier
        
        super().__init__(cm_engine, **kwargs)  # this is the *trainable* engine
        self.n_turns = n_turns
        self.verifier = verifier
        self.reward_function = reward_function  # optional override
        self.solver = MathSolver(solver_engine, use_tools=use_tools)
        self.cm = ContextManager(cm_engine)

    def collect_metrics(self, episode: Episode) -> None:
        """
        Collect metrics from the episode.
        
        Args:
            episode: The episode to collect metrics from.
        """
        # Call parent to get trajectory rewards
        super().collect_metrics(episode)
        
        # Only log pass@1 after n_turns
        if not hasattr(episode, 'metrics') or episode.metrics is None:
            episode.metrics = {}
        
        # Set pass@1 to 1.0 if final solution is correct, 0.0 otherwise
        episode.metrics["pass@1"] = 1.0 if episode.is_correct else 0.0

    # ---------------------- core episode logic ---------------------- #

    async def run(self, task: Dict, uid: str, **kwargs) -> Episode:
        self.reset(task, uid)

        # Expect these keys in Math tasks
        problem: str = task.get("question") or task.get("problem") or task.get("prompt", "")

        trajectories: List[Trajectory] = []
        metrics: Dict[str, float] = {}
        
        # Initialize lessons list
        lessons: List[str] = []

        # Initial solver attempt (with initial lessons if any)
        solver_traj = await self.solver.initial_attempt(problem, lessons)
        # Explicitly ensure initial solver trajectory steps have 0 reward (solver is not trained)
        for step in solver_traj.steps:
            step.reward = 0.0
        trajectories.append(solver_traj)

        # Evaluate
        result_prev = self.verifier(task, solver_traj.steps[0].action)

        # Early stop if already correct
        if result_prev.is_correct:
            metrics["pass@1"] = 1.0
            episode = Episode(
                id=uid,
                task=task,
                trajectories=trajectories,
                is_correct=True,
                metrics=metrics,
            )
            self.collect_metrics(episode)
            return episode

        # Interleave CM feedback/lesson extraction and solver refinements
        for t in range(1, self.n_turns + 1):
            # Format feedback
            formatted_feedback = format_math_feedback(result_prev)
            
            # Get the most recent solution (from previous iteration)
            previous_solution = solver_traj.steps[-1].action
            
            # CM feedback and lesson extraction (trainable step)
            cm_traj, lessons = await self.cm.give_feedback_and_extract_lessons(
                problem=problem,
                previous_solution=previous_solution,
                math_feedback=formatted_feedback,
                current_lessons=lessons,
                turn_index=t,
            )
            trajectories.append(cm_traj)
            
            # Track number of lessons for metrics
            metrics[f"lessons_count_t{t}"] = float(len(lessons))

            # Solver refines using CM advice and updated lessons
            solver_traj_next = await self.solver.refine_with_feedback(
                problem=problem,
                previous_solution=previous_solution,
                math_feedback=formatted_feedback,
                cm_feedback=cm_traj.steps[-1].action,
                lessons=lessons,
            )
            trajectories.append(solver_traj_next)

            # Evaluate the refined solution
            result_next = self.verifier(task, solver_traj_next.steps[-1].action)

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
                break

        is_correct = result_prev.is_correct
        # Populate metrics before passing to Episode
        metrics["pass@1"] = 1.0 if is_correct else 0.0
        episode = Episode(
            id=uid,
            task=task,
            trajectories=trajectories,
            is_correct=is_correct,
            metrics=metrics,
        )
        self.collect_metrics(episode)
        return episode

