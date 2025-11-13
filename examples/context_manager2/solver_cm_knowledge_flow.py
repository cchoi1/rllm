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

    async def initial_attempt(self, problem: str, knowledge_list: Optional[List[str]] = None, io_spec: Optional[str] = None) -> Trajectory:
        messages = [
            {
                "role": "user",
                "content": self._build_initial_solver_prompt(problem, knowledge_list, io_spec),
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

    async def refine_with_knowledge(
        self,
        problem: str,
        previous_solution: Optional[str],
        unit_test_feedback: str,
        knowledge_list: List[str],
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
                    knowledge_list=knowledge_list,
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

    def _build_initial_solver_prompt(self, problem: str, knowledge_list: Optional[List[str]] = None, io_spec: Optional[str] = None) -> str:
        prompt = problem
        if knowledge_list and len(knowledge_list) > 0:
            prompt += "\n\nKnowledge to consider:\n"
            for i, knowledge in enumerate(knowledge_list, 1):
                prompt += f"{i}. {knowledge}\n"
        if io_spec:
            prompt += f"\nI/O Specification:\n{io_spec}\n"
        return prompt

    def _build_refine_solver_prompt(
        self,
        problem: str,
        io_spec: Optional[str],
        previous_solution: Optional[str],
        unit_test_feedback: str,
        knowledge_list: List[str],
        truncation_info: Optional[str] = None,
    ) -> str:
        revise_instruction = "Here's the feedback from the previous attempt. Revise the code to fix the errors and improve the solution."
        prompt = f"{revise_instruction}\n"
        
        # Include the original problem
        prompt += f"\nProblem:\n{problem}\n"
        
        # Include I/O spec if available
        if io_spec:
            prompt += f"\nI/O Specification:\n{io_spec}\n"
        
        # Include knowledge list
        if knowledge_list and len(knowledge_list) > 0:
            prompt += "\nKnowledge gathered so far:\n"
            for i, knowledge in enumerate(knowledge_list, 1):
                prompt += f"{i}. {knowledge}\n"
            prompt += "\n"
        
        # Include test feedback
        prompt += f"{unit_test_feedback}\n"
        return prompt

    def _parse_code_from_response(self, response: str) -> str:
        # Prefer fenced code blocks
        fence = re.search(r"```[a-zA-Z0-9_\-]*\n(.*?)```", response, re.DOTALL)
        if fence:
            return fence.group(1).strip()
        return ""


###############################################################################
# Trainable Knowledge Manager (CM that maintains knowledge list)
###############################################################################

class KnowledgeManager:
    """Trainable agent that maintains and updates a knowledge list iteratively.

    This class *is* trained. It uses the actor/rollout engine passed to the
    Workflow via super().__init__(rollout_engine). The KM should output
    updated knowledge inside <knowledge>...</knowledge> tags.
    """

    def __init__(self, rollout_engine: RolloutEngine):
        self.rollout_engine = rollout_engine

    async def update_knowledge(
        self,
        problem: str,
        previous_solution: Optional[str],
        unit_test_feedback: str,
        current_knowledge: List[str],
        turn_index: int,
        io_spec: Optional[str] = None,
        truncation_info: Optional[str] = None,
    ) -> Tuple[Trajectory, List[str]]:
        messages = [
            {
                "role": "user",
                "content": self._build_knowledge_prompt(
                    problem=problem,
                    io_spec=io_spec,
                    previous_solution=previous_solution,
                    unit_test_feedback=unit_test_feedback,
                    current_knowledge=current_knowledge,
                    turn_index=turn_index,
                    truncation_info=truncation_info,
                ),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        updated_knowledge = self._parse_knowledge(output.content, current_knowledge)
        return Trajectory(
            name="knowledge_manager",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": getattr(output, "reasoning", None)}],
                    thought=getattr(output, "reasoning", None),
                    action=updated_knowledge,  # Store the knowledge list as action
                    model_output=output,
                )
            ],
        ), updated_knowledge

    # ---------------------- prompt & parse helpers ---------------------- #

    def _build_knowledge_prompt(
        self,
        problem: str,
        io_spec: Optional[str],
        previous_solution: Optional[str],
        unit_test_feedback: str,
        current_knowledge: List[str],
        turn_index: int,
        truncation_info: Optional[str] = None,
    ) -> str:
        prompt = (
            "You are a Knowledge Manager that maintains an iterative knowledge list "
            "to help solve programming problems. Your task is to update the knowledge list "
            "based on the Solver's attempts and test results.\n\n"
            "Extract key insights, patterns, constraints, and lessons learned from:\n"
            "- The problem statement\n"
            "- The Solver's previous attempt\n"
            "- Unit test feedback\n"
            "- Current knowledge list\n\n"
            f"Turn: {turn_index}\n"
            f"Problem:\n{problem}\n"
        )
        if io_spec:
            prompt += f"\nI/O Specification:\n{io_spec}\n"
        
        # Show current knowledge
        if current_knowledge and len(current_knowledge) > 0:
            prompt += "\nCurrent knowledge list:\n"
            for i, knowledge in enumerate(current_knowledge, 1):
                prompt += f"{i}. {knowledge}\n"
        else:
            prompt += "\nCurrent knowledge list: (empty)\n"
        
        # Show previous solution attempt
        if previous_solution:
            prompt += f"\nSolver's previous attempt:\n```python\n{previous_solution}\n```\n\n"
        else:
            if truncation_info:
                prompt += f"\n[Solver's previous attempt excluded: {truncation_info}]\n\n"
            else:
                prompt += "\n[Solver's previous attempt excluded due to truncation]\n\n"
        
        prompt += f"Unit-test feedback:\n{unit_test_feedback}\n\n"
        prompt += (
            "Instructions:\n"
            "- Analyze what worked and what didn't in the previous attempt\n"
            "- Identify patterns, edge cases, and constraints\n"
            "- Update the knowledge list: add new insights, remove outdated items, refine existing ones\n"
            "- Keep the knowledge list concise and actionable (max 10-15 items)\n"
            "- Each knowledge item should be a clear, standalone insight\n\n"
            "Return the complete updated knowledge list inside <knowledge>...</knowledge> tags.\n"
            "Format each knowledge item as a numbered list item.\n"
            "Example format:\n"
            "<knowledge>\n"
            "1. Item one\n"
            "2. Item two\n"
            "...\n"
            "</knowledge>"
        )
        return prompt

    def _parse_knowledge(self, response: str, current_knowledge: List[str]) -> List[str]:
        """Parse knowledge list from response, or return current knowledge if parsing fails."""
        m = re.search(r"<knowledge>(.*?)</knowledge>", response, re.DOTALL | re.IGNORECASE)
        if m:
            knowledge_text = m.group(1).strip()
            # Parse numbered list items
            knowledge_items = []
            for line in knowledge_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Remove leading number (e.g., "1. ", "2. ", etc.)
                item = re.sub(r'^\d+\.\s*', '', line)
                if item:
                    knowledge_items.append(item)
            if knowledge_items:
                return knowledge_items
        
        # Fallback: return current knowledge if parsing fails
        return current_knowledge


###############################################################################
# Workflow: train KM only, Solver fixed, iterative knowledge updates
###############################################################################

class DeepCoderKnowledgeFlowWorkflow(Workflow):
    """Workflow that *trains only the KnowledgeManager* on DeepCoder.

    - The Solver uses a fixed RolloutEngine (frozen weights).\n
    - The KnowledgeManager uses the trainable engine provided to Workflow.__init__.\n
    - Each episode runs up to `n_turns` KM↔Solver cycles:\n
        1) Solver attempts a program (using current knowledge).\n
        2) Verifier runs tests and produces textual feedback.\n
        3) KM updates knowledge list based on (problem, previous program, test feedback, current knowledge).\n
        4) Solver refines using updated knowledge list.\n
      The reward for each KM step is the *delta in unit-test pass ratio* between
      the solver attempt after KM update and the solver attempt before KM update.\n
    Notes
    -----
    * Ensure your trainer only updates the actor tied to the KM (e.g., filter by
      trajectory name == 'knowledge_manager').\n
    * The knowledge list is maintained across turns within an episode.\n
    * Knowledge accumulates and evolves based on solver attempts and test results.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine = None,
        *,
        solver_engine: RolloutEngine,
        verifier: VerifierFn,
        reward_function: Optional[RewardFunction] = None,
        n_turns: int = 4,
        language: str = "python",
        km_engine: RolloutEngine = None,
        exclude_truncated: bool = False,
        **kwargs,
    ) -> None:
        # Handle both rollout_engine (from AgentWorkflowEngine) and km_engine (from workflow_args)
        km_engine = km_engine or rollout_engine
        if km_engine is None:
            raise ValueError("Either rollout_engine or km_engine must be provided")
        super().__init__(km_engine, **kwargs)  # this is the *trainable* engine
        self.n_turns = n_turns
        self.verifier = verifier
        self.reward_function = reward_function  # optional override
        self.solver = DeepCoderSolver(solver_engine, language=language)
        self.km = KnowledgeManager(km_engine)
        self.exclude_truncated = exclude_truncated
        # Store max_response_length from engines for truncation messages
        self.solver_max_response_length = getattr(solver_engine, 'max_response_length', None)
        self.km_max_response_length = getattr(km_engine, 'max_response_length', None)

    # ---------------------- core episode logic ---------------------- #

    async def run(self, task: Dict, uid: str, **kwargs) -> Episode:
        self.reset(task, uid)

        # Expect these keys in DeepCoder tasks (adjust if your loader differs)
        problem: str = task.get("question") or task.get("problem") or task["prompt"]
        io_spec: Optional[str] = task.get("io_spec")  # optional (depends on dataset view)

        trajectories: List[Trajectory] = []
        metrics: Dict[str, float] = {}
        
        # Initialize knowledge list
        knowledge_list: List[str] = []

        # Initial solver attempt (with initial knowledge if any)
        solver_traj = await self.solver.initial_attempt(problem, knowledge_list, io_spec)
        trajectories.append(solver_traj)

        # Evaluate
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
                trajectories=trajectories,  # no KM steps; nothing to train on here
                is_correct=True,
                metrics=metrics,
            )

        # Interleave KM knowledge updates and solver refinements
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
                        # Response was truncated - exclude from KM prompt but provide truncation info
                        tokens_generated = model_output.completion_length
                        max_token_limit = self.solver_max_response_length
                        truncation_info_solver = f"Incomplete response (truncated at {tokens_generated} tokens, limit: {max_token_limit})"
                        previous_solution = None  # Will be handled in KM prompt building
            
            # KM updates knowledge list (trainable step)
            km_traj, knowledge_list = await self.km.update_knowledge(
                problem=problem,
                previous_solution=previous_solution,
                unit_test_feedback=formatted_test_feedback,
                current_knowledge=knowledge_list,
                turn_index=t,
                io_spec=io_spec,
                truncation_info=truncation_info_solver,
            )
            trajectories.append(km_traj)

            # Solver refines using updated knowledge list
            solver_traj_next = await self.solver.refine_with_knowledge(
                problem=problem,
                previous_solution=previous_solution,
                unit_test_feedback=formatted_test_feedback,
                knowledge_list=knowledge_list,
                io_spec=io_spec,
                truncation_info=None,  # Not needed for solver prompt
            )
            trajectories.append(solver_traj_next)

            # Evaluate the refined program
            model_response = solver_traj_next.steps[-1].model_response if hasattr(solver_traj_next.steps[-1], 'model_response') and solver_traj_next.steps[-1].model_response else solver_traj_next.steps[-1].action
            result_next = self.verifier(task, model_response)

            # Reward: delta pass ratio (after KM vs before KM)
            delta = result_next.ratio - result_prev.ratio
            km_traj.steps[-1].reward = float(delta)

            # Update rolling state & metrics
            metrics[f"delta_t{t}"] = float(delta)
            metrics[f"pass_ratio_t{t}"] = result_next.ratio
            metrics[f"passed_t{t}"] = float(result_next.passed)
            metrics[f"knowledge_items_t{t}"] = float(len(knowledge_list))

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

