import asyncio
import re
from typing import List, Optional

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import Workflow


class Solver:
    """Frozen proposal generator — NOT trained."""
    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def generate_solution(self, problem: str) -> Trajectory:
        messages = [{"role": "user", "content": f"{problem}. Output the final answer within <answer>...</answer>"}]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        return Trajectory(
            name="solver",
            steps=[
                Step(
                    chat_completions=messages + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=self._parse_solver_response(output.content),
                    model_output=output,
                )
            ],
        )

    async def generate_solutions(self, problem: str, n_solutions: int = 2) -> List[Trajectory]:
        tasks = [asyncio.create_task(self.generate_solution(problem)) for _ in range(n_solutions)]
        return await asyncio.gather(*tasks)

    def _parse_solver_response(self, response: str) -> str:
        m = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        return f"<answer>{m.group(1).strip()}</answer>" if m else "No solution found"


class Judge:
    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def judge_solutions(self, problem: str, solutions: List[str]) -> Trajectory:
        messages = [{"role": "user", "content": self._create_judge_prompt(problem, solutions)}]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        return Trajectory(
            name="judge",
            steps=[
                Step(
                    chat_completions=messages + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=self._parse_judge_response(output.content, solutions),
                    model_output=output,
                )
            ],
        )

    def _parse_judge_response(self, response: str, solutions: List[str]) -> str:
        m = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if not m:
            return ""
        try:
            idx = int(m.group(1).strip())
            return solutions[idx - 1]  # 1-based index
        except (ValueError, IndexError):
            return ""

    def _create_judge_prompt(self, problem: str, solutions: List[str]) -> str:
        prompt = f"""You are an expert verifier. Given a countdown problem and multiple solution attempts, select a correct solution.
Problem:
{problem}
Solutions to evaluate:
"""
        for i, sol in enumerate(solutions, 1):
            prompt += f"\nSolution {i}:\n{sol}\n"

        prompt += """
A correct solution must satisfy the following criteria:
1. The solution uses only the given numbers.
2. Each number is used exactly once.
3. Only basic arithmetic operations (+, -, *, /) are used.
4. The calculation results in the target number.
5. The final answer is clearly marked within <answer>...</answer> tags.

Output the index of your selected solution within <answer>...</answer> tags, e.g., <answer>1</answer>.
If multiple solutions are correct, prefer the first correct solution."""
        return prompt


class JudgeOnlyWorkflow(Workflow):
    """
    Train ONLY the judge:
    - Uses a frozen solver (or task-provided candidates) to produce proposals.
    - Returns ONLY the judge trajectory in the episode so the trainer updates the judge model/params.
    """
    def __init__(
        self,
        rollout_engine: RolloutEngine,
        n_solutions: int = 2,
        reward_function: Optional[RewardFunction] = None,
        solver_engine: Optional[RolloutEngine] = None,  # optional separate (frozen) engine for proposals
        **kwargs,
    ):
        super().__init__(rollout_engine, **kwargs)
        self.n_solutions = n_solutions
        self.reward_function = reward_function
        # If a distinct frozen solver engine is provided, use it; else reuse rollout_engine
        self.solver = Solver(solver_engine or rollout_engine)
        self.judge = Judge(rollout_engine)

    async def _get_candidate_solutions(self, task: dict) -> List[str]:
        # Option 1: use precomputed candidates if given
        if "candidates" in task and isinstance(task["candidates"], list) and task["candidates"]:
            return task["candidates"]

        # Option 2: generate with frozen solver (not trained)
        problem = task["question"]
        solver_trajectories = await self.solver.generate_solutions(problem, self.n_solutions)
        return [traj.steps[0].action for traj in solver_trajectories]

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task, uid)
        problem = task["question"]

        # Gather candidates without creating trainable solver trajectories
        solutions = await self._get_candidate_solutions(task)

        # Judge selects the best solution (this is the ONLY trainable trajectory we return)
        judge_traj = await self.judge.judge_solutions(problem, solutions)
        selected_solution = judge_traj.steps[0].action

        # Reward the judge's decision
        reward_result = self.reward_function(task, selected_solution) if self.reward_function else None
        if reward_result is not None:
            judge_traj.steps[0].reward = reward_result.reward
            is_correct = reward_result.is_correct
        else:
            # If no reward fn provided, default to 0 and unknown correctness
            judge_traj.steps[0].reward = 0.0
            is_correct = False

        # Metrics: we only care about the judge accuracy here
        metrics = {"judge_acc": int(is_correct)}

        # Return ONLY the judge trajectory so the trainer updates only the judge
        # (Optionally stash solutions for logging/debugging in episode extras)
        return Episode(
            id=uid,
            task=task,
            trajectories=[judge_traj],
            is_correct=is_correct,
            metrics=metrics,
            # extras={"candidate_solutions": solutions, "selected_solution": selected_solution},
        )
