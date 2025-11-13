import asyncio
import re
from typing import List

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import Workflow


class Solver:
    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def generate_solution(self, problem: str) -> Trajectory:
        messages = [
            {
                "role": "user",
                "content": f"{problem}. Output the final answer within <answer>...</answer>",
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)
        return Trajectory(
            name="solver",
            steps=[
                Step(
                    chat_completions=messages
                    + [
                        {
                            "role": "assistant",
                            "content": output.content,
                            "reasoning": output.reasoning,
                        }
                    ],
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
        match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if match:
            return f"<answer>{match.group(1).strip()}</answer>"
        return "No solution found"


class SolverWorkflow(Workflow):
    """
    Workflow that:
      1) Spawns N solver attempts in parallel
      2) Scores each attempt with the provided RewardFunction
      3) Selects the best attempt by reward
      4) Returns an Episode containing all solver trajectories and metrics
    """

    def __init__(
        self,
            rollout_engine: RolloutEngine,
            n_solutions: int = 2,
            reward_function: RewardFunction | None = None,
            **kwargs,
    ):
        super().__init__(rollout_engine, **kwargs)
        self.n_solutions = n_solutions
        self.reward_function = reward_function
        self.solver = Solver(rollout_engine)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        if self.reward_function is None:
            raise ValueError("SolverWorkflow requires a reward_function.")

        self.reset(task, uid)
        problem = task["question"]

        # 1) Generate multiple solutions in parallel
        solver_trajectories = await self.solver.generate_solutions(problem, self.n_solutions)

        # 2) Score each attempt
        results = []
        for traj in solver_trajectories:
            step = traj.steps[0]
            solution = step.action
            result = self.reward_function(task, solution)
            step.reward = result.reward
            # optionally stash correctness on the step (not required by core API)
            # you can also put this into step.info/metadata if your infra supports it
            results.append(result)

        # 3) Select best attempt by reward
        best_idx = max(range(len(results)), key=lambda i: results[i].reward) if results else 0
        is_correct = results[best_idx].is_correct if results else False

        # Metrics
        solver_acc = (sum(r.reward for r in results) / len(results)) if results else 0.0
        best_reward = results[best_idx].reward if results else 0.0

        # 4) Return episode
        return Episode(
            id=uid,
            task=task,
            trajectories=solver_trajectories,
            is_correct=is_correct,
            metrics={
                "solver_acc": solver_acc,          # average reward over attempts
                "best_idx": best_idx + 1,          # 1-based index of best attempt
                "best_reward": best_reward,        # reward of best attempt
            },
        )
