# solver_cm_train_cm_only_flow.py

import asyncio
import re
from typing import List, Optional

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import Workflow

FEEDBACK_TAG_RE = re.compile(r"<feedback>(.*?)</feedback>", re.IGNORECASE | re.DOTALL)
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


class Solver:
    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def generate_solution(self, problem: str, feedback: Optional[str] = None) -> Trajectory:
        """Optionally conditions the attempt on feedback from the ContextManager."""
        if feedback:
            user_content = (
                f"{problem}\n\n"
                "You previously attempted this problem. Here is feedback from an expert coach:\n"
                f"<feedback>{feedback}</feedback>\n\n"
                "Revise your approach and try again. Output the final answer within <answer>...</answer>."
            )
        else:
            user_content = f"{problem}. Output the final answer within <answer>...</answer>"

        messages = [{"role": "user", "content": user_content}]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)

        return Trajectory(
            name="solver",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=self._parse_solver_response(output.content),
                    model_output=output,
                )
            ],
        )

    async def generate_solutions(
        self, problem: str, n_solutions: int = 2, feedback: Optional[str] = None
    ) -> List[Trajectory]:
        tasks = [asyncio.create_task(self.generate_solution(problem, feedback=feedback)) for _ in range(n_solutions)]
        return await asyncio.gather(*tasks)

    def _parse_solver_response(self, response: str) -> str:
        match = ANSWER_TAG_RE.search(response)
        if match:
            return f"<answer>{match.group(1).strip()}</answer>"
        return "No solution found"


class ContextManager:
    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def provide_feedback(self, problem: str, solutions: List[str]) -> Trajectory:
        """Reviews solver attempts and returns actionable feedback to guide the next attempt."""
        messages = [{"role": "user", "content": self._create_cm_prompt(problem, solutions)}]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)

        return Trajectory(
            name="context_manager",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=self._parse_feedback(output.content),
                    model_output=output,
                )
            ],
        )

    def _parse_feedback(self, response: str) -> str:
        match = FEEDBACK_TAG_RE.search(response)
        if match:
            return match.group(1).strip()
        return response.strip()

    def _create_cm_prompt(self, problem: str, solutions: List[str]) -> str:
        prompt = (
            "You are an expert Context Manager (coach). Review the problem and the solver’s attempts, "
            "diagnose mistakes, and give actionable guidance for the NEXT attempt.\n\n"
            f"Problem:\n{problem}\n\nAttempts so far:\n"
        )
        for i, sol in enumerate(solutions, 1):
            prompt += f"\nAttempt {i}:\n{sol}\n"

        prompt += (
            "\nReturn your guidance inside <feedback>...</feedback> tags. Be specific: "
            "identify incorrect operations, unused/overused numbers, arithmetic mistakes, or missing steps. "
            "Then propose a concrete plan for the next attempt (e.g., operation ordering or number pairing)."
        )
        return prompt


class SolverContextManagerTrainCMOnlyWorkflow(Workflow):
    """
    Trains ONLY the Context Manager (CM).
      - Solver steps NEVER set a reward (so the solver policy won't update).
      - CM gets a reward equal to the improvement it causes in the next round's solver quality,
        optionally plus a success bonus if any correct appears post-CM.

    Requires n_rounds >= 2 for CM to receive any signal.
    Metrics:
      - solver_acc: average reward of solver attempts (reported for monitoring only; not used to train solver)
      - cm_acc: 1 if any correct solution appears in rounds >= 2 (post-CM), else 0
      - avg_improvement: final_avg - first_avg
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        n_rounds: int = 2,
        n_solutions: int = 1,
        reward_function: RewardFunction | None = None,
        success_bonus: float = 1.0,
        clip_improvement: Optional[float] = 1.0,
        **kwargs,
    ):
        super().__init__(rollout_engine, **kwargs)
        if reward_function is None:
            raise ValueError("SolverContextManagerTrainCMOnlyWorkflow requires a reward_function.")
        if n_rounds < 1:
            raise ValueError("n_rounds must be >= 1.")

        self.n_rounds = n_rounds
        self.n_solutions = n_solutions
        self.reward_function = reward_function
        self.success_bonus = success_bonus
        self.clip_improvement = clip_improvement

        self.solver = Solver(rollout_engine)
        self.context_manager = ContextManager(rollout_engine)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task, uid)
        problem = task["question"]

        all_solver_trajectories: List[Trajectory] = []
        cm_trajectories: List[Trajectory] = []
        round_metrics: List[dict] = []

        feedback: Optional[str] = None
        best_reward = float("-inf")
        best_is_correct = False
        prior_solutions: List[str] = []
        post_cm_any_correct = False

        # For back-filling CM reward
        prev_cm_traj: Optional[Trajectory] = None
        prev_round_avg: Optional[float] = None

        for round_idx in range(1, self.n_rounds + 1):
            # 1) Solver attempts (conditioned on prior feedback if available)
            solver_trajectories = await self.solver.generate_solutions(
                problem, n_solutions=self.n_solutions, feedback=feedback
            )
            all_solver_trajectories.extend(solver_trajectories)

            # 2) Score attempts (for monitoring + for CM reward computation ONLY)
            round_rewards: List[float] = []
            round_correct = 0

            for traj in solver_trajectories:
                step = traj.steps[0]
                solution = step.action
                prior_solutions.append(solution)

                result = self.reward_function(task, solution)
                # DO NOT set solver reward -> keep it None to avoid training solver
                # step.reward = None
                round_rewards.append(result.reward)

                if getattr(result, "is_correct", False):
                    round_correct += 1
                    best_is_correct = True

                if result.reward > best_reward:
                    best_reward = result.reward

            round_avg_reward = sum(round_rewards) / len(round_rewards) if round_rewards else 0.0
            round_max_reward = max(round_rewards) if round_rewards else 0.0
            round_has_correct = round_correct > 0
            round_attempts = len(solver_trajectories)

            # 3) Credit the PREVIOUS CM (it produced feedback that affected THIS round)
            if prev_cm_traj is not None:
                cm_impr = round_avg_reward - (prev_round_avg if prev_round_avg is not None else 0.0)
                if self.clip_improvement is not None:
                    c = float(self.clip_improvement)
                    cm_impr = max(-c, min(c, cm_impr))

                cm_reward = float(cm_impr)
                if round_has_correct and self.success_bonus:
                    cm_reward += float(self.success_bonus)

                prev_cm_traj.steps[0].reward = cm_reward  # <-- CM gets rewarded (only learner)

            # Track post-CM correctness
            if round_idx >= 2 and round_has_correct:
                post_cm_any_correct = True

            round_metrics.append(
                {
                    "round": round_idx,
                    "attempts": round_attempts,
                    "num_correct": round_correct,
                    "has_correct": round_has_correct,
                    "avg_reward": round_avg_reward,
                    "max_reward": round_max_reward,
                }
            )

            # 4) If more rounds remain, get CM feedback for NEXT round
            if round_idx < self.n_rounds:
                cm_traj = await self.context_manager.provide_feedback(problem, prior_solutions)
                cm_trajectories.append(cm_traj)
                feedback = cm_traj.steps[0].action

                # Save for next round's back-fill
                prev_cm_traj = cm_traj
                prev_round_avg = round_avg_reward
            else:
                feedback = None  # final round -> no CM after this

        # --- Aggregate metrics (solver_acc is monitoring only)
        all_rewards_for_monitor = []
        for traj in all_solver_trajectories:
            # solver rewards were never set; compute monitoring avg from recorded round metrics
            # We already computed per-round averages; to keep consistent, report their mean:
            pass
        # Use first/last round avgs for improvement
        solver_acc = (
            sum(m["avg_reward"] for m in round_metrics) / len(round_metrics)
            if round_metrics else 0.0
        )

        cm_acc = int(bool(post_cm_any_correct))
        improvement = 0.0
        if len(round_metrics) >= 2:
            improvement = round_metrics[-1]["avg_reward"] - round_metrics[0]["avg_reward"]

        metrics = {
            "solver_acc": float(solver_acc),          # monitoring only (solver is frozen in RL sense)
            "cm_acc": float(cm_acc),                  # 1.0 if any post-CM attempt correct
            "best_reward": float(best_reward if best_reward != float("-inf") else 0.0),
            "avg_improvement": float(improvement),
        }

        # Optional: per-round logs
        for i, m in enumerate(round_metrics, 1):
            metrics[f"r{i}_avg_reward"]   = float(m.get("avg_reward", 0.0))
            metrics[f"r{i}_max_reward"]   = float(m.get("max_reward", 0.0))
            metrics[f"r{i}_num_correct"]  = float(m.get("num_correct", 0))
            metrics[f"r{i}_attempts"]     = float(m.get("attempts", 0))
            metrics[f"r{i}_has_correct"]  = float(1.0 if m.get("has_correct", False) else 0.0)

        return Episode(
            id=uid,
            task=task,
            trajectories=[*all_solver_trajectories, *cm_trajectories],
            is_correct=best_is_correct,
            metrics=metrics,
        )
