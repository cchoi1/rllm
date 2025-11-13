# train_solver_cm_joint.py
#
# Joint-optimization variant: trains BOTH the Solver and the Context Manager (CM).
# Requires a workflow that assigns rewards to CM steps (e.g., improvement-based + optional success bonus).
#
# Example expects you to have:
#   examples/solver_judge/solver_cm_joint_flow.py
# with a class `SolverContextManagerJointWorkflow` that accepts:
#   - n_rounds: int
#   - n_solutions: int
#   - reward_function: callable
#   - cm_reward: str = "improvement"            # or "improvement+success"
#   - success_bonus: float = 1.0                 # bonus when post-CM round has any correct
#   - clip_improvement: float | None = 1.0       # clip delta reward to [-clip, clip]; None disables
#
# Hydra config name stays the same as the non-joint run (agent_ppo_trainer).

import hydra

from examples.countdown.solver_cm_joint import SolverContextManagerJointWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.countdown_reward import countdown_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # Datasets
    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    test_dataset = DatasetRegistry.load_dataset("countdown", "test")

    # Trainer (joint optimization of solver + CM)
    trainer = AgentTrainer(
        workflow_class=SolverContextManagerJointWorkflow,
        workflow_args={
            "n_rounds": 2,                 # at least 2 for CM to affect a later round
            "n_solutions": 1,              # attempts per round
            "reward_function": countdown_reward_fn,
            # --- Joint-optimization knobs (must be supported by your joint workflow) ---
            "cm_reward": "improvement+success",  # improvement signal + sparse success bonus
            "success_bonus": 1.0,                 # add 1.0 when post-CM round has any correct
            "clip_improvement": 1.0,              # clip delta to [-1, 1] for stability
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
