# train_solver_cm_only.py
#
# Trains ONLY the Context Manager (CM). The solver receives NO RL signal.
# Expects you have the workflow:
#   examples/solver_judge/solver_cm_train_cm_only_flow.py
# which defines `SolverContextManagerTrainCMOnlyWorkflow`.

import hydra

from examples.solver_judge.solver_cm_train_cm_only_flow import (
    SolverContextManagerTrainCMOnlyWorkflow,
)
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.countdown_reward import countdown_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # Datasets
    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    test_dataset = DatasetRegistry.load_dataset("countdown", "test")

    # Trainer: only CM will be optimized because only CM steps get rewards in this workflow
    trainer = AgentTrainer(
        workflow_class=SolverContextManagerTrainCMOnlyWorkflow,
        workflow_args={
            "n_rounds": 2,            # must be >=2 for CM to receive back-filled reward
            "n_solutions": 1,         # solver attempts per round (solver is not trained)
            "reward_function": countdown_reward_fn,
            # CM-only reward knobs (matched by your workflow):
            "success_bonus": 1.0,     # add bonus if post-CM round has any correct solution
            "clip_improvement": 1.0,  # clip delta reward to [-1, 1]; set None to disable
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
