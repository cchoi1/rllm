import hydra

from examples.solver_judge.judge_flow import JudgeOnlyWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.countdown_reward import countdown_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # Load datasets
    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    test_dataset = DatasetRegistry.load_dataset("countdown", "test")

    # Trainer configured to train ONLY the judge
    trainer = AgentTrainer(
        workflow_class=JudgeOnlyWorkflow,
        workflow_args={
            "n_solutions": 2,                 # number of proposals per problem
            "reward_function": countdown_reward_fn,
            # Optional: provide a distinct, frozen solver engine here if desired:
            # "solver_engine": some_other_rollout_engine,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()
