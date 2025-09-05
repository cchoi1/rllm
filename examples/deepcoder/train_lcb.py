import hydra
import os
import time
from omegaconf import OmegaConf

from rllm.agents.code_agent import CompetitionCodingAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import code_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


def format_time(seconds):
    """Format time in seconds to human readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    # Start timing
    start_time = time.time()
    print(f"Starting training at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    train_dataset = DatasetRegistry.load_dataset("lcb", "train")
    test_dataset = DatasetRegistry.load_dataset("lcb", "test")
    
    if train_dataset is None:
        print("Failed to register chunked dataset. Exiting.")
        return

    env_args = {"reward_fn": code_reward_fn}

    trainer = AgentTrainer(
        agent_class=CompetitionCodingAgent,
        agent_args={},
        env_args=env_args,
        env_class=SingleTurnEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        # Calculate and print total run time
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "="*60)
        print(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total run time: {format_time(total_time)} ({total_time:.2f} seconds)")
        print("="*60)


if __name__ == "__main__":
    main()