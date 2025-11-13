import hydra
import os
import time
from omegaconf import OmegaConf

from rllm.agents.code_agent import CompetitionCodingAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.code.competition_coding import CompetitionCodingEnv
from rllm.rewards.reward_fn import code_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


def register_deepcoder_chunked_dataset():
    """Register the chunked DeepCoder dataset for training."""
    
    # Path to the chunked DeepCoder dataset in rllm/data/datasets/
    deepcoder_chunks_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # Go up to rllm root
        "rllm", 
        "data", 
        "datasets", 
        "deepcoder_verl_chunks"
    )
    
    if not os.path.exists(deepcoder_chunks_dir):
        print(f"DeepCoder chunks directory not found: {deepcoder_chunks_dir}")
        print("Please run prepare_deepcoder_data.py first to generate the chunks.")
        return None, None
    
    print(f"Registering chunked DeepCoder dataset from: {deepcoder_chunks_dir}")
    
    # Register training chunks
    train_dataset = DatasetRegistry.register_chunked_dataset(
        name="deepcoder_chunked",
        chunks_dir=deepcoder_chunks_dir,
        split="train"
    )
    
    # Register test chunks
    test_dataset = DatasetRegistry.register_chunked_dataset(
        name="deepcoder_chunked",
        chunks_dir=deepcoder_chunks_dir,
        split="test"
    )
    
    print(f"Registered chunked dataset: {train_dataset.name}")
    print(f"Available splits: {DatasetRegistry.get_dataset_splits('deepcoder_chunked')}")
    
    return train_dataset, test_dataset


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


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # Start timing
    start_time = time.time()
    print(f"Starting training at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Register chunked DeepCoder dataset
    train_dataset, _ = register_deepcoder_chunked_dataset()
    test_dataset = DatasetRegistry.load_dataset("lcb", "test")
    
    if train_dataset is None:
        print("Failed to register chunked dataset. Exiting.")
        return

    env_args = {"reward_fn": code_reward_fn, "max_turns": 4}

    trainer = AgentTrainer(
        agent_class=CompetitionCodingAgent,
        agent_args={},
        env_args=env_args,
        env_class=CompetitionCodingEnv,
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
