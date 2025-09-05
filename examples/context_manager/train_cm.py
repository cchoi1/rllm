import asyncio
import multiprocessing as mp, asyncio
try: 
    mp.set_start_method("spawn", force=True)
except RuntimeError: 
    pass
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import hydra
import os
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "1000000000"

from pathlib import Path

from rllm.trainer.agent_trainer import AgentTrainer
from rllm.agents.context_manager_agent import ContextManagerAgent
from rllm.environments.base.context_manager_env import ContextManagerEnv
from rllm.rewards.cm_reward import rllm_reward_fn_context_assist
from rllm.data.dataset import DatasetRegistry


def register_deepcoder_chunked_dataset():
    """Register the chunked DeepCoder dataset for training."""
    
    # Path to the chunked DeepCoder dataset
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


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    # Register chunked DeepCoder dataset
    train_dataset, _ = register_deepcoder_chunked_dataset()
    # train_dataset = DatasetRegistry.load_dataset("lcb", "train")
    test_dataset = DatasetRegistry.load_dataset("lcb", "test")
    
    if train_dataset is None:
        print("Failed to register chunked dataset. Exiting.")
        return
    
    # Build env args (matching run_cm.py)
    # Get default values from config if available, otherwise use defaults
    solver_remote_url = "http://localhost:12345/v1"
    
    # Try to get values from config.env_args if it exists
    if hasattr(config, 'env_args') and config.env_args is not None:
        if hasattr(config.env_args, 'reward_kwargs') and hasattr(config.env_args.reward_kwargs, 'remote_url'):
            solver_remote_url = config.env_args.reward_kwargs.remote_url or solver_remote_url
        if hasattr(config.env_args, 'solver_remote') and hasattr(config.env_args.solver_remote, 'base_url'):
            solver_remote_url = config.env_args.solver_remote.base_url or solver_remote_url
    
    env_args = {
        "reward_fn": rllm_reward_fn_context_assist,
        "reward_kwargs": {
            "solver_model_path": config.actor_rollout_ref.model.path,
            "remote_url": solver_remote_url,
            "remote_api_key": "None",
            "gen": {
                "temperature": config.actor_rollout_ref.rollout.temperature,
                "max_new_tokens": config.env_args.solver_remote.max_tokens,
            },
            "use_marginal_improvement": True,
            "fractional_shaping": False,
            "use_together_code_interpreter": False,
        },
        "solver_remote": {
            "base_url": solver_remote_url,
            "api_key": "None",
            "model": config.actor_rollout_ref.model.path,
            "temperature": config.actor_rollout_ref.rollout.temperature,
            "max_tokens": config.env_args.solver_remote.max_tokens,
        },
        "max_turns": 4,
        "use_shaped_reward": False,
        "reward_bonus_coeff": 0.0,
        "truncate_trace_chars": 2000,
        "observation_key": "problem_text",
    }

    # Agent args (matching run_cm.py)
    agent_args = {
        "remove_cm_thinking": True,
        "system_instruction": "You are an expert programming assistant helping to generate feedback for code generation problems.",
        "use_memory": getattr(config.agent_args, 'use_memory', True) if hasattr(config, 'agent_args') else True,
        "use_solver_cot": getattr(config.agent_args, 'use_solver_cot', False) if hasattr(config, 'agent_args') else False,
    }

    trainer = AgentTrainer(
        agent_class=ContextManagerAgent,
        agent_args=agent_args,
        env_args=env_args,
        env_class=ContextManagerEnv,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
