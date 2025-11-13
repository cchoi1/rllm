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

# ───── NEW: wandb rewind patch ─────
import wandb

_orig_init = wandb.init

def _patched_init(*args, **kwargs):
    """
    If WANDB_RUN_ID and WANDB_REWIND_STEP are set, force wandb to
    rewind/truncate to that step before rllm starts logging.
    """
    run_id = os.environ.get("WANDB_RUN_ID")
    rewind_step = os.environ.get("WANDB_REWIND_STEP")
    if run_id and rewind_step:
        # this is the SDK way to truncate steps > rewind_step
        kwargs["resume_from"] = f"{run_id}?_step={rewind_step}"
        print(f"Rewinding to step {rewind_step} for run {run_id}")
    # you can still pass normal WANDB_RESUME etc. via env if rllm sets them
    return _orig_init(*args, **kwargs)

wandb.init = _patched_init
# ─── end patch ───

from pathlib import Path

from rllm.trainer.agent_trainer import AgentTrainer
from rllm.agents.context_manager_agent import ContextManagerAgent
from rllm.environments.base.context_manager_env import ContextManagerEnv
from rllm.rewards.cm_reward import (
    rllm_reward_fn_context_assist
)
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


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # Register chunked DeepCoder dataset
    train_dataset, _ = register_deepcoder_chunked_dataset()
    test_dataset = DatasetRegistry.load_dataset("lcb", "test")
    
    if train_dataset is None:
        print("Failed to register chunked dataset. Exiting.")
        return
    
    # Build env args (matching run_cm.py)
    # Resolve remote solver settings safely
    default_base_url = "http://localhost:12345/v1"
    default_model_name = "agentica-org/DeepCoder-1.5B-Preview"
    default_max_tokens = 16384

    solver_remote_cfg = getattr(getattr(config, "env_args", None), "solver_remote", None)
    solver_remote_url = getattr(solver_remote_cfg, "base_url", None) or default_base_url
    solver_remote_model = getattr(solver_remote_cfg, "model", None) or default_model_name
    solver_remote_api_key = getattr(solver_remote_cfg, "api_key", None) or "None"
    solver_remote_max_tokens = getattr(solver_remote_cfg, "max_tokens", None) or default_max_tokens
    solver_remote_temperature = getattr(solver_remote_cfg, "temperature", None) or 0.0
    penalize_code_in_feedback = getattr(getattr(config, "env_args", None), "penalize_code_in_feedback", None) or False
    code_penalty = getattr(getattr(config, "env_args", None), "code_penalty", None) or 0.0
    exclude_code = getattr(getattr(config, "env_args", None), "exclude_code", None) or False

    env_args = {
        "reward_fn": rllm_reward_fn_context_assist,
        "reward_kwargs": {
            "solver_model_path": solver_remote_model,
            "remote_url": solver_remote_url,
            "remote_api_key": solver_remote_api_key,
            "timeout_s": 600.0,
            "gen": {
                "temperature": solver_remote_temperature,
                "max_tokens": solver_remote_max_tokens,
            },
            "use_solver_cot": False,
            "use_marginal_improvement": True,
            "fractional_shaping": False,
            "use_together_code_interpreter": False,
            "penalize_code_in_feedback": penalize_code_in_feedback,
            "code_penalty": code_penalty,
        },
        "solver_remote": {
            "base_url": solver_remote_url,
            "api_key": solver_remote_api_key,
            "model": solver_remote_model,
            "temperature": solver_remote_temperature,
            "max_tokens": solver_remote_max_tokens,
        },
        "max_turns": 4,
        "use_shaped_reward": False,
        "reward_bonus_coeff": 0.0,
        "truncate_trace_chars": 2000,
        "observation_key": "problem_text",
        "exclude_code": exclude_code,
    }
    print(f"Env args: {env_args}")

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