# train_solver_cm_only_openai_solver.py

import hydra
from omegaconf import OmegaConf

from rllm.trainer.agent_trainer import AgentTrainer
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.countdown_reward import countdown_reward_fn

from examples.countdown.solver_cm_train_cm_only_flow2 import (
    SolverContextManagerTrainCMOnlyWorkflow,
)

def sel(cfg, path, default=None):
    try:
        v = OmegaConf.select(cfg, path)
        return default if v is None else v
    except Exception:
        return default

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(cfg):
    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    test_dataset  = DatasetRegistry.load_dataset("countdown", "test")

    # Build a **serializable** spec for the SOLVER OpenAI engine.
    # (Strings, ints, floats, dicts → all picklable. No live clients!)
    model_name = sel(cfg, "actor_rollout_ref.model.path") or sel(cfg, "actor_rollout_ref.model.name") or "gpt-4o-mini"
    max_prompt = int(sel(cfg, "data.max_prompt_length", 2048))
    max_resp   = int(sel(cfg, "data.max_response_length", 1024))
    max_model_length = max_prompt + max_resp

    openai_solver_spec = {
        "model": model_name,
        "max_prompt_length": max_prompt,
        "max_response_length": max_resp,
        "max_model_length": max_model_length,
        "base_url": sel(cfg, "actor_rollout_ref.rollout.openai_base_url", "https://api.openai.com/v1"),
        # generally omit api_key so the worker reads OPENAI_API_KEY from env
        "api_key": sel(cfg, "actor_rollout_ref.rollout.openai_api_key", None),
        "sampling_params": {
            "temperature": float(sel(cfg, "actor_rollout_ref.rollout.temperature", 0.6)),
            "top_p": float(sel(cfg, "actor_rollout_ref.rollout.top_p", 0.95)),
        },
        # keep tokenizer=None so OpenAIEngine uses chat completions path
        "tokenizer": None,
        "accumulate_reasoning": False,
    }

    trainer = AgentTrainer(
        workflow_class=SolverContextManagerTrainCMOnlyWorkflow,
        workflow_args={
            "n_rounds": 2,
            "n_solutions": 1,
            "reward_function": countdown_reward_fn,
            "success_bonus": 1.0,
            "clip_improvement": 1.0,
            # 👇 pass the serializable spec instead of an engine instance
            "rollout_engine_solver_spec": openai_solver_spec,
        },
        config=cfg,  # CM engine still constructed by AgentTrainer from Hydra
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()
