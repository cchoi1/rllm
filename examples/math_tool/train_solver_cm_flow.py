"""
Training script for Math Context Manager workflow.

This script trains a ContextManager (CM) that provides feedback to a Solver on math problems.

By default, the Solver uses the same engine as the CM (but solver steps are marked as non-trainable).
To use a separate frozen model for the Solver, set `solver_model_path` in your config:

    python train_solver_cm_flow.py solver_model_path=/path/to/solver/model ...

This will create a separate VerlEngine for the solver with the specified model, while the CM
continues to use the trainable engine specified in `actor_rollout_ref.model.path`.

Note: Using a separate solver engine requires additional GPU memory to load both models.
Make sure you have sufficient GPU resources or adjust your resource allocation accordingly.
"""
import asyncio
import multiprocessing as mp
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

import uuid
from collections import defaultdict

import numpy as np
from verl import DataProto

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer
from examples.math_tool.solver_cm_flow import MathCMWorkflow, MathResult, SumOfProblemsWorkflow


def create_verifier(task_data_source: str = "math"):
    """Create a verifier function that wraps math_reward_fn and returns MathResult."""
    def verifier(task: dict, solution: str) -> MathResult:
        # Use math_reward_fn to evaluate
        task_info = {
            "data_source": task.get("data_source", task_data_source),
            "ground_truth": task.get("ground_truth"),
            "question": task.get("question") or task.get("problem") or task.get("prompt"),
        }
        
        reward_output = math_reward_fn(task_info, solution)
        
        # Create feedback message
        if reward_output.is_correct:
            feedback = "Your solution is correct!"
        else:
            feedback = "Your solution is incorrect. Please review your calculations and reasoning."
            if reward_output.metadata:
                error_hint = reward_output.metadata.get("error", "")
                if error_hint:
                    feedback += f" Error: {error_hint}"
        
        return MathResult(
            is_correct=reward_output.is_correct,
            reward=reward_output.reward,
            feedback=feedback,
            metadata=reward_output.metadata if reward_output.metadata else {},
        )
    
    return verifier


class MathCMWorkflowTrainer(AgentWorkflowPPOTrainer):
    """Custom trainer that logs validation metrics with val/test_score/unknown format."""
    
    def _validate_agent(self):
        is_correct_lst = []
        uid_lst = []
        
        # Get n_val_samples from config
        n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["task_ids"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)

            test_batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=["raw_prompt_ids"])
            test_batch.meta_info = {"validate": True}

            test_output_gen_batch = self.generate_trajectories(batch=test_batch)
            repeat_counts = test_output_gen_batch.meta_info["repeat_counts"]
            test_batch = test_batch.sample_level_repeat(repeat_counts)
            test_output_gen_batch.meta_info.pop("repeat_counts", None)
            test_batch = test_batch.union(test_output_gen_batch)

            seen_episodes = set()
            selected_idxs = []
            for i, episode_id in enumerate(test_batch.non_tensor_batch["episode_ids"]):
                if episode_id not in seen_episodes:
                    seen_episodes.add(episode_id)
                    selected_idxs.append(i)
            test_batch = test_batch.select_idxs(selected_idxs)

            is_correct_lst.extend(test_batch.non_tensor_batch["is_correct"])
            uid_lst.extend(test_batch.non_tensor_batch["task_ids"])

        metrics = {}
        is_correct_array = np.array(is_correct_lst)
        uid_array = np.array(uid_lst)

        # Compute pass@1 and pass@k aggregated across all data sources
        pass_rates = defaultdict(list)
        for is_correct, uid in zip(is_correct_array, uid_array, strict=False):
            pass_rates[uid].append(is_correct)

        # Use test_score/unknown format
        # Only log pass@1 after n_turns
        metrics["val/test_score/unknown"] = np.mean(is_correct_array)
        
        # Compute pass@k if multiple samples
        if n_val_samples > 1:
            metrics[f"val/test_score/pass@k/unknown"] = np.mean([1 if any(pass_rate) else 0 for pass_rate in pass_rates.values()])

        return metrics


# @hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
# def main(config):
#     train_dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
#     if train_dataset is None:
#         print("Dataset not found, preparing dataset...")
#         from prepare_math_data import prepare_math_data
#         train_dataset, _ = prepare_math_data()
    
#     test_dataset = DatasetRegistry.load_dataset("aime2024", "test")
#     if test_dataset is None:
#         print("Dataset not found, preparing dataset...")
#         from prepare_math_data import prepare_math_data
#         _, test_dataset = prepare_math_data()

#     # Create verifier function
#     verifier = create_verifier("math")

#     # Monkey-patch AgentWorkflowPPOTrainer to use our custom validation and solver engine creation
#     from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer
    
#     # Store the original methods
#     original_validate = AgentWorkflowPPOTrainer._validate_agent
#     original_init_workers = AgentWorkflowPPOTrainer.init_workers
    
#     # Create wrapper functions
#     def custom_validate_agent(self):
#         return MathCMWorkflowTrainer._validate_agent(self)
    
#     def custom_init_workers(self):
#         # First, call super().init_workers() to initialize base workers
#         # This sets up actor_rollout_wg and other worker groups
#         from verl.trainer.ppo.ray_trainer import RayPPOTrainer
#         RayPPOTrainer.init_workers(self)
        
#         # Check if we need to create a separate solver engine
#         solver_model_path = getattr(self.config, "solver_model_path", None)
        
#         if solver_model_path is not None:
#             print(f"Creating separate solver engine with model: {solver_model_path}")
            
#             # Import necessary modules
#             from omegaconf import OmegaConf
#             from verl.experimental.agent_loop import AgentLoopManager
#             from verl.trainer.ppo.ray_trainer import (
#                 RayClassWithInitArgs,
#                 create_colocated_worker_cls,
#                 Role,
#             )
#             from rllm.engine.rollout.verl_engine import VerlEngine
#             from verl.utils.fs import copy_to_local
#             from verl.utils import hf_tokenizer
            
#             # Create a copy of config with solver model path
#             solver_config = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
#             solver_config.actor_rollout_ref.model.path = solver_model_path
            
#             # Download solver model and create tokenizer
#             solver_local_path = copy_to_local(solver_model_path, use_shm=solver_config.actor_rollout_ref.model.get("use_shm", False))
#             trust_remote_code = solver_config.data.get("trust_remote_code", False)
#             solver_tokenizer = hf_tokenizer(solver_local_path, trust_remote_code=trust_remote_code)
            
#             # Create solver worker group
#             # Reuse the same resource pool but create a separate worker group
#             solver_actor_rollout_cls = RayClassWithInitArgs(
#                 cls=self.role_worker_mapping[Role.ActorRollout],
#                 config=solver_config.actor_rollout_ref,
#                 role="actor_rollout"
#             )
            
#             # Use the same resource pool as the main trainer
#             solver_resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
#             solver_class_dict = {"actor_rollout": solver_actor_rollout_cls}
#             solver_worker_dict_cls = create_colocated_worker_cls(class_dict=solver_class_dict)
            
#             # Create solver worker group with unique name prefix
#             solver_wg_kwargs = {
#                 "name_prefix": f"{self.actor_rollout_wg.name_prefix}_solver",
#                 "config": solver_config,
#                 "tokenizer": solver_tokenizer,
#                 "device_name": self.device_name,
#             }
#             solver_wg_dict = self.ray_worker_group_cls(
#                 resource_pool=solver_resource_pool,
#                 ray_cls_with_init=solver_worker_dict_cls,
#                 **solver_wg_kwargs,
#             )
#             solver_actor_rollout_wg = solver_wg_dict.spawn(prefix_set=["actor_rollout"])["actor_rollout"]
#             solver_actor_rollout_wg.init_model()
            
#             # Create solver AsyncLLMServerManager
#             solver_async_rollout_manager = AgentLoopManager(
#                 config=solver_config,
#                 worker_group=solver_actor_rollout_wg,
#             )
            
#             # Create solver VerlEngine
#             solver_engine = VerlEngine(
#                 config=solver_config,
#                 rollout_manager=solver_async_rollout_manager,
#                 tokenizer=solver_tokenizer,
#             )
            
#             # Update workflow_args to include solver_engine
#             self.workflow_args = self.workflow_args or {}
#             self.workflow_args["solver_engine"] = solver_engine
            
#             # Store solver engine for cleanup
#             self.solver_engine = solver_engine
#             self.solver_async_rollout_manager = solver_async_rollout_manager
#             self.solver_actor_rollout_wg = solver_actor_rollout_wg
        
#         # Now create the workflow engine (the rest of AgentWorkflowPPOTrainer.init_workers)
#         from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
#         from rllm.engine.rollout.verl_engine import VerlEngine
        
#         rollout_engine = VerlEngine(
#             config=self.config,
#             rollout_manager=self.async_rollout_manager,
#             tokenizer=self.tokenizer,
#         )
        
#         self.agent_execution_engine = AgentWorkflowEngine(
#             workflow_cls=self.workflow_class,
#             workflow_args=self.workflow_args,
#             rollout_engine=rollout_engine,
#             config=self.config,
#             n_parallel_tasks=self.config.rllm.workflow.n_parallel_tasks,
#             retry_limit=self.config.rllm.workflow.retry_limit,
#         )
        
#         # init workflow workers
#         import asyncio
#         asyncio.run_coroutine_threadsafe(self.agent_execution_engine.initialize_pool(), self._loop).result()
    
#     # Replace with our custom methods
#     AgentWorkflowPPOTrainer._validate_agent = custom_validate_agent
#     AgentWorkflowPPOTrainer.init_workers = custom_init_workers
    
#     # Check if a separate solver model is specified in config
#     # If config.solver_model_path is set, the trainer will create a separate frozen solver engine
#     # Otherwise, the solver will use the same engine as the CM (but solver steps are marked as non-trainable)
    
#     # Workflow args
#     # Note: During training, rollout_engine will be provided by the trainer (VerlEngine)
#     # and passed to the workflow. If solver_model_path is set in config, a separate solver_engine
#     # will be created and passed via workflow_args. Otherwise, the workflow will use rollout_engine
#     # for both CM (trainable) and Solver (non-trainable, marked with info={"trainable": False} and reward=0.0).
#     trainer = AgentTrainer(
#         workflow_class=MathCMWorkflow,
#         workflow_args={
#             "verifier": verifier,
#             "n_turns": 4,
#             "use_tools": True,
#             # solver_engine will be created automatically if config.solver_model_path is set
#             # Otherwise, solver will use rollout_engine (solver steps are marked as non-trainable)
#         },
#         config=config,
#         train_dataset=train_dataset,
#         val_dataset=test_dataset,
#     )
    
#     try:
#         trainer.train()
#     finally:
#         # Restore original methods
#         AgentWorkflowPPOTrainer._validate_agent = original_validate
#         AgentWorkflowPPOTrainer.init_workers = original_init_workers

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")

    verifier = create_verifier("math")
    workflow_args = {
        "verifier": verifier,
        "n_turns": 4,
        "use_tools": True,
    }
    # Select workflow:
    # - Default: MathCMWorkflow (CM -> feedback, Solver -> refine)
    # - Sum mode: SumOfProblemsWorkflow (CM sums answers over N turns)
    sum_mode = bool(getattr(config, "sum_mode", False)) or os.environ.get("CM_SUM_MODE") == "1"
    workflow_class = SumOfProblemsWorkflow if sum_mode else MathCMWorkflow

    trainer = AgentTrainer(
        workflow_class=workflow_class,
        workflow_args=workflow_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
