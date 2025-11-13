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

from pathlib import Path

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import code_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer
from solver_cm_flow import DeepCoderCMWorkflow, UnitTestResult, format_test_results


def create_verifier(task_data_source: str = "livecodebench"):
    """Create a verifier function that wraps code_reward_fn and returns UnitTestResult."""
    def verifier(task: dict, code: str) -> UnitTestResult:
        # code_reward_fn will extract code from markdown itself
        # so we pass the full model response (which may contain markdown blocks)
        # Use code_reward_fn to evaluate
        task_info = {
            "data_source": task.get("data_source", task_data_source),
            "ground_truth": task.get("ground_truth"),
            "question": task.get("question") or task.get("problem") or task.get("prompt"),
        }
        
        reward_output = code_reward_fn(task_info, code)
        metadata = reward_output.metadata
        
        # Extract test results from metadata
        test_results = metadata.get("test_results", [])
        passed = metadata.get("passed_tests", 0)
        total = metadata.get("total_tests", 0)
        
        # Format feedback using format_test_results if test_results available
        if test_results:
            feedback = format_test_results(test_results)
        else:
            # Fallback feedback
            all_passed = metadata.get("all_passed", False)
            if all_passed:
                feedback = "All tests passed!"
            else:
                feedback = f"Tests passed: {passed}/{total}"
                if metadata.get("error"):
                    feedback += f"\nError: {metadata['error']}"
        
        return UnitTestResult(
            passed=passed,
            total=total,
            feedback=feedback,
            test_results=test_results if test_results else None,
        )
    
    return verifier


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("lcb", "train")
    if train_dataset is None:
        print("Train dataset 'lcb' not found in registry.")
        print("Available datasets:", DatasetRegistry.list_datasets())
        return
    
    test_dataset = DatasetRegistry.load_dataset("lcb", "test")
    if test_dataset is None:
        print("Test dataset 'lcb' not found in registry.")
        print("Available datasets:", DatasetRegistry.list_datasets())
        return

    # Create verifier function
    verifier = create_verifier("livecodebench")
    
    # Workflow args
    # Note: During training, rollout_engine will be provided by the trainer (VerlEngine)
    # and passed to the workflow. The workflow will use rollout_engine for both CM (trainable)
    # and Solver (non-trainable, marked with info={"trainable": False}). If you want a separate
    # solver_engine, you can pass it via config.workflow_args.solver_engine.
    workflow_args = {
        "verifier": verifier,
        "n_turns": 4,
        "language": "python",
        "exclude_truncated": True,  # Exclude truncated responses from prompts
        # solver_engine is optional - if not provided, will use rollout_engine
        # Solver steps are marked as non-trainable via info={"trainable": False}
    }
    
    trainer = AgentTrainer(
        workflow_class=DeepCoderCMWorkflow,
        workflow_args=workflow_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()

