import asyncio
import os
from datetime import datetime

from transformers import AutoTokenizer

from rllm.agents.context_manager_agent import ContextManagerAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.base.context_manager_env import ContextManagerEnv
# from rllm.rewards.cm_reward import rllm_reward_fn_context_assist
from rllm.rewards.cm_reward_old import rllm_reward_fn_context_assist
from rllm.utils import save_trajectories

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 10

    # model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    # MAX_TOKENS = 8192
    model_name = "agentica-org/DeepCoder-1.5B-Preview"
    MAX_TOKENS = 16384

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    reward_fn = rllm_reward_fn_context_assist

    env_args = {
        "reward_fn": reward_fn,
        "reward_kwargs": {
            "solver_model_path": model_name,
            "remote_url": "http://localhost:12345/v1",
            "remote_api_key": "None",
            "gen": {
                "temperature": 0.2,
                "max_new_tokens": MAX_TOKENS,
            },
        },
        "solver_remote": {
            "base_url": "http://localhost:12345/v1",
            "api_key": "None",
            "model": model_name,
            "temperature": 0.2,
            "max_tokens": MAX_TOKENS,
        },
        "max_turns": 4,
        "use_shaped_reward": False,
        "reward_bonus_coeff": 0.0,
        "truncate_trace_chars": 2000,
    }

    # Sampling parameters for the ContextManager agent (different from solver)
    sampling_params = {"temperature": 1.0, "top_p": 0.95, "model": model_name}

    agent_args = {
        "remove_cm_thinking": True,
        "system_instruction": "You are an expert programming assistant helping to generate feedback for code generation problems.",
        "use_memory": False,
        "use_solver_cot": True,
    }

    engine = AgentExecutionEngine(
        agent_class=ContextManagerAgent,
        env_class=ContextManagerEnv,
        agent_args=agent_args,
        env_args=env_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=MAX_TOKENS,
        max_prompt_length=32768,
        n_parallel_agents=n_parallel_agents,
    )

    test_dataset = DatasetRegistry.load_dataset("lcb", "test")
    tasks = test_dataset.get_data()
    
    if not tasks:
        print("No tasks found in dataset!")
        exit(1)

    print(f"Running ContextManager with {len(tasks)} tasks")
    print(f"Model: {model_name}")
    print(f"Environment args: {env_args}")
    
    try:
        results = asyncio.run(engine.execute_tasks(tasks))
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_name[model_name.find("/")+1:]
    save_trajectories(results, filename=f"{model_name}_context_manager_trajectories_{len(tasks)}_{timestamp}.pt")
    print(f"Results saved to: {model_name}_context_manager_trajectories_{len(tasks)}_{timestamp}.pt")