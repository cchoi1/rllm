import asyncio
import os
from datetime import datetime

from transformers import AutoTokenizer

from rllm.agents.context_manager_agent import ContextManagerAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.base.context_manager_env import ContextManagerEnv
from rllm.rewards.cm_reward import rllm_reward_fn_context_assist
from rllm.utils import save_trajectories

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 10

    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    MAX_TOKENS = 8192
    # model_name = "agentica-org/DeepCoder-1.5B-Preview"
    # MAX_TOKENS = 16384

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
        max_prompt_length=8192,
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

# import asyncio
# import os
# import time
# from datetime import datetime

# from transformers import AutoTokenizer

# from rllm.agents.context_manager_agent import ContextManagerAgent
# from rllm.data.dataset import DatasetRegistry
# from rllm.engine.agent_execution_engine import AgentExecutionEngine
# from rllm.environments.base.context_manager_env import ContextManagerEnv
# from rllm.rewards.cm_reward import rllm_reward_fn_context_assist
# from rllm.utils import save_trajectories

# if __name__ == "__main__":
#     os.environ["TOKENIZERS_PARALLELISM"] = "true"

#     n_parallel_agents = 8

#     model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
#     MAX_TOKENS = 8192
#     # model_name = "agentica-org/DeepCoder-1.5B-Preview"
#     # MAX_TOKENS = 16384
#     NUM_GPUS = 1

#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     reward_fn = rllm_reward_fn_context_assist

#     # --- eight local replicas on ports 8000..8007 ---
#     # remote_urls = [f"http://tiger8.stanford.edu:{8000+i}/v1" for i in range(NUM_GPUS)]
#     remote_urls = [f"http://tiger7.stanford.edu:{8000+i}/v1" for i in range(NUM_GPUS)]
#     legacy_url = remote_urls[0]  # keep single for BC

#     env_args = {
#         "reward_fn": reward_fn,
#         "reward_kwargs": {
#             "solver_model_path": model_name,
#             # multi-endpoint preferred:
#             "remote_urls": remote_urls,
#             # keep single for BC:
#             "remote_url": legacy_url,
#             "remote_api_key": "None",
#             "gen": {
#                 "temperature": 0.2,
#                 "max_new_tokens": MAX_TOKENS,
#             },
#             # optional tuning:
#             "timeout_s": 60.0,
#             "max_retries": 3,
#         },
#         "solver_remote": {
#             # multi-endpoint for env-side defaults/telemetry:
#             "base_urls": remote_urls,
#             # keep single for BC:
#             "base_url": legacy_url,
#             "api_key": "None",
#             "model": model_name,
#             "temperature": 0.2,
#             "max_tokens": MAX_TOKENS,
#         },
#         "max_turns": 4,
#         "use_shaped_reward": False,
#         "reward_bonus_coeff": 0.0,
#         "truncate_trace_chars": 2000,
#     }

#     # Sampling parameters for the ContextManager agent (different from solver)
#     sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

#     agent_args = {
#         "remove_cm_thinking": True,
#         "system_instruction": "You are an expert programming assistant helping to generate feedback for code generation problems.",
#         "use_memory": False,
#         "use_solver_cot": True,
#     }

#     engine = AgentExecutionEngine(
#         agent_class=ContextManagerAgent,
#         env_class=ContextManagerEnv,
#         agent_args=agent_args,
#         env_args=env_args,
#         engine_name="openai",
#         tokenizer=tokenizer,
#         sampling_params=sampling_params,
#         rollout_engine_args={
#             # "base_url": "http://localhost:30000/v1",
#             "base_url": remote_urls[0],
#             "api_key": "None",
#         },
#         max_response_length=MAX_TOKENS,
#         max_prompt_length=8192,
#         n_parallel_agents=n_parallel_agents,
#     )

#     test_dataset = DatasetRegistry.load_dataset("lcb", "test")
#     tasks = test_dataset.get_data()
#     tasks = tasks[:16]

#     if not tasks:
#         print("No tasks found in dataset!")
#         exit(1)

#     print(f"Running ContextManager with {len(tasks)} tasks")
#     print(f"Model: {model_name}")
#     print(f"Environment args: {env_args}")

#     # Start timing
#     start_time = time.time()
    
#     try:
#         results = asyncio.run(engine.execute_tasks(tasks))
#     except Exception as e:
#         print(f"Error during execution: {e}")
#         import traceback
#         traceback.print_exc()
#         exit(1)
    
#     # End timing and calculate duration
#     end_time = time.time()
#     total_time = end_time - start_time
    
#     print(f"\n=== Execution Summary ===")
#     print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
#     print(f"Tasks completed: {len(tasks)}")
#     print(f"Average time per task: {total_time/len(tasks):.2f} seconds")
#     print(f"Parallel agents: {n_parallel_agents}")
#     print(f"========================\n")
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     save_trajectories(results, filename=f"context_manager_trajectories_{len(tasks)}_{timestamp}.pt")
#     print(f"Results saved to: context_manager_trajectories_{len(tasks)}_{timestamp}.pt")
