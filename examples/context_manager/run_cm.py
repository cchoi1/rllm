import argparse
import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict, Any

from transformers import AutoTokenizer

from rllm.agents.context_manager_agent import ContextManagerAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.base.context_manager_env import ContextManagerEnv
from rllm.rewards.cm_reward import rllm_reward_fn_context_assist
# from rllm.rewards.cm_reward_old import rllm_reward_fn_context_assist
from rllm.utils import save_trajectories
from rllm.agents.agent import Trajectory


def parse_args():
    parser = argparse.ArgumentParser(description="Run Context Manager with configurable environment arguments")
    
    # General configuration
    parser.add_argument("--n_parallel_agents", type=int, default=64, help="Number of parallel agents")
    parser.add_argument("--num_turns", type=int, default=4, help="Number of turns per episode")
    parser.add_argument("--K", type=int, default=1, help="Number of repetitions per task for pass@K evaluation")
    parser.add_argument("--model_name", type=str, default="agentica-org/DeepCoder-1.5B-Preview", help="Model name for CM and solver")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Maximum tokens for generation")
    
    # Reward kwargs
    parser.add_argument("--solver_model_path", type=str, default=None, help="Solver model path (defaults to model_name)")
    parser.add_argument("--remote_url", type=str, default="http://localhost:12345/v1", help="Remote API URL for solver")
    parser.add_argument("--remote_api_key", type=str, default="None", help="Remote API key")
    parser.add_argument("--use_solver_cot", action="store_true", help="Use chain-of-thought for solver")
    parser.add_argument("--use_marginal_improvement", action="store_true", default=True, help="Use marginal improvement reward shaping")
    parser.add_argument("--fractional_shaping", action="store_true", help="Use fractional reward shaping")
    parser.add_argument("--use_together_code_interpreter", action="store_true", help="Use Together code interpreter")
    parser.add_argument("--penalize_code_in_feedback", action="store_true", help="Penalize code in feedback")
    parser.add_argument("--code_penalty", type=float, default=0.0, help="Penalty coefficient for code in feedback")
    
    # Solver remote config
    parser.add_argument("--solver_base_url", type=str, default=None, help="Solver base URL (defaults to remote_url)")
    parser.add_argument("--solver_api_key", type=str, default=None, help="Solver API key (defaults to remote_api_key)")
    parser.add_argument("--solver_temperature", type=float, default=0.0, help="Temperature for solver")
    parser.add_argument("--solver_max_tokens", type=int, default=16384, help="Max tokens for solver (defaults to max_tokens)")
    
    # Environment args
    parser.add_argument("--use_shaped_reward", action="store_true", help="Use shaped reward")
    parser.add_argument("--reward_bonus_coeff", type=float, default=0.0, help="Reward bonus coefficient")
    parser.add_argument("--truncate_trace_chars", type=int, default=2000, help="Character limit for trace truncation")
    parser.add_argument("--observation_key", type=str, default="problem_text", help="Key for observation in task dict")
    parser.add_argument("--exclude_code", action="store_true", help="Exclude code from observations")
    
    # CM agent args
    parser.add_argument("--cm_temperature", type=float, default=0.6, help="Temperature for CM agent")
    parser.add_argument("--cm_top_p", type=float, default=0.95, help="Top-p for CM agent")
    parser.add_argument("--remove_cm_thinking", action="store_true", default=True, help="Remove CM thinking tags")
    parser.add_argument("--no_remove_cm_thinking", dest="remove_cm_thinking", action="store_false", help="Keep CM thinking tags")
    parser.add_argument("--system_instruction", type=str, default="You are an expert programming assistant helping to generate feedback for code generation problems.", help="System instruction for CM")
    parser.add_argument("--use_memory", action="store_true", help="Use memory for CM agent")
    parser.add_argument("--cm_use_solver_cot", action="store_true", help="Use solver CoT in CM")
    
    # Engine args
    parser.add_argument("--rollout_base_url", type=str, default="http://localhost:30000/v1", help="Base URL for rollout engine")
    parser.add_argument("--rollout_api_key", type=str, default="None", help="API key for rollout engine")
    parser.add_argument("--max_response_length", type=int, default=16384, help="Maximum response length")
    parser.add_argument("--max_prompt_length", type=int, default=10000, help="Maximum prompt length")
    
    return parser.parse_args()


def extract_cm_prompt_from_chat_completions(chat_completions: List[Dict[str, str]]) -> str:
    """
    Extract the CM prompt from chat_completions messages.
    Returns the last user message content (the main prompt sent to the CM for this turn).
    """
    # Find the last user message (this is the prompt for the current turn)
    for msg in reversed(chat_completions):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def extract_turn_data(trajectories: List[Trajectory]) -> List[Dict[str, Any]]:
    """
    Extract CM prompt, CM feedback, Solver solution, and test results for each turn of each problem.
    
    Returns:
        List of dictionaries, each containing:
        - problem: The problem prompt
        - turns: List of turn data with:
            - turn: Turn number
            - cm_prompt: The CM prompt sent to the context manager
            - cm_feedback: The feedback generated by the CM
            - solver_solution: The solver's code/solution
            - pass_at_1: Whether all tests passed (boolean)
            - passed_tests: Number of tests passed
            - total_tests: Total number of tests
            - verifier_results: Detailed verifier results dictionary
    """
    results = []
    
    for trajectory in trajectories:
        problem_data = {
            "problem": trajectory.task.get("prompt", "") if trajectory.task else "",
            "turns": []
        }
        
        for step_idx, step in enumerate(trajectory.steps):
            # Extract CM prompt from chat_completions (last user message)
            cm_prompt = extract_cm_prompt_from_chat_completions(step.chat_completions)
            
            # Extract CM feedback (action)
            cm_feedback = step.action if step.action else ""
            
            # Extract Solver solution and test results from raw observation dict (stored in step.info)
            solver_solution = ""
            pass_at_1 = False
            passed_tests = 0
            total_tests = 0
            verifier_results = {}
            
            raw_obs = step.info.get('raw_observation') if step.info else None
            if raw_obs and isinstance(raw_obs, dict):
                # Extract solver solution
                solver_solution = raw_obs.get("solver_output", "") or raw_obs.get("solver_full_output", "")
                # Extract test results
                pass_at_1 = bool(raw_obs.get("solved", False))
                passed_tests = raw_obs.get("passed_tests", 0)
                total_tests = raw_obs.get("total_tests", 0)
                verifier_results = raw_obs.get("verifier_results", {})
            elif step.observation and isinstance(step.observation, dict):
                # Fallback: try to get from step.observation if it's a dict
                solver_solution = step.observation.get("solver_output", "") or step.observation.get("solver_full_output", "")
                pass_at_1 = bool(step.observation.get("solved", False))
                passed_tests = step.observation.get("passed_tests", 0)
                total_tests = step.observation.get("total_tests", 0)
                verifier_results = step.observation.get("verifier_results", {})
            
            turn_data = {
                "turn": step_idx + 1,
                "cm_prompt": cm_prompt,
                "cm_feedback": cm_feedback,
                "solver_solution": solver_solution,
                "pass_at_1": pass_at_1,
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "verifier_results": verifier_results,
            }
            
            problem_data["turns"].append(turn_data)
        
        results.append(problem_data)
    
    return results


def save_cm_data_to_json(trajectories: List[Trajectory], filename: str):
    """
    Save CM prompts, feedback, solver solutions, and test results to a JSON file.
    """
    data = extract_turn_data(trajectories)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"CM data saved to: {filename}")
    print(f"Total problems: {len(data)}")
    print(f"Total turns: {sum(len(problem['turns']) for problem in data)}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    args = parse_args()

    n_parallel_agents = args.n_parallel_agents
    num_turns = args.num_turns
    model_name = args.model_name
    MAX_TOKENS = args.max_tokens

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    reward_fn = rllm_reward_fn_context_assist

    # Set defaults for solver_model_path if not provided
    solver_model_path = args.solver_model_path if args.solver_model_path else model_name
    solver_max_tokens = args.solver_max_tokens if args.solver_max_tokens else MAX_TOKENS
    solver_base_url = args.solver_base_url if args.solver_base_url else args.remote_url
    solver_api_key = args.solver_api_key if args.solver_api_key else args.remote_api_key

    env_args = {
        "reward_fn": reward_fn,
        "reward_kwargs": {
            "solver_model_path": solver_model_path,
            "remote_url": args.remote_url,
            "remote_api_key": args.remote_api_key,
            "gen": {
                "temperature": args.solver_temperature,
                "max_new_tokens": solver_max_tokens,
            },
            "use_solver_cot": args.use_solver_cot,
            "use_marginal_improvement": args.use_marginal_improvement,
            "fractional_shaping": args.fractional_shaping,
            "use_together_code_interpreter": args.use_together_code_interpreter,
            "penalize_code_in_feedback": args.penalize_code_in_feedback,
            "code_penalty": args.code_penalty,
        },
        "solver_remote": {
            "base_url": solver_base_url,
            "api_key": solver_api_key,
            "model": solver_model_path,
            "temperature": args.solver_temperature,
            "max_tokens": solver_max_tokens,
        },
        "max_turns": num_turns,
        "use_shaped_reward": args.use_shaped_reward,
        "reward_bonus_coeff": args.reward_bonus_coeff,
        "truncate_trace_chars": args.truncate_trace_chars,
        "observation_key": args.observation_key,
        "exclude_code": args.exclude_code,
    }

    print(f"Env args: {env_args}")

    # Sampling parameters for the ContextManager agent (different from solver)
    sampling_params = {"temperature": args.cm_temperature, "top_p": args.cm_top_p, "model": model_name}

    agent_args = {
        "remove_cm_thinking": args.remove_cm_thinking,
        "system_instruction": args.system_instruction,
        "use_memory": args.use_memory,
        "use_solver_cot": args.cm_use_solver_cot,
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
            "base_url": args.rollout_base_url,
            "api_key": args.rollout_api_key,
        },
        max_response_length=args.max_response_length,
        max_prompt_length=args.max_prompt_length,
        n_parallel_agents=n_parallel_agents,
    )

    test_dataset = DatasetRegistry.load_dataset("lcb", "test")
    tasks = test_dataset.get_data()
    print(f"Loaded {len(tasks)} tasks")
    if not tasks:
        print("No tasks found in dataset!")
        exit(1)

    # Repeat each task K times for pass@K evaluation
    K = args.K
    if K > 1:
        original_tasks = tasks
        tasks = []
        for task in original_tasks:
            for _ in range(K):
                tasks.append(task)
        print(f"Repeated each task {K} times: {len(original_tasks)} tasks -> {len(tasks)} total tasks")

    print(f"Running ContextManager with {len(tasks)} tasks (K={K})")
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
    model_name_short = model_name[model_name.find("/")+1:] if "/" in model_name else model_name
    
    # Calculate original number of unique tasks
    original_num_tasks = len(tasks) // K if K > 1 else len(tasks)
    
    # Save trajectories in original format
    k_suffix = f"_K{K}" if K > 1 else ""
    trajectory_file = f"{model_name_short}_context_manager_trajectories_{num_turns}turns_{original_num_tasks}{k_suffix}_{timestamp}.pt"
    save_trajectories(results, filename=trajectory_file)
    print(f"Results saved to: {trajectory_file}")
    
    # Save CM data to JSON
    json_file = f"./trajectories/{model_name_short}_context_manager_data_{num_turns}turns_{original_num_tasks}{k_suffix}_{timestamp}.json"
    save_cm_data_to_json(results, filename=json_file)