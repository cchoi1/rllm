import asyncio
import json
import os
from copy import deepcopy
from typing import Dict

from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine import OpenAIEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.rewards.reward_fn import code_reward_fn
from solver_cm_knowledge_flow import DeepCoderKnowledgeFlowWorkflow, UnitTestResult, format_test_results


def create_verifier(task_data_source: str):
    """Create a verifier function that wraps code_reward_fn and returns UnitTestResult."""
    def verifier(task: Dict, code: str) -> UnitTestResult:
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


def load_data(n=1, split="test"):
    """Load DeepCoder data using the Dataset interface."""
    dataset = DatasetRegistry.load_dataset("lcb", split)
    if dataset is None:
        print(f"Dataset 'lcb' split '{split}' not found in registry.")
        print("Available datasets:", DatasetRegistry.list_datasets())
        return []
    
    data = []
    for idx, example in enumerate(dataset):
        processed = process_task(example, idx)
        for i in range(n):
            data.append(deepcopy(processed))
    return data


def process_task(example, idx):
    """Process example into the expected format."""
    question = example.get("question", "")
    ground_truth = example.get("ground_truth", example.get("tests", {}))
    data_source = example.get("data_source", "livecodebench")
    
    task = {
        "question": question,
        "ground_truth": ground_truth,
        "idx": idx,
        "data_source": data_source,
    }
    return task


def exclude_token_ids(data):
    """Recursively remove prompt_ids and completion_ids from the data structure."""
    if isinstance(data, dict):
        # Create a new dict without prompt_ids and completion_ids
        result = {}
        for key, value in data.items():
            if key not in ["prompt_ids", "completion_ids"]:
                result[key] = exclude_token_ids(value)
        return result
    elif isinstance(data, list):
        # Recursively process list items
        return [exclude_token_ids(item) for item in data]
    else:
        # Return other types as-is
        return data


def print_prompts_and_responses(results):
    """Print solver and knowledge manager prompts and responses at each turn."""
    for episode_idx, episode in enumerate(results):
        print("\n" + "=" * 80)
        print(f"EPISODE {episode_idx + 1}")
        print("=" * 80)
        
        problem = episode.task.get("question", episode.task.get("problem", episode.task.get("prompt", "")))
        print(f"\nProblem:\n{problem}")
        
        # Track turns: Initial solver (turn 0), then KM + Solver pairs for each turn
        turn = 0
        solver_turn = 0
        for traj_idx, trajectory in enumerate(episode.trajectories):
            if trajectory.name == "solver":
                print(f"\n{'='*80}")
                if solver_turn == 0:
                    print("SOLVER - Initial Attempt")
                else:
                    print(f"SOLVER - Turn {solver_turn}")
                print("="*80)
                solver_turn += 1
                
                if trajectory.steps:
                    step = trajectory.steps[0]
                    # Print prompt (last user message)
                    user_messages = [msg for msg in step.chat_completions if msg.get("role") == "user"]
                    if user_messages:
                        print(f"\n📝 Prompt:")
                        print("-" * 80)
                        prompt = user_messages[-1].get("content", "")
                        print(prompt)
                    
                    # Print response (last assistant message)
                    assistant_messages = [msg for msg in step.chat_completions if msg.get("role") == "assistant"]
                    if assistant_messages:
                        print(f"\n🤖 Response:")
                        print("-" * 80)
                        response = assistant_messages[-1].get("content", "")
                        print(response)
                    
                    # Print action (code)
                    if step.action:
                        print(f"\n💻 Code (Action):")
                        print("-" * 80)
                        action = str(step.action)
                        print(action)
                    
                    if step.reward != 0.0:
                        print(f"\n💰 Reward: {step.reward:.4f}")
            
            elif trajectory.name == "knowledge_manager":
                turn += 1  # Increment turn for each KM (starts at 1)
                print(f"\n{'='*80}")
                print(f"KNOWLEDGE MANAGER - Turn {turn}")
                print("="*80)
                
                if trajectory.steps:
                    step = trajectory.steps[0]
                    # Print prompt (last user message)
                    user_messages = [msg for msg in step.chat_completions if msg.get("role") == "user"]
                    if user_messages:
                        print(f"\n📝 Prompt:")
                        print("-" * 80)
                        prompt = user_messages[-1].get("content", "")
                        print(prompt)
                    
                    # Print response (last assistant message)
                    assistant_messages = [msg for msg in step.chat_completions if msg.get("role") == "assistant"]
                    if assistant_messages:
                        print(f"\n🤖 Response:")
                        print("-" * 80)
                        response = assistant_messages[-1].get("content", "")
                        print(response)
                    
                    # Print action (knowledge list)
                    if step.action:
                        print(f"\n📚 Knowledge List (Action):")
                        print("-" * 80)
                        if isinstance(step.action, list):
                            for i, knowledge_item in enumerate(step.action, 1):
                                print(f"{i}. {knowledge_item}")
                        else:
                            print(str(step.action))
                    
                    if step.reward != 0.0:
                        print(f"\n💰 Reward: {step.reward:.4f}")
        
        print(f"\n{'='*80}")
        print(f"Episode {episode_idx + 1} Result: {'✅ CORRECT' if episode.is_correct else '❌ INCORRECT'}")
        if episode.metrics:
            print(f"Metrics: {episode.metrics}")
        print("="*80 + "\n")


def evaluate_results(results):
    """Evaluate the results and compute cumulative pass@1 at each turn.
    
    This function deduplicates by problem and tracks when each unique problem
    first passes, then computes cumulative pass@1 at each turn.
    """
    import hashlib
    
    if not results:
        print("No results to evaluate.")
        return 0.0
    
    # Find maximum turn number by checking all metrics
    max_turn = 0
    for episode in results:
        if episode.metrics:
            for key in episode.metrics.keys():
                if key.startswith("pass_ratio_t"):
                    try:
                        turn_num = int(key.replace("pass_ratio_t", ""))
                        max_turn = max(max_turn, turn_num)
                    except ValueError:
                        pass
    
    # Deduplicate by problem: track the first turn at which each unique problem passes
    # Use question string as the problem identifier (similar to rllm/utils.py)
    problem_first_pass_turn = {}  # problem_hash -> first turn where it passes, or None if never
    problem_seen = set()  # Track which problems we've seen
    
    for episode in results:
        if not episode.metrics:
            continue
        
        # Create problem identifier from task
        task = episode.task
        problem_str = task.get("question") or task.get("problem") or task.get("prompt", "")
        problem_hash = hashlib.md5(problem_str.encode()).hexdigest()
        
        # Only process each problem once (take the first episode for each problem)
        if problem_hash in problem_seen:
            continue
        problem_seen.add(problem_hash)
        
        # Find the first turn at which this problem passes
        first_pass_turn = None
        for t in range(max_turn + 1):
            ratio_key = f"pass_ratio_t{t}"
            if ratio_key in episode.metrics:
                if episode.metrics[ratio_key] >= 1.0:
                    first_pass_turn = t
                    break
        
        problem_first_pass_turn[problem_hash] = first_pass_turn
    
    # Compute cumulative pass@1 at each turn
    total_unique_problems = len(problem_first_pass_turn)
    cumulative_pass_at_1 = {}
    
    for turn in range(max_turn + 1):
        # Count how many unique problems passed at or before this turn
        passed_count = sum(
            1 for first_pass_turn in problem_first_pass_turn.values()
            if first_pass_turn is not None and first_pass_turn <= turn
        )
        
        cumulative_pass_at_1[f"turn_{turn}"] = (
            passed_count / total_unique_problems if total_unique_problems > 0 else 0.0
        )
    
    # Print results
    print("=" * 60)
    print("📊 EVALUATION RESULTS - Cumulative Pass@1 at Each Turn")
    print("=" * 60)
    print(f"Total unique problems: {total_unique_problems}")
    print(f"Total episodes: {len(results)}")
    print(f"Maximum turns: {max_turn}")
    print("\nCumulative Pass@1 (per unique problem):")
    for turn in range(max_turn + 1):
        turn_key = f"turn_{turn}"
        pass_at_1 = cumulative_pass_at_1[turn_key]
        passed_count = sum(
            1 for first_pass_turn in problem_first_pass_turn.values()
            if first_pass_turn is not None and first_pass_turn <= turn
        )
        print(f"  Turn {turn}: {pass_at_1:.4f} ({pass_at_1*100:.2f}%) - {passed_count}/{total_unique_problems} problems passed")
    
    # Return final pass@1
    final_turn_key = f"turn_{max_turn}"
    final_pass_at_1 = cumulative_pass_at_1[final_turn_key]
    
    print("=" * 60)
    
    return final_pass_at_1


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Configuration
    n_parallel_tasks = 32
    n_turns = 4  # Number of KM↔Solver interaction turns
    split = "test"  # "train" or "test"
    exclude_truncated = True  # If True, exclude truncated responses from prompts

    # Model configuration - both can use the same model or different models
    solver_model_name = "agentica-org/DeepCoder-1.5B-Preview"
    km_model_name = "agentica-org/DeepCoder-1.5B-Preview"  # KM uses trainable engine
    
    tokenizer = AutoTokenizer.from_pretrained(solver_model_name)

    # Create verifier function
    verifier = create_verifier("livecodebench")

    # Create solver engine (frozen - not trainable)
    solver_engine = OpenAIEngine(
        model=solver_model_name,
        tokenizer=tokenizer,
        max_prompt_length=4096,
        max_response_length=16384,
        base_url="http://localhost:30000/v1",
        api_key="None",
        sampling_params={"temperature": 0.0, "top_p": 0.95},
    )

    # Create KM engine (trainable - this is what gets trained)
    km_engine = OpenAIEngine(
        model=km_model_name,
        tokenizer=tokenizer,
        max_prompt_length=4096,
        max_response_length=16384,
        base_url="http://localhost:30000/v1",
        api_key="None",
        sampling_params={"temperature": 0.6, "top_p": 0.95},
    )

    # Create workflow engine
    # Note: AgentWorkflowEngine passes rollout_engine as a keyword arg to the workflow.
    # The workflow now handles both rollout_engine and km_engine from workflow_args.
    
    engine = AgentWorkflowEngine(
        workflow_cls=DeepCoderKnowledgeFlowWorkflow,
        workflow_args={
            "solver_engine": solver_engine,
            "verifier": verifier,
            "n_turns": n_turns,
            "language": "python",
            "km_engine": km_engine,  # Pass km_engine via workflow_args
            "exclude_truncated": exclude_truncated,  # Exclude truncated responses from prompts
        },
        rollout_engine=km_engine,  # Also passed as rollout_engine, workflow will use km_engine if provided
        config=None,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=1,
    )

    # Load tasks
    print("Loading tasks...")
    tasks = load_data(n=1, split=split)
    tasks = tasks[:4]
    print(f"Loaded {len(tasks)} tasks")
    
    if not tasks:
        print("No tasks loaded. Exiting.")
        exit(1)

    # Execute workflow
    print(f"Executing workflow with {n_turns} turns per episode...")
    results = asyncio.run(engine.execute_tasks(tasks))

    # Print prompts and responses for each episode
    print("\n" + "="*80)
    print("PRINTING PROMPTS AND RESPONSES")
    print("="*80)
    print_prompts_and_responses(results)

    # Evaluate results
    print("\nEvaluating results...")
    pass_at_1 = evaluate_results(results)

    # Save results (excluding prompt_ids and completion_ids to reduce file size)
    os.makedirs("logs", exist_ok=True)
    output_file = f"logs/solver_cm_knowledge_flow_lcb{split}_{len(tasks)}.json"
    episodes_dict = [exclude_token_ids(episode.to_dict()) for episode in results]
    with open(output_file, "w") as f:
        json.dump(episodes_dict, f, indent=4)

    print(f"\n✅ Results saved to {output_file}")

