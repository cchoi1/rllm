import asyncio
import json
import os
from copy import deepcopy
from typing import Dict

from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine import OpenAIEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.rewards.reward_fn import math_reward_fn
from examples.math_tool.solver_cm_knowledge_flow import MathCMKnowledgeWorkflow, MathResult


def create_verifier(task_data_source: str = "math"):
    """Create a verifier function that wraps math_reward_fn and returns MathResult."""
    def verifier(task: Dict, solution: str) -> MathResult:
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


def load_data(n=1, split="test"):
    """Load math data using the Dataset interface."""
    dataset = DatasetRegistry.load_dataset("aime2024", split)
    if dataset is None:
        print(f"Dataset 'aime2024' split '{split}' not found in registry.")
        print("Available datasets:", DatasetRegistry.list_datasets())
        print("Preparing dataset...")
        from prepare_math_data import prepare_math_data

        _, test_dataset = prepare_math_data()
        dataset = DatasetRegistry.load_dataset("aime2024", split)
        if dataset is None:
            print("Failed to load dataset after preparation.")
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
    ground_truth = example.get("ground_truth", "")
    data_source = example.get("data_source", "math")
    
    task = {
        "question": question,
        "ground_truth": ground_truth,
        "idx": idx,
        "data_source": data_source,
    }
    return task


def remove_prompt_ids(data):
    """Recursively remove prompt_ids from dictionary structure."""
    if isinstance(data, dict):
        # Create a new dict without prompt_ids
        result = {}
        for key, value in data.items():
            if key == "prompt_ids":
                continue  # Skip prompt_ids
            result[key] = remove_prompt_ids(value)  # Recursively process nested structures
        return result
    elif isinstance(data, list):
        # Process each item in the list
        return [remove_prompt_ids(item) for item in data]
    else:
        # Return primitives as-is
        return data


def evaluate_results(results):
    """Evaluate the results and compute metrics."""
    from collections import defaultdict
    
    # Create a map to store correct answers per problem
    problem_correct_map = defaultdict(int)
    problem_total_map = defaultdict(int)
    
    # Extract metrics from episodes
    all_metrics = []
    
    # Track cumulative correct at each turn
    cumulative_correct_by_turn = defaultdict(int)
    total_episodes = 0
    
    # Track lessons learned across turns
    lessons_by_turn = defaultdict(list)
    
    # Count correct answers for each problem
    for episode in results:
        problem = episode.task.get("question", f"idx_{episode.task.get('idx', 'unknown')}")
        
        # Use the episode-level is_correct flag set by the workflow
        is_correct = episode.is_correct
        
        problem_correct_map[problem] += int(is_correct)
        problem_total_map[problem] += 1
        total_episodes += 1
        
        # Collect metrics
        if episode.metrics:
            all_metrics.append(episode.metrics)
            
            # Track lessons learned at each turn
            for turn_key in episode.metrics.keys():
                if turn_key.startswith("lessons_count_t"):
                    turn_num = int(turn_key.split("_t")[1])
                    lessons_by_turn[turn_num].append(episode.metrics[turn_key])
            
            # Track when each problem becomes correct
            for turn_key in ["is_correct_t0", "is_correct_t1", "is_correct_t2", "is_correct_t3", "is_correct_t4"]:
                if turn_key in episode.metrics:
                    turn_num = int(turn_key.split("_t")[1])
                    # If this turn shows correct, count it (and all future turns)
                    if episode.metrics[turn_key] > 0:
                        # Mark as correct from this turn forward
                        for future_turn in range(turn_num, 10):  # up to turn 9
                            cumulative_correct_by_turn[future_turn] += 1
                        break
    
    # Calculate pass@1 (final correctness)
    total_problems = len(problem_correct_map)
    
    if total_problems > 0:
        pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
    else:
        pass_at_1 = 0.0
    
    print("=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total unique problems: {total_problems}")
    print(f"Total episodes: {total_episodes}")
    print(f"Final Pass@1 Accuracy: {pass_at_1:.4f}")
    
    # Print cumulative pass@1 at each turn
    if cumulative_correct_by_turn and total_episodes > 0:
        print("\nCumulative Pass@1 at each turn:")
        max_turn = max(cumulative_correct_by_turn.keys()) if cumulative_correct_by_turn else 0
        for turn in range(0, max_turn + 1):
            correct_count = cumulative_correct_by_turn.get(turn, 0)
            pass_at_k = correct_count / total_episodes
            print(f"  Turn {turn}: {pass_at_k:.4f} ({correct_count}/{total_episodes})")
    
    # Print average lessons learned at each turn
    if lessons_by_turn:
        print("\nAverage lessons learned at each turn:")
        for turn in sorted(lessons_by_turn.keys()):
            avg_lessons = sum(lessons_by_turn[turn]) / len(lessons_by_turn[turn])
            print(f"  Turn {turn}: {avg_lessons:.2f} lessons")
    
    # Print aggregate metrics if available
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                avg_metrics[key] = sum(values) / len(values)
        
        if avg_metrics:
            print("\nAverage metrics across episodes:")
            for key, value in sorted(avg_metrics.items()):
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    print("=" * 60)
    
    return pass_at_1


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Configuration
    n_parallel_tasks = 64
    n_turns = 4  # Number of CM↔Solver interaction turns
    split = "test"  # "train" or "test"

    # Model configuration - both can use the same model or different models
    solver_model_name = "Qwen/Qwen3-4B"
    cm_model_name = "Qwen/Qwen3-4B"  # CM uses trainable engine
    
    tokenizer = AutoTokenizer.from_pretrained(solver_model_name)

    # Create verifier function
    verifier = create_verifier("math")

    # Create solver engine (frozen - not trainable)
    solver_engine = OpenAIEngine(
        model=solver_model_name,
        tokenizer=tokenizer,
        max_prompt_length=10000,
        max_response_length=8192,
        base_url="http://localhost:30000/v1",
        api_key="None",
        sampling_params={"temperature": 0.6, "top_p": 0.95},
    )

    # Create CM engine (trainable - this is what gets trained)
    cm_engine = OpenAIEngine(
        model=cm_model_name,
        tokenizer=tokenizer,
        max_prompt_length=10000,
        max_response_length=8192,
        base_url="http://localhost:30000/v1",
        api_key="None",
        sampling_params={"temperature": 0.6, "top_p": 0.95},
    )

    # Create workflow engine
    # Note: AgentWorkflowEngine passes rollout_engine as a keyword arg to the workflow.
    # The workflow now handles both rollout_engine and cm_engine from workflow_args.
    
    engine = AgentWorkflowEngine(
        workflow_cls=MathCMKnowledgeWorkflow,
        workflow_args={
            "solver_engine": solver_engine,
            "verifier": verifier,
            "n_turns": n_turns,
            "use_tools": True,  # Enable Python tools for math calculations
            "cm_engine": cm_engine,  # Pass cm_engine via workflow_args
        },
        rollout_engine=cm_engine,  # Also passed as rollout_engine, workflow will use cm_engine if provided
        config=None,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=1,
    )

    # Load tasks
    print("Loading tasks...")
    tasks = load_data(n=1, split=split)
    print(f"Loaded {len(tasks)} tasks")
    
    if not tasks:
        print("No tasks loaded. Exiting.")
        exit(1)

    # Execute workflow
    print(f"Executing workflow with {n_turns} turns per episode...")
    results = asyncio.run(engine.execute_tasks(tasks))

    # Evaluate results
    print("\nEvaluating results...")
    pass_at_1 = evaluate_results(results)

    # Save results (with prompt_ids removed)
    os.makedirs("logs", exist_ok=True)
    output_file = f"logs/solver_cm_knowledge_flow_math_{split}_{len(tasks)}.json"
    with open(output_file, "w") as f:
        # Remove prompt_ids from all episodes before saving
        episodes_dict = [remove_prompt_ids(episode.to_dict()) for episode in results]
        json.dump(episodes_dict, f, indent=4)

    print(f"\n✅ Results saved to {output_file}")

