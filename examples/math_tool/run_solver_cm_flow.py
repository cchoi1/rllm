import asyncio
import json
import os
from copy import deepcopy
from typing import Dict

from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine import OpenAIEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.rewards.math_reward import rllm_reward_fn_math
from examples.math_tool.solver_cm_flow import MathCMWorkflow, MathResult, SumOfProblemsWorkflow


def create_verifier(task_data_source: str = "math"):
    """Create a verifier function that wraps rllm_reward_fn_math and returns MathResult."""
    from rllm.rewards.math_utils.utils import extract_answer
    from rllm.globals import THOUGHT_DELIMITER_END
    
    def verifier(task: Dict, solution: str) -> MathResult:
        # Use rllm_reward_fn_math to evaluate
        data_source = task.get("data_source", task_data_source)
        ground_truth = task.get("ground_truth")
        
        reward_output = rllm_reward_fn_math(data_source, solution, ground_truth)
        
        # Extract model_answer using the same logic as RewardMathFn
        model_response = solution
        if THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            model_solution = model_response
        
        model_answer = extract_answer(model_solution)
        
        # Build metadata with extracted answers
        metadata = reward_output.metadata.copy() if reward_output.metadata else {}
        metadata["model_answer"] = model_answer
        metadata["ground_truth"] = ground_truth
        
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
            metadata=metadata,
        )
    
    return verifier


def load_data(n=1, split="test"):
    """Load math data using the Dataset interface."""
    dataset = DatasetRegistry.load_dataset("aime2024", split)
    if dataset is None:
        raise ValueError(f"Dataset 'aime2024' split '{split}' not found in registry.")
    
    data = []
    for idx, example in enumerate(dataset):
        processed = process_task(example, idx)
        for i in range(n):
            data.append(deepcopy(processed))
    return data


def load_sum_data(num_episodes=1, n_problems=4, split="test"):
    """Build tasks for the SumOfProblemsWorkflow: each task has N problems."""
    dataset = DatasetRegistry.load_dataset("aime2024", split)
    if dataset is None:
        raise ValueError(f"Dataset 'aime2024' split '{split}' not found in registry.")

    # Flatten candidate problems
    examples = list(dataset)
    if not examples:
        return []

    tasks = []
    cursor = 0
    for ep in range(num_episodes):
        problems = []
        ground_truths = []
        for _ in range(n_problems):
            ex = examples[cursor % len(examples)]
            problems.append(ex.get("question", ""))
            ground_truths.append(ex.get("ground_truth", ""))
            cursor += 1
        tasks.append(
            {
                "problems": problems,
                "ground_truths": ground_truths,
                "idx": ep,
                "data_source": "math",
            }
        )
    return tasks


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


def remove_token_ids(data):
    """Recursively remove prompt_ids and completion_ids from dictionaries."""
    if isinstance(data, dict):
        # Create a new dict without prompt_ids and completion_ids
        result = {}
        for key, value in data.items():
            if key not in ["prompt_ids", "completion_ids"]:
                result[key] = remove_token_ids(value)
        return result
    elif isinstance(data, list):
        return [remove_token_ids(item) for item in data]
    else:
        return data


def remove_model_output(data):
    """Recursively remove model_output from dictionaries."""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key != "model_output":
                result[key] = remove_model_output(value)
        return result
    elif isinstance(data, list):
        return [remove_model_output(item) for item in data]
    else:
        return data


def evaluate_sum_workflow_results(results):
    """Evaluate results from SumOfProblemsWorkflow.
    
    Reports pred_final_sum, true_final_sum, and pass@1 for each episode,
    and prints average pass@1 across all episodes.
    
    Args:
        results: List of Episode objects from SumOfProblemsWorkflow
        
    Returns:
        float: Average pass@1 across all episodes
    """
    if not results:
        print("No results to evaluate.")
        return 0.0
    
    print("=" * 60)
    print("📊 EVALUATION RESULTS (SumOfProblemsWorkflow)")
    print("=" * 60)
    
    total_episodes = len(results)
    pass_at_1_scores = []
    
    print(f"\nTotal episodes: {total_episodes}\n")
    print("Per-episode results:")
    print("-" * 60)
    
    for idx, episode in enumerate(results):
        metrics = episode.metrics or {}
        pred_sum = metrics.get("pred_final_sum", 0.0)
        true_sum = metrics.get("true_final_sum", 0.0)
        pass_at_1 = metrics.get("pass@1", 0.0)
        n_turns = metrics.get("n_turns", 0.0)
        
        pass_at_1_scores.append(pass_at_1)
        
        print(f"Episode {idx + 1}:")
        print(f"  Predicted final sum: {pred_sum:.6f}")
        print(f"  True final sum:      {true_sum:.6f}")
        print(f"  Pass@1:              {pass_at_1:.4f} ({'✓' if pass_at_1 > 0.5 else '✗'})")
        print(f"  Number of turns:     {int(n_turns)}")
        
        # Extract and display solver answers and ground truths from verifier metadata
        task = episode.task
        problems = task.get("problems", [])
        ground_truths = task.get("ground_truths", [])
        
        print(f"  Solver answers vs Ground truths:")
        problem_idx = 0
        for trajectory in episode.trajectories:
            if trajectory.name == "solver" and trajectory.steps and problem_idx < len(ground_truths):
                step = trajectory.steps[0]
                verifier_metadata = step.info.get("verifier_metadata", {}) if step.info else {}
                model_answer = verifier_metadata.get("model_answer", "N/A")
                ground_truth = verifier_metadata.get("ground_truth", ground_truths[problem_idx] if problem_idx < len(ground_truths) else "N/A")
                
                # Format display
                model_answer_str = str(model_answer) if model_answer is not None else "N/A"
                ground_truth_str = str(ground_truth) if ground_truth is not None else "N/A"
                
                print(f"    Problem {problem_idx + 1}: Solver={model_answer_str}, Ground truth={ground_truth_str}")
                problem_idx += 1
        
        print()
    
    # Calculate average pass@1
    avg_pass_at_1 = sum(pass_at_1_scores) / len(pass_at_1_scores) if pass_at_1_scores else 0.0
    
    print("-" * 60)
    print(f"Average Pass@1 across all episodes: {avg_pass_at_1:.4f}")
    print("=" * 60)
    
    return avg_pass_at_1


def evaluate_math_cm_workflow_results(results):
    """Evaluate results from MathCMWorkflow.
    
    Tracks per-problem correctness across episodes and reports cumulative
    pass@1 at each turn.
    
    Args:
        results: List of Episode objects from MathCMWorkflow
        
    Returns:
        float: Final Pass@1 accuracy
    """
    if not results:
        print("No results to evaluate.")
        return 0.0
    
    from collections import defaultdict
    
    # Create a map to store correct answers per problem
    problem_correct_map = defaultdict(int)
    problem_total_map = defaultdict(int)
    
    # Track when each problem first becomes correct (per problem, not per episode)
    # Key: problem identifier, Value: first turn where it became correct (None if never correct)
    problem_first_correct_turn = {}
    
    # Extract metrics from episodes
    all_metrics = []
    total_episodes = 0
    
    # Process each episode to track per-problem correctness
    for episode in results:
        problem = episode.task.get("question", f"idx_{episode.task.get('idx', 'unknown')}")
        
        # Use the episode-level is_correct flag set by the workflow
        is_correct = episode.is_correct
        
        problem_correct_map[problem] += int(is_correct)
        problem_total_map[problem] += 1
        total_episodes += 1
        
        # Initialize problem tracking if not already present
        if problem not in problem_first_correct_turn:
            problem_first_correct_turn[problem] = None
        
        # Collect metrics
        if episode.metrics:
            all_metrics.append(episode.metrics)
            
            # Find the first turn where this problem became correct in this episode
            # Sort turn keys to process in order (t0, t1, t2, ...)
            turn_keys = sorted([k for k in episode.metrics.keys() if k.startswith("is_correct_t")],
                              key=lambda x: int(x.split("_t")[1]))
            
            episode_first_correct_turn = None
            for turn_key in turn_keys:
                turn_num = int(turn_key.split("_t")[1])
                if episode.metrics[turn_key] > 0:
                    # This turn is correct
                    episode_first_correct_turn = turn_num
                    break  # Once we find it correct, we know it stays correct (due to workflow logic)
            
            # Track the earliest turn across all episodes for this problem
            if episode_first_correct_turn is not None:
                # Keep the earliest turn where this problem becomes correct
                if problem_first_correct_turn[problem] is None or episode_first_correct_turn < problem_first_correct_turn[problem]:
                    problem_first_correct_turn[problem] = episode_first_correct_turn
    
    # Calculate pass@1 (final correctness)
    total_problems = len(problem_correct_map)
    
    if total_problems > 0:
        pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
    else:
        pass_at_1 = 0.0
    
    print("=" * 60)
    print("📊 EVALUATION RESULTS (MathCMWorkflow)")
    print("=" * 60)
    print(f"Total unique problems: {total_problems}")
    print(f"Total episodes: {total_episodes}")
    print(f"Final Pass@1 Accuracy: {pass_at_1:.4f}")
    
    # Calculate cumulative pass@1 at each turn
    # For each turn, count how many unique problems are correct by that turn
    if total_problems > 0:
        print("\n📈 Cumulative Pass@1 at each turn (per problem):")
        # Find max turn number from metrics
        max_turn = 0
        for episode in results:
            if episode.metrics:
                turn_keys = [k for k in episode.metrics.keys() if k.startswith("is_correct_t")]
                if turn_keys:
                    max_turn = max(max_turn, max(int(k.split("_t")[1]) for k in turn_keys))
        
        # Calculate cumulative pass@1 for each turn
        for turn in range(0, max_turn + 1):
            # Count how many problems are correct by this turn (or earlier)
            problems_correct_by_turn = 0
            for problem, first_correct_turn in problem_first_correct_turn.items():
                if first_correct_turn is not None and first_correct_turn <= turn:
                    problems_correct_by_turn += 1
            
            cumulative_pass_at_1 = problems_correct_by_turn / total_problems
            print(f"  Turn {turn}: {cumulative_pass_at_1:.4f} ({problems_correct_by_turn}/{total_problems} problems)")
    
    # Print aggregate metrics if available
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                avg_metrics[key] = sum(values) / len(values)
        
        if avg_metrics:
            print("\nAverage metrics across episodes:")
            for key, value in avg_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    # Print per-episode solver answers and ground truths
    print("\n📋 Per-episode Solver Answers vs Ground Truths:")
    print("-" * 60)
    for idx, episode in enumerate(results):
        problem = episode.task.get("question", f"idx_{episode.task.get('idx', 'unknown')}")
        ground_truth = episode.task.get("ground_truth", "N/A")
        is_correct = episode.is_correct
        
        print(f"\nEpisode {idx + 1} ({'✓' if is_correct else '✗'}):")
        print(f"  Problem: {problem[:100]}..." if len(problem) > 100 else f"  Problem: {problem}")
        
        # Find initial and final solver attempts
        solver_trajectories = [t for t in episode.trajectories if t.name == "solver"]
        if solver_trajectories:
            # Initial attempt
            initial_traj = solver_trajectories[0]
            if initial_traj.steps:
                initial_step = initial_traj.steps[0]
                initial_verifier_metadata = initial_step.info.get("verifier_metadata", {}) if initial_step.info else {}
                initial_model_answer = initial_verifier_metadata.get("model_answer", "N/A")
                initial_ground_truth = initial_verifier_metadata.get("ground_truth", ground_truth)
                
                initial_model_answer_str = str(initial_model_answer) if initial_model_answer is not None else "N/A"
                initial_ground_truth_str = str(initial_ground_truth) if initial_ground_truth is not None else "N/A"
                
                print(f"  Initial attempt: Solver={initial_model_answer_str}, Ground truth={initial_ground_truth_str}")
            
            # Final attempt (last solver trajectory)
            if len(solver_trajectories) > 1:
                final_traj = solver_trajectories[-1]
                if final_traj.steps:
                    final_step = final_traj.steps[-1]
                    final_verifier_metadata = final_step.info.get("verifier_metadata", {}) if final_step.info else {}
                    final_model_answer = final_verifier_metadata.get("model_answer", "N/A")
                    final_ground_truth = final_verifier_metadata.get("ground_truth", ground_truth)
                    
                    final_model_answer_str = str(final_model_answer) if final_model_answer is not None else "N/A"
                    final_ground_truth_str = str(final_ground_truth) if final_ground_truth is not None else "N/A"
                    
                    print(f"  Final attempt:   Solver={final_model_answer_str}, Ground truth={final_ground_truth_str}")
    
    print("\n" + "=" * 60)
    
    return pass_at_1


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Configuration
    n_parallel_tasks = 64
    n_turns = 4  # Number of problems to sum (N turns)
    split = "test"  # "train" or "test"

    # Model configuration - both can use the same model or different models
    solver_model_name = "Qwen/Qwen3-4B"
    cm_model_name = "Qwen/Qwen3-4B"
    # solver_model_name = "o4-mini"
    # cm_model_name = "o4-mini"
    
    tokenizer = AutoTokenizer.from_pretrained(solver_model_name)

    # Create verifier function
    verifier = create_verifier("math")

    # Create CM engine (trainable - used by SumContextManager)
    cm_engine = OpenAIEngine(
        model=cm_model_name,
        tokenizer=tokenizer,
        max_prompt_length=10000,
        max_response_length=8192,
        base_url="http://localhost:30000/v1",
        api_key="None",
        sampling_params={"temperature": 0.6, "top_p": 0.95},
    )

    # Create solver engine (frozen - used by MathSolver)
    solver_engine = OpenAIEngine(
        model=solver_model_name,
        tokenizer=tokenizer,
        max_prompt_length=10000,
        max_response_length=8192,
        base_url="http://localhost:30000/v1",
        api_key="None",
        sampling_params={"temperature": 0.6, "top_p": 0.95},
    )

    # cm_engine = OpenAIEngine(
    #     model="o4-mini",
    #     base_url="https://api.openai.com/v1",
    #     api_key=os.environ["OPENAI_API_KEY"],
    #     sampling_params={
    #         "max_completion_tokens": 8192,
    #     },
    # )

    # solver_engine = OpenAIEngine(
    #     model="o4-mini",
    #     base_url="https://api.openai.com/v1",
    #     api_key=os.environ["OPENAI_API_KEY"],
    #     sampling_params={
    #         "max_completion_tokens": 8192,
    #     },
    # )

    # Select workflow
    # Set workflow_class to either SumOfProblemsWorkflow or MathCMWorkflow
    workflow_class = SumOfProblemsWorkflow
    # workflow_class = MathCMWorkflow  # Uncomment to use MathCMWorkflow instead
    
    # Create workflow engine
    # Note: AgentWorkflowEngine passes rollout_engine as a keyword arg to the workflow.
    # The workflow uses rollout_engine for CM and solver_engine (from workflow_args) for Solver.
    
    if workflow_class == SumOfProblemsWorkflow:
        engine = AgentWorkflowEngine(
            workflow_cls=SumOfProblemsWorkflow,
            workflow_args={
                "solver_engine": solver_engine,
                "verifier": verifier,
                "n_turns": n_turns,
                "use_tools": True,  # Enable Python tools for math calculations
            },
            rollout_engine=cm_engine,  # CM uses this trainable engine
            config=None,
            n_parallel_tasks=n_parallel_tasks,
            retry_limit=1,
        )
        
        # Load tasks for SumOfProblemsWorkflow
        print("Loading tasks...")
        tasks = load_sum_data(num_episodes=100, n_problems=n_turns, split=split)
        tasks = tasks[:100]
        print(f"Loaded {len(tasks)} tasks")
    else:
        # MathCMWorkflow
        engine = AgentWorkflowEngine(
            workflow_cls=MathCMWorkflow,
            workflow_args={
                "solver_engine": solver_engine,
                "verifier": verifier,
                "n_turns": n_turns,
                "use_tools": True,  # Enable Python tools for math calculations
            },
            rollout_engine=cm_engine,  # CM uses this trainable engine
            config=None,
            n_parallel_tasks=n_parallel_tasks,
            retry_limit=1,
        )
        
        # Load tasks for MathCMWorkflow
        print("Loading tasks...")
        tasks = load_data(n=1, split=split)
        print(f"Loaded {len(tasks)} tasks")
    
    if not tasks:
        print("No tasks loaded. Exiting.")
        exit(1)

    # Execute workflow
    workflow_name = "SumOfProblemsWorkflow" if workflow_class == SumOfProblemsWorkflow else "MathCMWorkflow"
    print(f"Executing {workflow_name} with {n_turns} turns per episode...")
    results = asyncio.run(engine.execute_tasks(tasks))

    # Evaluate results using the appropriate evaluation function
    print("\nEvaluating results...")
    if workflow_class == SumOfProblemsWorkflow:
        pass_at_1 = evaluate_sum_workflow_results(results)
    else:
        pass_at_1 = evaluate_math_cm_workflow_results(results)

    # Save results
    os.makedirs("logs", exist_ok=True)
    output_file = f"logs/solver_cm_flow_math_{split}_{len(tasks)}.json"
    # Remove prompt_ids and completion_ids before saving
    serialized_results = [remove_token_ids(episode.to_dict()) for episode in results]
    
    # For SumOfProblemsWorkflow, also remove model_output
    if workflow_class == SumOfProblemsWorkflow:
        serialized_results = [remove_model_output(result) for result in serialized_results]
    
    with open(output_file, "w") as f:
        json.dump(serialized_results, f, indent=4)

    print(f"\n✅ Results saved to {output_file}")

