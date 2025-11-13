import asyncio
import json
import os
import sys
from copy import deepcopy
from statistics import mean

from transformers import AutoTokenizer

from solver_cm_flow import SolverContextManagerWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.engine import OpenAIEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.rewards.countdown_reward import countdown_reward_fn

# Make sure countdown helpers are importable if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "countdown"))


def load_data(n=1):
    """Load countdown data using the Dataset interface."""
    dataset = DatasetRegistry.load_dataset("countdown", "test")
    if dataset is None:
        print("Dataset not found, preparing dataset...")
        from prepare_countdown_data import prepare_countdown_data

        _, dataset, _, _ = prepare_countdown_data()

    data = []
    for idx, example in enumerate(dataset):
        processed = process_countdown_fn(example, idx)
        for _ in range(n):
            data.append(deepcopy(processed))
    return data


def process_countdown_fn(example, idx):
    """Process countdown example into the expected format."""
    question = example["question"]
    target = example["target"]
    nums = example["nums"]

    # Ground truth format expected by countdown_reward_fn
    ground_truth = {"target": target, "numbers": nums}

    task = {
        "question": question,
        "ground_truth": ground_truth,
        "idx": idx,
        "data_source": "countdown",
        "target": target,
        "nums": nums,
    }
    return task


def evaluate_results(results):
    """Evaluate pass@k and summarize solver/context-manager metrics."""
    from collections import defaultdict

    problem_correct_map = defaultdict(int)
    problem_total_map = defaultdict(int)

    solver_accs = []
    cm_accs = []
    improvements = []
    best_rewards = []

    for episode in results:
        problem = episode.task["question"]
        is_correct = episode.is_correct
        problem_correct_map[problem] += int(is_correct)
        problem_total_map[problem] += 1

        # Aggregate metrics if present
        m = getattr(episode, "metrics", {}) or {}
        if "solver_acc" in m:
            solver_accs.append(m["solver_acc"])
        if "cm_acc" in m:
            cm_accs.append(m["cm_acc"])
        if "avg_improvement" in m:
            improvements.append(m["avg_improvement"])
        if "best_reward" in m:
            best_rewards.append(m["best_reward"])

    # pass@1 and pass@k
    k = max(problem_total_map.values()) if problem_total_map else 1
    total_problems = len(problem_correct_map)

    if total_problems > 0:
        pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
        pass_at_k = sum(1 for _, correct in problem_correct_map.items() if correct > 0) / total_problems
    else:
        pass_at_1 = 0.0
        pass_at_k = 0.0

    print("Total unique problems:", total_problems)
    print("Average Pass@1 Accuracy:", pass_at_1)
    print(f"Average Pass@{k} Accuracy:", pass_at_k)

    # Print solver/cm summaries
    if solver_accs:
        print("Avg solver_acc (mean reward over all solver attempts):", mean(solver_accs))
    if cm_accs:
        print("Avg cm_acc (post-CM correctness rate):", mean(cm_accs))
    if improvements:
        print("Avg improvement (last_round_avg - first_round_avg):", mean(improvements))
    if best_rewards:
        print("Avg best_reward:", mean(best_rewards))


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Configuration
    n_parallel_tasks = 128
    n_rounds = 4        # Number of CM rounds (>=1). If 1, no CM feedback is used.
    # n_solutions = 2     # Solutions per round
    n_solutions = 1

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    rollout_engine = OpenAIEngine(
        model=model_name,
        tokenizer=tokenizer,
        max_prompt_length=2048,
        max_response_length=1024,
        base_url="http://localhost:30000/v1",
        api_key="None",
        sampling_params={"temperature": 0.6, "top_p": 0.95},
    )

    engine = AgentWorkflowEngine(
        workflow_cls=SolverContextManagerWorkflow,
        workflow_args={
            "n_rounds": n_rounds,
            "n_solutions": n_solutions,
            "reward_function": countdown_reward_fn,
        },
        rollout_engine=rollout_engine,
        config=None,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=1,
    )

    # Load countdown tasks
    tasks = load_data(n=1)
    print(f"Loaded {len(tasks)} countdown tasks")

    results = asyncio.run(engine.execute_tasks(tasks))

    # Evaluate results
    print("Evaluating results...")
    evaluate_results(results)

    # Save results
    os.makedirs("logs", exist_ok=True)
    out_path = "logs/solver_cm_countdown.json"
    with open(out_path, "w") as f:
        json.dump([episode.to_dict() for episode in results], f, indent=4)

    print(f"\nResults saved to {out_path}")
