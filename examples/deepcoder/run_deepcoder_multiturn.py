import asyncio
import os
from datetime import datetime

import numpy as np
from transformers import AutoTokenizer

from rllm.agents.code_agent import CompetitionCodingAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.code.competition_coding import CompetitionCodingEnv
from rllm.rewards.reward_fn import code_reward_fn
from rllm.utils import save_trajectories


def extract_response_texts_from_trajectories(trajectories):
    """
    Extract response texts from trajectory results.
    """
    response_texts = []
    is_truncated_flags = []

    for trajectory in trajectories:
        for step in trajectory.steps:
            if hasattr(step, 'model_response') and step.model_response:
                response_texts.append(step.model_response)
                is_truncated_flags.append(getattr(step, 'done', False))
            elif hasattr(step, 'action') and step.action:
                response_texts.append(str(step.action))
                is_truncated_flags.append(getattr(step, 'done', False))

    return response_texts, is_truncated_flags


def calculate_token_counts(response_texts, tokenizer, max_response_length=16384):
    token_counts = []
    is_truncated_flags = []

    for text in response_texts:
        if text:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_count = len(tokens)
            token_counts.append(token_count)
            is_truncated = token_count >= (max_response_length - 50)
            is_truncated_flags.append(is_truncated)

    return token_counts, is_truncated_flags


def print_token_statistics(token_counts, is_truncated_flags=None):
    if not token_counts:
        print("No token counts available.")
        return

    token_counts_array = np.array(token_counts)

    print("\n" + "=" * 60)
    print("📊 RESPONSE LENGTH TOKEN STATISTICS")
    print("=" * 60)
    print(f"Total responses analyzed: {len(token_counts)}")

    if is_truncated_flags is not None:
        truncated_count = sum(is_truncated_flags)
        truncation_percentage = (truncated_count / len(token_counts) * 100) if len(token_counts) > 0 else 0
        print(f"Truncated responses: {truncated_count}/{len(token_counts)} ({truncation_percentage:.1f}%)")

    print(f"Min tokens: {np.min(token_counts_array)}")
    print(f"Max tokens: {np.max(token_counts_array)}")
    print(f"Mean tokens: {np.mean(token_counts_array):.2f}")
    print(f"Median tokens: {np.median(token_counts_array):.2f}")
    print(f"Standard deviation: {np.std(token_counts_array):.2f}")

    q25 = np.percentile(token_counts_array, 25)
    q50 = np.percentile(token_counts_array, 50)
    q75 = np.percentile(token_counts_array, 75)

    print(f"\nQuartile distribution:")
    print(f"  25th percentile (Q1): {q25:.2f}")
    print(f"  50th percentile (Q2/Median): {q50:.2f}")
    print(f"  75th percentile (Q3): {q75:.2f}")
    print(f"  Interquartile range (IQR): {q75 - q25:.2f}")

    print(f"\nAdditional percentiles:")
    for p in [10, 90, 95, 99]:
        value = np.percentile(token_counts_array, p)
        print(f"  {p}th percentile: {value:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    K = 1
    n_parallel_agents = 64
    model_name = "agentica-org/DeepCoder-1.5B-Preview"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    reward_fn = code_reward_fn
    env_args = {
        "reward_fn": reward_fn,
        "max_turns": 4,
    }

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    engine = AgentExecutionEngine(
        agent_class=CompetitionCodingAgent,
        env_class=CompetitionCodingEnv,
        agent_args={},
        env_args=env_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=16384,
        max_prompt_length=4096,
        n_parallel_agents=n_parallel_agents,
        max_steps=4,
    )

    dataset = DatasetRegistry.load_dataset("lcb", "test")
    tasks = dataset.get_data()
    tasks = dataset.repeat(n=K)

    results = asyncio.run(engine.execute_tasks(tasks))

    # Analyze token counts for response lengths
    print("\n🔍 Analyzing response token counts...")
    response_texts, step_truncated_flags = extract_response_texts_from_trajectories(results)
    token_counts, token_truncated_flags = calculate_token_counts(response_texts, tokenizer, max_response_length=16384)
    print_token_statistics(token_counts, token_truncated_flags)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_trajectories(results, filename=f"deepcoder_multiturn_pass@{K}_trajectories_{len(tasks)}_{timestamp}.pt")
