#!/usr/bin/env python3
"""
Benchmark script to measure tokens/second for OpenAIEngine and VerlEngine
with the agentica deepcoder model.

Usage:
    # Test OpenAIEngine with DeepCoder training dataset (requires OpenAI-compatible API server like vLLM/SGLang)
    python scripts/benchmark/deepcoder_tokens_per_second.py \
        --engine openai \
        --model agentica-org/DeepCoder-1.5B-Preview \
        --base-url http://localhost:12345/v1 \
        --dataset-split train \
        --num-requests 50 \
        --num-parallel 4

    # Test with test dataset
    python scripts/benchmark/deepcoder_tokens_per_second.py \
        --engine openai \
        --model agentica-org/DeepCoder-1.5B-Preview \
        --base-url http://localhost:12345/v1 \
        --dataset-split test \
        --num-requests 50 \
        --num-parallel 4

    # Test VerlEngine (requires proper veRL config setup)
    python scripts/benchmark/deepcoder_tokens_per_second.py \
        --engine verl \
        --config-path path/to/config.yaml \
        --dataset-split train \
        --num-requests 10
"""

import argparse
import asyncio
import time
import os
from typing import List, Dict, Any, Optional

import numpy as np
from transformers import AutoTokenizer

from rllm.engine.rollout.openai_engine import OpenAIEngine
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.data.dataset import DatasetRegistry


def load_deepcoder_prompts(
    dataset_split: str = "train",
    max_prompts: Optional[int] = None,
) -> List[str]:
    """Load prompts from DeepCoder dataset.
    
    Args:
        dataset_split: Dataset split to use ("train" or "test")
        max_prompts: Maximum number of prompts to load (None for all)
        
    Returns:
        List of prompt strings
    """
    print(f"Loading DeepCoder {dataset_split} dataset...")
    try:
        dataset = DatasetRegistry.load_dataset("lcb", dataset_split)
        if dataset is None:
            print(f"Dataset 'lcb' split '{dataset_split}' not found in registry.")
            print("Available datasets:", DatasetRegistry.list_datasets())
            print("Attempting to prepare dataset...")
            # Try to prepare the dataset
            try:
                import sys
                import importlib.util
                # Try to import the preparation script
                prep_script_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "examples", "deepcoder", "prepare_deepcoder_data_lcb_only.py"
                )
                if os.path.exists(prep_script_path):
                    spec = importlib.util.spec_from_file_location("prepare_deepcoder_data_lcb_only", prep_script_path)
                    prep_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(prep_module)
                    prep_module.prepare_lcb_data()
                else:
                    # Fallback to direct import
                    from examples.deepcoder.prepare_deepcoder_data_lcb_only import prepare_lcb_data
                    prepare_lcb_data()
            except ImportError as ie:
                print(f"Could not import dataset preparation script: {ie}")
                print("Please run: python examples/deepcoder/prepare_deepcoder_data_lcb_only.py")
                raise ValueError(f"Dataset 'lcb' split '{dataset_split}' not found and could not prepare it")
            
            dataset = DatasetRegistry.load_dataset("lcb", dataset_split)
            if dataset is None:
                raise ValueError(f"Failed to load dataset 'lcb' split '{dataset_split}' after preparation")
        
        data = dataset.get_data()
        print(f"Loaded {len(data)} examples from dataset")
        
        # Extract prompts from the "question" field
        prompts = []
        for example in data:
            if "question" in example:
                prompts.append(example["question"])
            elif "problem" in example:
                # Fallback to "problem" field if "question" is not available
                prompts.append(example["problem"])
            else:
                # Skip examples without a prompt field
                continue
        
        if max_prompts is not None and len(prompts) > max_prompts:
            prompts = prompts[:max_prompts]
            print(f"Limited to {max_prompts} prompts")
        
        print(f"Extracted {len(prompts)} prompts from dataset")
        return prompts
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to sample prompts...")
        # Fallback to sample prompts if dataset loading fails
        return [
            "Write a Python function to calculate the factorial of a number.",
            "Implement a binary search algorithm in Python.",
            "Create a function that finds the longest common subsequence between two strings.",
            "Write a Python function to sort a list of dictionaries by a specific key.",
            "Implement a function to check if a number is prime.",
            "Write a Python function to reverse a linked list.",
            "Create a function that finds all permutations of a string.",
            "Implement a function to calculate the Fibonacci sequence up to n terms.",
            "Write a Python function to merge two sorted arrays.",
            "Create a function that checks if two strings are anagrams.",
        ]


async def benchmark_openai_engine(
    engine: OpenAIEngine,
    prompts: List[str],
    num_requests: int,
    num_parallel: int,
    sampling_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Benchmark OpenAIEngine and return metrics."""
    print(f"\n{'='*60}")
    print(f"Benchmarking OpenAIEngine")
    print(f"{'='*60}")
    print(f"Number of requests: {num_requests}")
    print(f"Number of parallel requests: {num_parallel}")
    print(f"Number of unique prompts: {len(prompts)}")
    print(f"Sampling params: {sampling_params}")
    
    # Prepare messages for each prompt
    messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
    # Repeat to reach num_requests
    while len(messages_list) < num_requests:
        messages_list.extend(messages_list)
    messages_list = messages_list[:num_requests]
    
    results = []
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_time = 0.0
    request_times = []
    token_counts = []
    
    semaphore = asyncio.Semaphore(num_parallel)
    
    async def single_request(idx: int, messages: List[Dict[str, str]]):
        nonlocal total_tokens, total_prompt_tokens, total_completion_tokens, total_time, request_times, token_counts, results
        async with semaphore:
            application_id = f"benchmark_{idx}"
            start_time = time.time()
            try:
                output: ModelOutput = await engine.get_model_response(
                    messages, 
                    application_id=application_id,
                    **sampling_params
                )
                elapsed = time.time() - start_time
                
                prompt_tokens = output.prompt_length
                completion_tokens = output.completion_length
                tokens = prompt_tokens + completion_tokens
                
                total_tokens += tokens
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_time += elapsed
                request_times.append(elapsed)
                token_counts.append(completion_tokens)
                
                results.append({
                    "idx": idx,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": tokens,
                    "time": elapsed,
                    "tokens_per_second": completion_tokens / elapsed if elapsed > 0 else 0,
                    "success": True,
                })
                
                if (idx + 1) % 10 == 0:
                    print(f"Completed {idx + 1}/{num_requests} requests...")
            except Exception as e:
                elapsed = time.time() - start_time
                results.append({
                    "idx": idx,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "time": elapsed,
                    "tokens_per_second": 0,
                    "success": False,
                    "error": str(e),
                })
                print(f"Error in request {idx}: {e}")
    
    # Run all requests
    start_time = time.time()
    tasks = [single_request(i, msg) for i, msg in enumerate(messages_list)]
    await asyncio.gather(*tasks)
    end_time = time.time()
    
    total_wall_time = end_time - start_time
    successful_requests = [r for r in results if r["success"]]
    failed_requests = [r for r in results if not r.get("success", True)]
    
    # Calculate metrics
    if successful_requests:
        avg_tokens_per_second = np.mean([r["tokens_per_second"] for r in successful_requests])
        median_tokens_per_second = np.median([r["tokens_per_second"] for r in successful_requests])
        p95_tokens_per_second = np.percentile([r["tokens_per_second"] for r in successful_requests], 95)
        p99_tokens_per_second = np.percentile([r["tokens_per_second"] for r in successful_requests], 99)
        
        avg_request_time = np.mean(request_times)
        median_request_time = np.median(request_times)
        p95_request_time = np.percentile(request_times, 95)
        p99_request_time = np.percentile(request_times, 99)
    else:
        avg_tokens_per_second = 0
        median_tokens_per_second = 0
        p95_tokens_per_second = 0
        p99_tokens_per_second = 0
        avg_request_time = 0
        median_request_time = 0
        p95_request_time = 0
        p99_request_time = 0
    
    # Overall throughput (total completion tokens / total wall time)
    overall_throughput = total_completion_tokens / total_wall_time if total_wall_time > 0 else 0
    
    metrics = {
        "engine": "openai",
        "num_requests": num_requests,
        "num_successful": len(successful_requests),
        "num_failed": len(failed_requests),
        "total_wall_time": total_wall_time,
        "total_completion_tokens": total_completion_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_tokens": total_tokens,
        "overall_throughput_tokens_per_second": overall_throughput,
        "avg_tokens_per_second": avg_tokens_per_second,
        "median_tokens_per_second": median_tokens_per_second,
        "p95_tokens_per_second": p95_tokens_per_second,
        "p99_tokens_per_second": p99_tokens_per_second,
        "avg_request_time": avg_request_time,
        "median_request_time": median_request_time,
        "p95_request_time": p95_request_time,
        "p99_request_time": p99_request_time,
        "avg_completion_tokens": np.mean(token_counts) if token_counts else 0,
        "median_completion_tokens": np.median(token_counts) if token_counts else 0,
        "min_completion_tokens": np.min(token_counts) if token_counts else 0,
        "max_completion_tokens": np.max(token_counts) if token_counts else 0,
    }
    
    return metrics


async def benchmark_verl_engine(
    engine,
    prompts: List[str],
    num_requests: int,
    num_parallel: int,
    sampling_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Benchmark VerlEngine and return metrics."""
    print(f"\n{'='*60}")
    print(f"Benchmarking VerlEngine")
    print(f"{'='*60}")
    print(f"Number of requests: {num_requests}")
    print(f"Number of parallel requests: {num_parallel}")
    print(f"Number of unique prompts: {len(prompts)}")
    print(f"Sampling params: {sampling_params}")
    
    # Prepare messages for each prompt
    messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
    # Repeat to reach num_requests
    while len(messages_list) < num_requests:
        messages_list.extend(messages_list)
    messages_list = messages_list[:num_requests]
    
    results = []
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_time = 0.0
    request_times = []
    token_counts = []
    
    semaphore = asyncio.Semaphore(num_parallel)
    
    async def single_request(idx: int, messages: List[Dict[str, str]]):
        nonlocal total_tokens, total_prompt_tokens, total_completion_tokens, total_time, request_times, token_counts, results
        async with semaphore:
            application_id = f"benchmark_{idx}"
            start_time = time.time()
            try:
                output: ModelOutput = await engine.get_model_response(
                    messages,
                    application_id=application_id,
                    validate=False,
                    **sampling_params
                )
                elapsed = time.time() - start_time
                
                prompt_tokens = output.prompt_length
                completion_tokens = output.completion_length
                tokens = prompt_tokens + completion_tokens
                
                total_tokens += tokens
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_time += elapsed
                request_times.append(elapsed)
                token_counts.append(completion_tokens)
                
                results.append({
                    "idx": idx,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": tokens,
                    "time": elapsed,
                    "tokens_per_second": completion_tokens / elapsed if elapsed > 0 else 0,
                    "success": True,
                })
                
                if (idx + 1) % 10 == 0:
                    print(f"Completed {idx + 1}/{num_requests} requests...")
            except Exception as e:
                elapsed = time.time() - start_time
                results.append({
                    "idx": idx,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "time": elapsed,
                    "tokens_per_second": 0,
                    "success": False,
                    "error": str(e),
                })
                print(f"Error in request {idx}: {e}")
    
    # Run all requests
    start_time = time.time()
    tasks = [single_request(i, msg) for i, msg in enumerate(messages_list)]
    await asyncio.gather(*tasks)
    end_time = time.time()
    
    total_wall_time = end_time - start_time
    successful_requests = [r for r in results if r["success"]]
    failed_requests = [r for r in results if not r.get("success", True)]
    
    # Calculate metrics
    if successful_requests:
        avg_tokens_per_second = np.mean([r["tokens_per_second"] for r in successful_requests])
        median_tokens_per_second = np.median([r["tokens_per_second"] for r in successful_requests])
        p95_tokens_per_second = np.percentile([r["tokens_per_second"] for r in successful_requests], 95)
        p99_tokens_per_second = np.percentile([r["tokens_per_second"] for r in successful_requests], 99)
        
        avg_request_time = np.mean(request_times)
        median_request_time = np.median(request_times)
        p95_request_time = np.percentile(request_times, 95)
        p99_request_time = np.percentile(request_times, 99)
    else:
        avg_tokens_per_second = 0
        median_tokens_per_second = 0
        p95_tokens_per_second = 0
        p99_tokens_per_second = 0
        avg_request_time = 0
        median_request_time = 0
        p95_request_time = 0
        p99_request_time = 0
    
    # Overall throughput (total completion tokens / total wall time)
    overall_throughput = total_completion_tokens / total_wall_time if total_wall_time > 0 else 0
    
    metrics = {
        "engine": "verl",
        "num_requests": num_requests,
        "num_successful": len(successful_requests),
        "num_failed": len(failed_requests),
        "total_wall_time": total_wall_time,
        "total_completion_tokens": total_completion_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_tokens": total_tokens,
        "overall_throughput_tokens_per_second": overall_throughput,
        "avg_tokens_per_second": avg_tokens_per_second,
        "median_tokens_per_second": median_tokens_per_second,
        "p95_tokens_per_second": p95_tokens_per_second,
        "p99_tokens_per_second": p99_tokens_per_second,
        "avg_request_time": avg_request_time,
        "median_request_time": median_request_time,
        "p95_request_time": p95_request_time,
        "p99_request_time": p99_request_time,
        "avg_completion_tokens": np.mean(token_counts) if token_counts else 0,
        "median_completion_tokens": np.median(token_counts) if token_counts else 0,
        "min_completion_tokens": np.min(token_counts) if token_counts else 0,
        "max_completion_tokens": np.max(token_counts) if token_counts else 0,
    }
    
    return metrics


def print_metrics(metrics: Dict[str, Any]):
    """Print benchmarking metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Benchmark Results: {metrics['engine'].upper()}Engine")
    print(f"{'='*60}")
    print(f"Requests: {metrics['num_successful']}/{metrics['num_requests']} successful, {metrics['num_failed']} failed")
    print(f"Total wall time: {metrics['total_wall_time']:.2f}s")
    print(f"Total completion tokens: {metrics['total_completion_tokens']:,}")
    print(f"Total prompt tokens: {metrics['total_prompt_tokens']:,}")
    print(f"Total tokens: {metrics['total_tokens']:,}")
    print(f"\n{'─'*60}")
    print(f"Throughput (Tokens/Second):")
    print(f"  Overall throughput: {metrics['overall_throughput_tokens_per_second']:.2f} tokens/s")
    print(f"  Average: {metrics['avg_tokens_per_second']:.2f} tokens/s")
    print(f"  Median: {metrics['median_tokens_per_second']:.2f} tokens/s")
    print(f"  95th percentile: {metrics['p95_tokens_per_second']:.2f} tokens/s")
    print(f"  99th percentile: {metrics['p99_tokens_per_second']:.2f} tokens/s")
    print(f"\n{'─'*60}")
    print(f"Request Latency:")
    print(f"  Average: {metrics['avg_request_time']:.3f}s")
    print(f"  Median: {metrics['median_request_time']:.3f}s")
    print(f"  95th percentile: {metrics['p95_request_time']:.3f}s")
    print(f"  99th percentile: {metrics['p99_request_time']:.3f}s")
    print(f"\n{'─'*60}")
    print(f"Completion Token Statistics:")
    print(f"  Average: {metrics['avg_completion_tokens']:.1f} tokens")
    print(f"  Median: {metrics['median_completion_tokens']:.1f} tokens")
    print(f"  Min: {metrics['min_completion_tokens']} tokens")
    print(f"  Max: {metrics['max_completion_tokens']} tokens")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark tokens/second for OpenAIEngine and VerlEngine with agentica deepcoder model"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["openai", "verl"],
        required=True,
        help="Engine to benchmark (openai or verl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="agentica-org/DeepCoder-1.5B-Preview",
        help="Model name or path",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:12345/v1",
        help="Base URL for OpenAI-compatible API (for openai engine)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for OpenAI-compatible API (default: from OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=50,
        help="Number of requests to make",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=4,
        help="Number of parallel requests",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens to generate per request",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=2048,
        help="Maximum prompt length",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=16384,
        help="Maximum response length",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to veRL config file (required for verl engine)",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable thinking/reasoning tokens",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Dataset split to use for prompts (default: train)",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to load from dataset (default: None, use all or up to num_requests)",
    )
    parser.add_argument(
        "--use-sample-prompts",
        action="store_true",
        help="Use hardcoded sample prompts instead of loading from dataset",
    )
    
    args = parser.parse_args()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    # Load prompts from dataset or use sample prompts
    if args.use_sample_prompts:
        print("Using sample prompts (--use-sample-prompts)")
        prompts = [
            "Write a Python function to calculate the factorial of a number.",
            "Implement a binary search algorithm in Python.",
            "Create a function that finds the longest common subsequence between two strings.",
            "Write a Python function to sort a list of dictionaries by a specific key.",
            "Implement a function to check if a number is prime.",
            "Write a Python function to reverse a linked list.",
            "Create a function that finds all permutations of a string.",
            "Implement a function to calculate the Fibonacci sequence up to n terms.",
            "Write a Python function to merge two sorted arrays.",
            "Create a function that checks if two strings are anagrams.",
        ]
    else:
        # Load prompts from DeepCoder dataset
        max_prompts = args.max_prompts or args.num_requests
        prompts = load_deepcoder_prompts(
            dataset_split=args.dataset_split,
            max_prompts=max_prompts,
        )
    
    # Load tokenizer
    print(f"Loading tokenizer for model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("Tokenizer loaded successfully.")
    
    # Prepare sampling params
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.engine == "openai":
        sampling_params["model"] = args.model
    
    # Initialize engine and run benchmark
    metrics = None
    
    if args.engine == "openai":
        engine = OpenAIEngine(
            model=args.model,
            tokenizer=tokenizer,
            base_url=args.base_url,
            api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
            max_prompt_length=args.max_prompt_length,
            max_response_length=args.max_response_length,
            disable_thinking=args.disable_thinking,
        )
        print(f"Initialized OpenAIEngine with base_url: {args.base_url}")
        
        # Run benchmark
        metrics = asyncio.run(
            benchmark_openai_engine(
                engine,
                prompts,
                args.num_requests,
                args.num_parallel,
                sampling_params,
            )
        )
        
    elif args.engine == "verl":
        print("\n" + "="*60)
        print("NOTE: VerlEngine benchmarking requires proper veRL infrastructure setup.")
        print("="*60)
        print("\nVerlEngine requires:")
        print("  1. A veRL config file (--config-path)")
        print("  2. A rollout_manager (AsyncLLMServerManager or RayWorkerGroup)")
        print("  3. Proper initialization of async LLM servers")
        print("\nFor benchmarking VerlEngine, you have two options:")
        print("\nOption 1: Use AgentExecutionEngine with engine_name='verl'")
        print("  This requires the full training infrastructure setup.")
        print("  See examples in: rllm/trainer/verl/agent_ppo_trainer.py")
        print("\nOption 2: Set up VerlEngine directly (advanced)")
        print("  This requires manually creating the rollout_manager.")
        print("  See: verl/tests/experimental/agent_loop/agent_utils.py")
        print("\n" + "="*60)
        
        if args.config_path is None:
            raise ValueError(
                "\n--config-path is required for verl engine.\n"
                "Please provide a path to a veRL config file.\n"
                "Example configs can be found in: rllm/trainer/config/"
            )
        
        from omegaconf import OmegaConf
        
        # Load config
        print(f"\nLoading config from: {args.config_path}")
        config = OmegaConf.load(args.config_path)
        print("Config loaded successfully.")
        
        # Try to use AgentExecutionEngine approach
        # This is simpler but still requires proper setup
        print("\nAttempting to use AgentExecutionEngine with VerlEngine...")
        print("NOTE: This requires the rollout_manager to be properly initialized.")
        print("If you get errors, you may need to set up the veRL infrastructure first.")
        
        try:
            # This approach requires the rollout_manager to be set up
            # In practice, this is usually done as part of the training setup
            from rllm.engine.agent_execution_engine import AgentExecutionEngine
            
            # For VerlEngine, we need to pass the rollout_manager
            # This is typically created in the trainer initialization
            # For benchmarking, we'll try to use a simplified approach
            
            # Check if we can get rollout_manager from config
            # This is a simplified example - actual usage requires proper setup
            raise NotImplementedError(
                "\nVerlEngine benchmarking via AgentExecutionEngine requires:\n"
                "  1. A properly initialized rollout_manager\n"
                "  2. Async LLM servers running (vLLM or SGLang)\n"
                "  3. Proper Ray cluster setup (if using Ray)\n"
                "\n"
                "For a working example, see:\n"
                "  - rllm/trainer/verl/agent_ppo_trainer.py\n"
                "  - verl/tests/experimental/agent_loop/agent_utils.py\n"
                "\n"
                "Alternatively, use --engine openai to benchmark against an OpenAI-compatible API server."
            )
            
        except Exception as e:
            print(f"\nError setting up VerlEngine: {e}")
            print("\nSuggestion: Use --engine openai for simpler benchmarking.")
            raise
    
    # Print results
    if metrics is not None:
        print_metrics(metrics)
        return metrics
    else:
        print("No metrics collected. Benchmark failed.")
        return None


if __name__ == "__main__":
    main()

