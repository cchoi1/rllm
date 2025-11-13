#!/usr/bin/env python3
"""
Benchmark script for vLLM server with exact actor policy parameters.
Measures tokens/s to compare with vLLM qwen speed test results.

Usage:
    # Sync mode with actual DeepCoder training problems (default)
    python benchmark_vllm_deepcoder.py --num-prompts 128 --num-iterations 10

    # Async mode (matches actor_rollout_ref.rollout.mode=async)
    python benchmark_vllm_deepcoder.py --use-async --num-prompts 128 --num-iterations 10

    # Use synthetic prompts instead of dataset
    python benchmark_vllm_deepcoder.py --use-synthetic --num-prompts 128

    # Use test split instead of train
    python benchmark_vllm_deepcoder.py --dataset-split test --num-prompts 128

    # Match Qwen benchmark methodology (batch_size=1, 2048 tokens, Qwen sampling params)
    python benchmark_vllm_deepcoder.py --qwen-benchmark --input-length 1

Parameters match exactly those from scripts/train/cm/mc.sh (default):
    - temperature=0.6
    - top_p=0.95
    - gpu_memory_utilization=0.85
    - max_num_batched_tokens=32768
    - max_num_seqs=128
    - enforce_eager=False
    - tensor_parallel_size=1
    - dtype=float16

When --qwen-benchmark is used, matches Qwen speed benchmark methodology:
    - batch_size=1 (num_prompts=1)
    - max_tokens=2048
    - temperature=0.7, top_p=0.8, top_k=20
    - gpu_memory_utilization=0.9
    - max_model_len=32768 (or based on input_length)
    - See: https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html
"""
import argparse
import asyncio
import time
from typing import List, Optional

import torch
from vllm import LLM, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark vLLM with exact actor policy parameters")
    parser.add_argument("--model", type=str, default="agentica-org/DeepCoder-1.5B-Preview",
                        help="Model path")
    parser.add_argument("--num-prompts", type=int, default=128,
                        help="Number of concurrent prompts to benchmark")
    parser.add_argument("--num-iterations", type=int, default=10,
                        help="Number of benchmark iterations")
    parser.add_argument("--warmup-iterations", type=int, default=3,
                        help="Number of warmup iterations")
    parser.add_argument("--prompt-length", type=int, default=100,
                        help="Length of each prompt in tokens (only used if dataset not available)")
    parser.add_argument("--max-tokens", type=int, default=16384,
                        help="Maximum tokens to generate")
    parser.add_argument("--use-async", action="store_true",
                        help="Use AsyncLLM (matches actor_rollout_ref.rollout.mode=async)")
    parser.add_argument("--dataset-split", type=str, default="train",
                        choices=["train", "test"],
                        help="Dataset split to use (train or test)")
    parser.add_argument("--use-synthetic", action="store_true",
                        help="Use synthetic prompts instead of actual dataset problems")
    parser.add_argument("--qwen-benchmark", action="store_true",
                        help="Use Qwen benchmark methodology (batch_size=1, 2048 tokens, Qwen sampling params)")
    parser.add_argument("--input-length", type=int, default=None,
                        help="Input length in tokens (for Qwen benchmark mode). Options: 1, 6144, 14336, 30720, 63488, 129024")
    return parser.parse_args()


def load_deepcoder_problems(num_prompts: int, split: str = "train") -> List[str]:
    """Load actual problems from DeepCoder training dataset."""
    try:
        from rllm.data.dataset import DatasetRegistry
        
        # Try to load the chunked dataset first (used in training)
        try:
            dataset = DatasetRegistry.load_dataset("deepcoder_chunked", split)
            print(f"Loaded 'deepcoder_chunked' dataset with {len(dataset.get_data())} examples")
        except Exception as e:
            print(f"Could not load 'deepcoder_chunked', trying 'deepcoder': {e}")
            # Fallback to regular deepcoder dataset
            dataset = DatasetRegistry.load_dataset("deepcoder", split)
            print(f"Loaded 'deepcoder' dataset with {len(dataset.get_data())} examples")
        
        data = dataset.get_data()
        
        # Extract questions from the dataset
        prompts = []
        for example in data[:num_prompts]:
            if "question" in example:
                prompts.append(example["question"])
            elif "prompt" in example:
                # Handle case where prompt might be a list of messages
                if isinstance(example["prompt"], list):
                    # Extract text from messages
                    prompt_text = ""
                    for msg in example["prompt"]:
                        if isinstance(msg, dict) and "content" in msg:
                            prompt_text += msg["content"] + "\n"
                    prompts.append(prompt_text.strip())
                else:
                    prompts.append(str(example["prompt"]))
            else:
                # Fallback: use the first string value
                for key, value in example.items():
                    if isinstance(value, str) and len(value) > 50:
                        prompts.append(value)
                        break
        
        if len(prompts) < num_prompts:
            print(f"Warning: Only found {len(prompts)} examples, requested {num_prompts}")
            # Repeat prompts if needed
            while len(prompts) < num_prompts:
                prompts.extend(prompts[:num_prompts - len(prompts)])
        
        print(f"Loaded {len(prompts)} prompts from DeepCoder {split} dataset")
        return prompts[:num_prompts]
        
    except Exception as e:
        print(f"Error loading DeepCoder dataset: {e}")
        print("Falling back to synthetic prompts...")
        return None


def create_test_prompts(num_prompts: int, prompt_length: int, tokenizer) -> List[str]:
    """Create synthetic test prompts for benchmarking (fallback)."""
    # Use a simple prompt that will generate tokens
    base_prompt = "Write a Python function to solve the following problem:\n"
    
    prompts = []
    for i in range(num_prompts):
        # Create varied prompts
        prompt = f"{base_prompt}Problem {i}: " + "Write code " * (prompt_length // 10)
        # Tokenize and truncate to exact length
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) > prompt_length:
            token_ids = token_ids[:prompt_length]
            prompt = tokenizer.decode(token_ids)
        elif len(token_ids) < prompt_length:
            # Pad with tokens to reach exact length
            padding = tokenizer.encode(" pad", add_special_tokens=False)
            while len(token_ids) < prompt_length:
                token_ids.extend(padding)
            token_ids = token_ids[:prompt_length]
            prompt = tokenizer.decode(token_ids)
        prompts.append(prompt)
    
    return prompts


def benchmark_sync_llm(llm: LLM, prompts: List[str], sampling_params: SamplingParams, 
                       num_iterations: int, warmup_iterations: int):
    """Benchmark synchronous LLM."""
    print(f"Running {warmup_iterations} warmup iterations...")
    for _ in range(warmup_iterations):
        _ = llm.generate(prompts[:8], sampling_params, use_tqdm=False)
    
    print(f"Running {num_iterations} benchmark iterations...")
    total_tokens = 0
    total_time = 0.0
    
    all_iteration_stats = []
    
    for iteration in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        iteration_time = end_time - start_time
        
        # Count generated tokens
        iteration_tokens = 0
        for output in outputs:
            for choice in output.outputs:
                iteration_tokens += len(choice.token_ids)
        
        all_iteration_stats.append({
            'tokens': iteration_tokens,
            'time': iteration_time,
            'tokens_per_sec': iteration_tokens / iteration_time if iteration_time > 0 else 0
        })
        
        total_tokens += iteration_tokens
        total_time += iteration_time
        
        print(f"Iteration {iteration + 1}/{num_iterations}: "
              f"{iteration_tokens} tokens in {iteration_time:.2f}s = "
              f"{iteration_tokens/iteration_time:.2f} tokens/s")
    
    avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    
    # Calculate statistics
    tokens_per_sec_list = [s['tokens_per_sec'] for s in all_iteration_stats]
    mean_tps = sum(tokens_per_sec_list) / len(tokens_per_sec_list) if tokens_per_sec_list else 0
    min_tps = min(tokens_per_sec_list) if tokens_per_sec_list else 0
    max_tps = max(tokens_per_sec_list) if tokens_per_sec_list else 0
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average throughput: {avg_tokens_per_sec:.2f} tokens/s")
    print(f"Mean tokens/s: {mean_tps:.2f}")
    print(f"Min tokens/s: {min_tps:.2f}")
    print(f"Max tokens/s: {max_tps:.2f}")
    print("="*60)
    
    return {
        'total_tokens': total_tokens,
        'total_time': total_time,
        'avg_tokens_per_sec': avg_tokens_per_sec,
        'mean_tps': mean_tps,
        'min_tps': min_tps,
        'max_tps': max_tps,
        'all_stats': all_iteration_stats
    }


async def benchmark_async_llm(async_llm: AsyncLLM, prompt_token_ids: List[List[int]], 
                               sampling_params: SamplingParams,
                               num_iterations: int, warmup_iterations: int):
    """Benchmark asynchronous LLM."""
    print(f"Running {warmup_iterations} warmup iterations...")
    for _ in range(warmup_iterations):
        tasks = []
        for prompt_ids in prompt_token_ids[:8]:
            prompt_obj = TokensPrompt(prompt_token_ids=prompt_ids)
            task = async_llm.generate(prompt=prompt_obj, sampling_params=sampling_params, 
                                     request_id=f"warmup_{_}_{len(tasks)}")
            tasks.append(task)
        # Consume all outputs
        for task in tasks:
            final_output = None
            async for output in task:
                final_output = output
            # Consume the output
    
    print(f"Running {num_iterations} benchmark iterations...")
    total_tokens = 0
    total_time = 0.0
    
    all_iteration_stats = []
    
    for iteration in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Create tasks for all prompts
        tasks = []
        for i, prompt_ids in enumerate(prompt_token_ids):
            prompt_obj = TokensPrompt(prompt_token_ids=prompt_ids)
            task = async_llm.generate(prompt=prompt_obj, sampling_params=sampling_params,
                                     request_id=f"bench_{iteration}_{i}")
            tasks.append(task)
        
        # Wait for all generations to complete
        outputs = []
        for task in tasks:
            final_output = None
            async for output in task:
                final_output = output
            if final_output:
                outputs.append(final_output)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        iteration_time = end_time - start_time
        
        # Count generated tokens
        iteration_tokens = 0
        for output in outputs:
            if hasattr(output, 'outputs') and output.outputs:
                for choice in output.outputs:
                    if hasattr(choice, 'token_ids'):
                        iteration_tokens += len(choice.token_ids)
        
        all_iteration_stats.append({
            'tokens': iteration_tokens,
            'time': iteration_time,
            'tokens_per_sec': iteration_tokens / iteration_time if iteration_time > 0 else 0
        })
        
        total_tokens += iteration_tokens
        total_time += iteration_time
        
        print(f"Iteration {iteration + 1}/{num_iterations}: "
              f"{iteration_tokens} tokens in {iteration_time:.2f}s = "
              f"{iteration_tokens/iteration_time:.2f} tokens/s")
    
    avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    
    # Calculate statistics
    tokens_per_sec_list = [s['tokens_per_sec'] for s in all_iteration_stats]
    mean_tps = sum(tokens_per_sec_list) / len(tokens_per_sec_list) if tokens_per_sec_list else 0
    min_tps = min(tokens_per_sec_list) if tokens_per_sec_list else 0
    max_tps = max(tokens_per_sec_list) if tokens_per_sec_list else 0
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average throughput: {avg_tokens_per_sec:.2f} tokens/s")
    print(f"Mean tokens/s: {mean_tps:.2f}")
    print(f"Min tokens/s: {min_tps:.2f}")
    print(f"Max tokens/s: {max_tps:.2f}")
    print("="*60)
    
    return {
        'total_tokens': total_tokens,
        'total_time': total_time,
        'avg_tokens_per_sec': avg_tokens_per_sec,
        'mean_tps': mean_tps,
        'min_tps': min_tps,
        'max_tps': max_tps,
        'all_stats': all_iteration_stats
    }


def main():
    args = parse_args()
    
    # Determine if using Qwen benchmark methodology
    if args.qwen_benchmark:
        # Qwen benchmark parameters (from https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html)
        TEMPERATURE = 0.7
        TOP_P = 0.8
        TOP_K = 20
        GPU_MEMORY_UTILIZATION = 0.9
        MAX_NUM_BATCHED_TOKENS = 32768
        MAX_NUM_SEQS = 1  # Qwen uses batch_size=1
        ENFORCE_EAGER = False
        TENSOR_PARALLEL_SIZE = 1
        DTYPE = "float16"
        MAX_RESPONSE_LENGTH = 2048  # Qwen generates 2048 tokens
        # Set num_prompts to 1 for Qwen benchmark
        if args.num_prompts != 1:
            print(f"Warning: Qwen benchmark uses batch_size=1, overriding num_prompts from {args.num_prompts} to 1")
            args.num_prompts = 1
        
        # Determine max_model_len based on input_length
        if args.input_length is not None:
            input_length = args.input_length
            # Qwen uses max_model_len based on context length requirements
            # For input_length <= 30720, use 32768; for larger, use appropriate values
            if input_length <= 30720:
                MAX_MODEL_LEN = 32768
            elif input_length <= 63488:
                MAX_MODEL_LEN = 65536
            else:
                MAX_MODEL_LEN = 131072
        else:
            # Default to 32768 if input_length not specified
            MAX_MODEL_LEN = 32768
            args.input_length = 1  # Default input length
    else:
        # Exact parameters from actor_rollout_ref.rollout config
        TEMPERATURE = 0.6
        TOP_P = 0.95
        TOP_K = None  # Not used in actor policy
        GPU_MEMORY_UTILIZATION = 0.85
        MAX_NUM_BATCHED_TOKENS = 32768
        MAX_NUM_SEQS = 128
        ENFORCE_EAGER = False
        TENSOR_PARALLEL_SIZE = 1
        DTYPE = "float16"
        MAX_RESPONSE_LENGTH = args.max_tokens
        MAX_MODEL_LEN = None  # Let vLLM determine automatically
    
    print("="*60)
    if args.qwen_benchmark:
        print("vLLM Benchmark - Qwen Methodology")
    else:
        print("vLLM Benchmark with Actor Policy Parameters")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Mode: {'async' if args.use_async else 'sync'}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Top-p: {TOP_P}")
    if TOP_K is not None:
        print(f"Top-k: {TOP_K}")
    print(f"GPU Memory Utilization: {GPU_MEMORY_UTILIZATION}")
    print(f"Max Num Batched Tokens: {MAX_NUM_BATCHED_TOKENS}")
    print(f"Max Num Seqs: {MAX_NUM_SEQS}")
    if MAX_MODEL_LEN is not None:
        print(f"Max Model Length: {MAX_MODEL_LEN}")
    print(f"Enforce Eager: {ENFORCE_EAGER}")
    print(f"Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")
    print(f"Dtype: {DTYPE}")
    print(f"Max Response Length: {MAX_RESPONSE_LENGTH}")
    print(f"Num Prompts: {args.num_prompts}")
    if args.qwen_benchmark and args.input_length:
        print(f"Input Length: {args.input_length} tokens")
    print("="*60)
    
    # Create sampling params
    sampling_params_dict = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_RESPONSE_LENGTH,
        "n": 1,  # Number of outputs per prompt
    }
    if TOP_K is not None:
        sampling_params_dict["top_k"] = TOP_K
    # Qwen benchmark also uses these parameters
    if args.qwen_benchmark:
        sampling_params_dict["repetition_penalty"] = 1.0
        sampling_params_dict["presence_penalty"] = 0.0
        sampling_params_dict["frequency_penalty"] = 0.0
    
    sampling_params = SamplingParams(**sampling_params_dict)
    
    if args.use_async:
        # Use AsyncLLM (matches actor_rollout_ref.rollout.mode=async)
        print("\nInitializing AsyncLLM...")
        engine_args_dict = {
            "model": args.model,
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "dtype": DTYPE,
            "enforce_eager": ENFORCE_EAGER,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "enable_prefix_caching": True,
            "trust_remote_code": True,
        }
        if MAX_MODEL_LEN is not None:
            engine_args_dict["max_model_len"] = MAX_MODEL_LEN
        
        engine_args = AsyncEngineArgs(**engine_args_dict)
        
        async_llm = AsyncLLM.from_engine_args(engine_args)
        
        # Get tokenizer to create prompts
        # Note: AsyncLLM might not have get_tokenizer, so we'll use transformers
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        
        # Load actual DeepCoder problems or use synthetic
        if args.qwen_benchmark and args.input_length:
            # For Qwen benchmark, create prompts with exact input length
            prompts = create_test_prompts(args.num_prompts, args.input_length, tokenizer)
        elif args.use_synthetic:
            prompts = create_test_prompts(args.num_prompts, args.prompt_length, tokenizer)
        else:
            prompts = load_deepcoder_problems(args.num_prompts, args.dataset_split)
            if prompts is None:
                # Fallback to synthetic if dataset loading fails
                if args.qwen_benchmark and args.input_length:
                    prompts = create_test_prompts(args.num_prompts, args.input_length, tokenizer)
                else:
                    prompts = create_test_prompts(args.num_prompts, args.prompt_length, tokenizer)
        
        # Convert prompts to token IDs for AsyncLLM
        prompt_token_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
        
        # Run async benchmark
        results = asyncio.run(benchmark_async_llm(
            async_llm, prompt_token_ids, sampling_params,
            args.num_iterations, args.warmup_iterations
        ))
    else:
        # Use synchronous LLM
        print("\nInitializing LLM...")
        llm_kwargs = {
            "model": args.model,
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "dtype": DTYPE,
            "enforce_eager": ENFORCE_EAGER,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "max_num_seqs": MAX_NUM_SEQS,
            "enable_prefix_caching": True,
            "trust_remote_code": True,
        }
        if MAX_MODEL_LEN is not None:
            llm_kwargs["max_model_len"] = MAX_MODEL_LEN
        
        llm = LLM(**llm_kwargs)
        
        # Get tokenizer to create prompts
        try:
            tokenizer = llm.get_tokenizer()
        except AttributeError:
            # Fallback to transformers if get_tokenizer doesn't exist
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        
        # Load actual DeepCoder problems or use synthetic
        if args.qwen_benchmark and args.input_length:
            # For Qwen benchmark, create prompts with exact input length
            prompts = create_test_prompts(args.num_prompts, args.input_length, tokenizer)
        elif args.use_synthetic:
            prompts = create_test_prompts(args.num_prompts, args.prompt_length, tokenizer)
        else:
            prompts = load_deepcoder_problems(args.num_prompts, args.dataset_split)
            if prompts is None:
                # Fallback to synthetic if dataset loading fails
                if args.qwen_benchmark and args.input_length:
                    prompts = create_test_prompts(args.num_prompts, args.input_length, tokenizer)
                else:
                    prompts = create_test_prompts(args.num_prompts, args.prompt_length, tokenizer)
        
        # Run sync benchmark
        results = benchmark_sync_llm(
            llm, prompts, sampling_params,
            args.num_iterations, args.warmup_iterations
        )
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()

