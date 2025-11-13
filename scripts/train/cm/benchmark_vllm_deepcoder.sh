#!/bin/bash
#SBATCH --partition=sphinx
#SBATCH --account=nlp
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --job-name="vllm_benchmark_deepcoder"
#SBATCH --output=benchmark_vllm_deepcoder.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchoi1@stanford.edu
#SBATCH --nodelist=sphinx[4-8]

# Load conda environment
source /nlp/scr/cchoi1/miniconda3/etc/profile.d/conda.sh
conda activate rllm2
cd /nlp/scr/cchoi1/rllm

set -a
. /nlp/scr/cchoi1/LiveCodeBench/.env
set +a

set -x

# Print GPU info
srun -l bash -c 'echo "Node: $(hostname -s)"; nvidia-smi -L'

# --- vLLM / torch env
unset ROCR_VISIBLE_DEVICES ROCM_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HYDRA_FULL_ERROR=1
export RAY_DISABLE_DASHBOARD=1
export CUDA_VISIBLE_DEVICES=0

export TORCH_CUDNN_V8_API_ENABLE=1
export CUBLAS_WORKSPACE_CONFIG=:16:8
export NVIDIA_TF32_OVERRIDE=0
python -c "import torch; torch.backends.cuda.matmul.allow_tf32=True"

# Run benchmark
python scripts/train/cm/benchmark_vllm_deepcoder.py \
    --num-prompts 128 \
    --num-iterations 10 \
    --warmup-iterations 3 \
    --max-tokens 16384 \
    --qwen-benchmark \
    --input-length 14336

