#!/bin/bash
#SBATCH --partition=jag-standard
#SBATCH --account=nlp
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --job-name="eval_cm_base_sol_base"
#SBATCH --output=cm_base_sol_base.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchoi1@stanford.edu
#SBATCH --exclude=jagupard[19-20,26-31]

source /nlp/scr/cchoi1/miniconda3/etc/profile.d/conda.sh
conda activate eval
cd /nlp/scr/cchoi1/LiveCodeBench

set -a
. /nlp/scr/cchoi1/LiveCodeBench/.env
set +a

# Start first vLLM server
CUDA_VISIBLE_DEVICES=0 vllm serve agentica-org/DeepCoder-1.5B-Preview \
  --host 0.0.0.0 --port 12345 \
  --served-model-name agentica-org/DeepCoder-1.5B-Preview \
  --tensor-parallel-size 1 --dtype bfloat16 &

MODEL_PATH="agentica-org/DeepCoder-1.5B-Preview"
# Start second vLLM server
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_PATH \
  --host 0.0.0.0 --port 12346 \
  --served-model-name agentica-org/DeepCoder-1.5B-Preview \
  --tensor-parallel-size 1 --dtype bfloat16 &

# Function to wait for a port to be ready
wait_for_port() {
  local port=$1
  echo "Waiting for port $port to be ready..."
  until nc -z 127.0.0.1 $port; do
    sleep 10
  done
  echo "Port $port is up."
}

# Wait for both servers
wait_for_port 12345
wait_for_port 12346

# Run the python command once both are up
python -m lcb_runner.multiround.cli \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --scenario codegeneration \
  --release_version release_latest \
  --tensor_parallel_size 2 \
  --codegen_n 1 --n 1 --evaluate \
  --enable_prefix_caching \
  --config ./lcb_runner/config/deepcoder-1.5b-qwen_all.yaml \
  --start_date 2025-01-01 --end_date 2025-05-01