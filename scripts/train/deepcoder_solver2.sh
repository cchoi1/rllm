#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000123
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=32
#SBATCH --job-name="deepcoder_1.5b_server2"
#SBATCH --output=deepcoder_1.5b_server2.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchoi1@stanford.edu
#SBATCH --exclude

module load slurm 
module load nvhpc 
module load cudnn/cuda12/9.3.0.75
export CC=gcc
export CXX=g++

source /scratch/m000123/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000123/envs/rllm
set -a; . ~/rllm/.env; set +a

set -x
echo "Node: $(hostname -s)"
nvidia-smi -L

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false

MODEL="agentica-org/DeepCoder-1.5B-Preview"
PORT=12345
HOST="0.0.0.0"   # listen on all NICs
NUM_GPUS=1

RUN_DIR=/scratch/m000123/context_manager_caroline/rllm/runs/$(date +%m-%d-%H-%M)
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/vllm_server.log"

# Advertise the endpoint for clients (FQDN works best on many clusters)
SERVER_HOST=$(hostname -f)                     # e.g., n02.marlowe.stanford.edu
ENDPOINT_FILE=/scratch/m000123/context_manager_caroline/vllm_endpoint.txt
echo "http://${SERVER_HOST}:${PORT}/v1" | tee "$ENDPOINT_FILE"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype auto \
  --tensor-parallel-size "$NUM_GPUS" \
  2>&1 | tee -a "$LOG" &

SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# Wait until server is ready (local probe)
BASE_URL="http://127.0.0.1:${PORT}/v1"
for i in {1..30}; do
  if curl -sS "${BASE_URL}/models" >/dev/null; then
    echo "vLLM ready: ${BASE_URL}"
    break
  fi
  echo "Not ready yet... ($i/30)"; sleep 10
done

wait "$SERVER_PID"