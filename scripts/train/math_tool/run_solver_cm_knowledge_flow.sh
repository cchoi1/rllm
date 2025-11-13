#!/bin/bash
#SBATCH --partition=sphinx
#SBATCH --account=nlp
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=64
#SBATCH --job-name="math_tool_solver_cm_knowledge_flow"
#SBATCH --output=math_tool_solver_cm_knowledge_flow.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchoi1@stanford.edu
#SBATCH --exclude=sphinx[1-2]

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

# Config
MODEL_NAME="Qwen/Qwen3-4B"
VLLM_PORT="${VLLM_PORT:-30000}"

# Clean any previous instances from *this user* and free the port
pkill -9 -u "$USER" -f "[v]llm\.serve" || true
sleep 3

# Start vLLM server in background; write its noisy logs to a separate file
echo "Starting vLLM server on port ${VLLM_PORT} for model: ${MODEL_NAME} ..."
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --served-model-name "$MODEL_NAME" \
  --data-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  > vllm_server.log 2>&1 &

VLLM_PID=$!
trap 'echo "Cleaning up vLLM ($VLLM_PID)"; kill $VLLM_PID 2>/dev/null || true; pkill -9 -u "$USER" -f "[v]llm\.serve" 2>/dev/null || true' EXIT

# Wait for server to be ready (60 * 5s = 5 minutes)
echo "Waiting for vLLM server to start..."
for i in {1..60}; do
  if curl -fsS "http://localhost:${VLLM_PORT}/v1/models" | grep -q "$MODEL_NAME"; then
    echo "vLLM server is ready!"
    break
  fi
  echo "Waiting for vLLM server... (${i}/60)"
  sleep 5
  if [[ $i -eq 60 ]]; then
    echo "ERROR: vLLM server did not start within 5 minutes"
    exit 1
  fi
done

# Smoke test
echo "Testing vLLM server..."
curl -fsS -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}]}" \
  "http://localhost:${VLLM_PORT}/v1/chat/completions" | head -20

# Run the solver CM knowledge flow evaluation (make Python unbuffered so prints hit the .log immediately)
echo "Starting solver CM knowledge flow evaluation..."
PYTHONUNBUFFERED=1 python3 -u examples/math_tool/run_solver_cm_knowledge_flow.py

echo "Job completed!"

