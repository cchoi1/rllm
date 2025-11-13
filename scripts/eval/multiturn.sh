#!/bin/bash
#SBATCH --partition=jag-standard
#SBATCH --account=nlp
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --job-name="eval_multiturn_mc_step50"
#SBATCH --output=multiturn_mc_step50.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchoi1@stanford.edu
#SBATCH --exclude=jagupard[19-20,26-31]

source /nlp/scr/cchoi1/miniconda3/etc/profile.d/conda.sh
conda activate rllm2
cd /nlp/scr/cchoi1/rllm

set -a
. /nlp/scr/cchoi1/LiveCodeBench/.env
set +a

set -x

# Print GPU info
echo "Node: $(hostname -s)"
nvidia-smi -L

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
MODEL_PATH="/nlp/scr/cchoi1/rllm/checkpoints/rllm-deepcoder/multiturn-deepcoder1.5b-mc-4steps/global_step_50/actor/checkpoint"
MODEL_NAME="agentica-org/DeepCoder-1.5B-Preview"
VLLM_PORT=12345

# Clean any previous instances
pkill -9 -f "vllm.serve" || true
sleep 5

# Start vLLM server in background
echo "Starting vLLM server on port ${VLLM_PORT} for model: ${MODEL_PATH}..."
CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --served-model-name "$MODEL_NAME" \
    --data-parallel-size 2 \
    --dtype float16 \
    --gpu-memory-utilization 0.85 \
    > vllm_server.log 2>&1 &

VLLM_PID=$!
trap 'echo "Cleaning up vLLM ($VLLM_PID)"; kill $VLLM_PID 2>/dev/null || true; pkill -9 -f "vllm.serve" 2>/dev/null || true' EXIT

# Wait for server to be ready
echo "Waiting for vLLM server to start..."
for i in {1..60}; do
    if curl -fsS "http://localhost:${VLLM_PORT}/v1/models" | grep -q "$MODEL_NAME"; then
        echo "vLLM server is ready!"
        break
    fi
    echo "Waiting for vLLM server... (${i}/60)"
    sleep 10
    if [[ $i -eq 60 ]]; then
        echo "ERROR: vLLM server did not start within 10 minutes"
        exit 1
    fi
done

# Test vLLM server
echo "Testing vLLM server..."
curl -fsS -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}]}" \
    "http://localhost:${VLLM_PORT}/v1/chat/completions" | head -20

# Run the context manager script
echo "Running context manager evaluation..."
PYTHONUNBUFFERED=1 python3 -u examples/rllm/cm/run_cm.py 2>&1

echo "Evaluation completed!"
echo "Cleaning up..."
kill $VLLM_PID || true
