#!/bin/bash
#SBATCH --partition=jag-standard
#SBATCH --account=nlp
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --job-name="eval_cm_mc_11-2_step50_dual"
#SBATCH --output=eval_cm_mc_11-2_step50_dual.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchoi1@stanford.edu
#SBATCH --exclude=jagupard[19-20,26-31]

# ---------------------------------------------------------------------
# Env setup
# ---------------------------------------------------------------------
source /nlp/scr/cchoi1/miniconda3/etc/profile.d/conda.sh
conda activate rllm2
cd /nlp/scr/cchoi1/rllm

set -a
. /nlp/scr/cchoi1/LiveCodeBench/.env
set +a

set -x

# Print GPU info
echo "Node: $(hostname -s)"
nvidia-smi -L || true

# --- vLLM / torch env
unset ROCR_VISIBLE_DEVICES ROCM_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HYDRA_FULL_ERROR=1

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MODEL_PATH="/nlp/scr/cchoi1/rllm/checkpoints/rllm-deepcoder/cm-deepcoder1.5b-mc-4steps-fp16-11-2/global_step_50/actor/checkpoint"
MODEL_NAME="agentica-org/DeepCoder-1.5B-Preview"

PORT0=30000   # GPU 0
PORT1=12345   # GPU 1

LOG0="$PWD/vllm_server_gpu0.log"
LOG1="$PWD/vllm_server_gpu1.log"

BASE_URL0="http://127.0.0.1:${PORT0}"
BASE_URL1="http://127.0.0.1:${PORT1}"

# Clean any previous instances (best-effort)
pkill -9 -f "vllm.serve" 2>/dev/null || true
sleep 5

# ---------------------------------------------------------------------
# Start vLLM on GPU 0, port 30000
# ---------------------------------------------------------------------
echo "Starting vLLM (GPU 0) on port ${PORT0} for model: ${MODEL_PATH}..."
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT0" \
    --served-model-name "$MODEL_NAME" \
    --data-parallel-size 1 \
    --dtype float16 \
    --gpu-memory-utilization 0.85 \
    >> "$LOG0" 2>&1 &

VLLM_PID0=$!
echo "Launched vLLM (GPU0) pid=$VLLM_PID0; log: $LOG0"

# ---------------------------------------------------------------------
# Start vLLM on GPU 1, port 12345
# ---------------------------------------------------------------------
echo "Starting vLLM (GPU 1) on port ${PORT1} for model: ${MODEL_PATH}..."
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT1" \
    --served-model-name "$MODEL_NAME" \
    --data-parallel-size 1 \
    --dtype float16 \
    --gpu-memory-utilization 0.85 \
    >> "$LOG1" 2>&1 &

VLLM_PID1=$!
echo "Launched vLLM (GPU1) pid=$VLLM_PID1; log: $LOG1"

# make sure we clean up both real processes
trap '
  echo "Cleaning up vLLM servers...";
  kill "$VLLM_PID0" 2>/dev/null || true;
  kill "$VLLM_PID1" 2>/dev/null || true;
  pkill -9 -f "vllm.serve" 2>/dev/null || true
' EXIT

# ---------------------------------------------------------------------
# Wait for GPU 0 server to be ready
# ---------------------------------------------------------------------
echo "Waiting for vLLM (GPU0, port ${PORT0}) to start..."
READY0=0
for i in {1..30}; do
    if ! kill -0 "$VLLM_PID0" 2>/dev/null; then
        echo "ERROR: vLLM (GPU0) process ($VLLM_PID0) died while starting."
        echo "Last 200 lines of $LOG0:"
        tail -200 "$LOG0" 2>/dev/null || true
        exit 1
    fi

    if curl -fsS "${BASE_URL0}/v1/models" | grep -q "$MODEL_NAME"; then
        echo "vLLM (GPU0) server is ready!"
        READY0=1
        break
    fi

    echo "Waiting for vLLM (GPU0)... (${i}/30)"
    sleep 10
done

if [[ "$READY0" -ne 1 ]]; then
    echo "ERROR: vLLM (GPU0) did not start within the expected time."
    echo "Last 200 lines of $LOG0:"
    tail -200 "$LOG0" 2>/dev/null || true
    exit 1
fi

# ---------------------------------------------------------------------
# Wait for GPU 1 server to be ready
# ---------------------------------------------------------------------
echo "Waiting for vLLM (GPU1, port ${PORT1}) to start..."
READY1=0
for i in {1..30}; do
    if ! kill -0 "$VLLM_PID1" 2>/dev/null; then
        echo "ERROR: vLLM (GPU1) process ($VLLM_PID1) died while starting."
        echo "Last 200 lines of $LOG1:"
        tail -200 "$LOG1" 2>/dev/null || true
        exit 1
    fi

    if curl -fsS "${BASE_URL1}/v1/models" | grep -q "$MODEL_NAME"; then
        echo "vLLM (GPU1) server is ready!"
        READY1=1
        break
    fi

    echo "Waiting for vLLM (GPU1)... (${i}/30)"
    sleep 10
done

if [[ "$READY1" -ne 1 ]]; then
    echo "ERROR: vLLM (GPU1) did not start within the expected time."
    echo "Last 200 lines of $LOG1:"
    tail -200 "$LOG1" 2>/dev/null || true
    exit 1
fi

# ---------------------------------------------------------------------
# Test both servers
# ---------------------------------------------------------------------
echo "Testing vLLM (GPU0, ${BASE_URL0})..."
curl -fsS -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"test gpu0\"}]}" \
    "${BASE_URL0}/v1/chat/completions" | head -20 || {
        echo "WARNING: test request to GPU0 failed; see $LOG0."
    }

echo "Testing vLLM (GPU1, ${BASE_URL1})..."
curl -fsS -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"test gpu1\"}]}" \
    "${BASE_URL1}/v1/chat/completions" | head -20 || {
        echo "WARNING: test request to GPU1 failed; see $LOG1."
    }

# ---------------------------------------------------------------------
# Run the context manager script
# (export both so Python can pick)
# ---------------------------------------------------------------------
export VLLM_BASE_URL_GPU0="$BASE_URL0"
export VLLM_BASE_URL_GPU1="$BASE_URL1"

echo "Running context manager evaluation..."
PYTHONUNBUFFERED=1 python3 -u examples/cm/run_cm.py 2>&1

echo "Evaluation completed!"
echo "Done."
