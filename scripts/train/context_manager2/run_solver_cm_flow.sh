#!/bin/bash
#SBATCH --partition=sphinx
#SBATCH --account=nlp
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=64
#SBATCH --job-name="deepcoder_solver_cm_flow_eval"
#SBATCH --output=deepcoder_solver_cm_flow_eval.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchoi1@stanford.edu

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

# Clean any previous instances
pkill -9 -f "vllm.serve" || true
sleep 5

# Start vLLM server in background
MODEL_NAME="agentica-org/DeepCoder-1.5B-Preview"
echo "Starting vLLM server on port 30000 for model: $MODEL_NAME..."
# Start vLLM server in background (logs to its own file to reduce noise)
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 30000 \
    --served-model-name "$MODEL_NAME" \
    --data-parallel-size 4 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    > vllm_server.log 2>&1 &

VLLM_PID=$!
trap 'echo "Cleaning up vLLM ($VLLM_PID)"; kill $VLLM_PID 2>/dev/null || true' EXIT

echo "Waiting for vLLM server to start..."
for i in {1..30}; do
    if curl -fsS "http://localhost:30000/v1/models" | grep -q "$MODEL_NAME"; then
        echo "vLLM server is ready!"
        break
    fi
    echo "Waiting for vLLM server... (${i}/30)"
    sleep 20
    if [[ $i -eq 30 ]]; then
        echo "ERROR: vLLM server did not start within 5 minutes"
        exit 1
    fi
done

echo "Testing vLLM server..."
curl -fsS -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}]}" \
    "http://localhost:30000/v1/chat/completions" | head -20

echo "Starting solver CM flow evaluation..."
# Unbuffered so stdout/stderr show up in the Slurm .log immediately
PYTHONUNBUFFERED=1 python3 -u examples/context_manager2/run_solver_cm_flow.py 2>&1

echo "Cleaning up..."
kill $VLLM_PID || true

