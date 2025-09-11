#!/bin/bash
#SBATCH --partition=tiger
#SBATCH --account=tiger
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=24
#SBATCH --job-name="deepcoder_solver"
#SBATCH --output=deepcoder_solver.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchoi1@stanford.edu
#SBATCH --exclude=tiger[1-5]

# ---- Env
source /nlp/scr/cchoi1/miniconda3/etc/profile.d/conda.sh
conda activate rllm
cd /nlp/scr/cchoi1/rllm

set -a
. /nlp/scr/cchoi1/LiveCodeBench/.env
set +a

# ------------------------------
# Run dir under scratch + logging
# ------------------------------
rm -rf /scr/cchoi1/rllm/runs/*
RUN_DIR=/scr/cchoi1/rllm/runs/$(date +%m-%d-%H-%M)
mkdir -p "$RUN_DIR"

# Redirect all subsequent stdout/stderr into a run log inside RUN_DIR
RUN_LOG="$RUN_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

set -x
echo "Node: $(hostname -s)"
nvidia-smi -L

# ---- vLLM / torch env
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false

# ---- scratch; isolate by user+pid or Slurm job id
export RAY_TMPDIR="/tmp/$USER/ray_${SLURM_JOB_ID:-$$}"
export TMPDIR="/tmp/$USER/tmp_${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

# ---- guarantee we don't attach to someone else's Ray
unset RAY_ADDRESS RAY_NAMESPACE RAY_RUNTIME_ENV_DIR RAY_HEAD_NODE
export RAY_NAMESPACE="cm-${USER}-$$"
ray stop -f || true
rm -rf /tmp/ray /tmp/*/ray 2>/dev/null || true
echo "RAY_TMPDIR=$RAY_TMPDIR  RAY_NAMESPACE=$RAY_NAMESPACE"

# ------------------------------
# Config
# ------------------------------
MODEL="agentica-org/DeepCoder-1.5B-Preview"   # solver model replicated on each GPU
NUM_GPUS=4
BASE_PORT=8000                                 # endpoints will be 8000..8007

# ------------------------------
# Launch vLLM server
# ------------------------------
LOG="$RUN_DIR/vllm_server.log"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 12345 \
  --tensor-parallel-size $NUM_GPUS \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --dtype auto \
  2>&1 | tee -a "$LOG" &

SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
echo "Launched vLLM server (pid=$SERVER_PID); logs: $LOG"

# ------------------------------
# Readiness check (with API key)
# ------------------------------
READY=0
BASE_URL="http://127.0.0.1:12345/v1"
for i in {1..30}; do
  if curl -sSf "${BASE_URL}/models" >/dev/null; then
    echo "vLLM server ready on ${BASE_URL}"
    READY=1
    break
  fi
  echo "Not ready yet... (${i}/30). Sleeping 10s."
  sleep 10
done

if [[ "$READY" -ne 1 ]]; then
  echo "ERROR: vLLM did not become ready." >&2
  exit 1
fi

wait "$SERVER_PID"