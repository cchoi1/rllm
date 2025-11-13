#!/bin/bash
#SBATCH --partition=tiger
#SBATCH --account=tiger
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=48
#SBATCH --job-name="deepcoder_solver_memory"
#SBATCH --output=deepcoder_solver_memory.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchoi1@stanford.edu
#SBATCH --exclude=tiger[1-6],tiger-hgx-1

source /nlp/scr/cchoi1/miniconda3/etc/profile.d/conda.sh
conda activate rllm
cd /nlp/scr/cchoi1/rllm

# ---- Load secrets (if needed by clients)
set -a
. /nlp/scr/cchoi1/LiveCodeBench/.env
set +a

# ==========================
# Run dir + logging
# ==========================
rm -rf /scr/cchoi1/rllm/runs/*
RUN_DIR=/scr/cchoi1/rllm/runs/$(date +%m-%d-%H-%M)
mkdir -p "$RUN_DIR"

RUN_LOG="$RUN_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

set -x
echo "Node: $(hostname -s)"
nvidia-smi -L

# ==========================
# General runtime env
# ==========================
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# (Optional) flash-attn / memory knobs (safe defaults)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

# Scratch isolation
export RAY_TMPDIR="/tmp/$USER/ray_${SLURM_JOB_ID:-$$}"
export TMPDIR="/tmp/$USER/tmp_${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

# Ensure we don't accidentally attach to a stray Ray cluster
unset RAY_ADDRESS RAY_NAMESPACE RAY_RUNTIME_ENV_DIR RAY_HEAD_NODE
export RAY_NAMESPACE="cm-${USER}-$$"
ray stop -f || true
rm -rf /tmp/ray /tmp/*/ray 2>/dev/null || true
echo "RAY_TMPDIR=$RAY_TMPDIR  RAY_NAMESPACE=$RAY_NAMESPACE"

# ==========================
# Config
# ==========================
MODEL="agentica-org/DeepCoder-1.5B-Preview"
NUM_GPUS=8
HOST="0.0.0.0"
PORT=12345

LOG="$RUN_DIR/sglang_server.log"
SERVER_HOST_FQDN="$(hostname -f)"
ENDPOINT_FILE=/scr/cchoi1/rllm/sglang_endpoint_memory.txt

# Advertise endpoint (use FQDN so other machines can reach it)
echo "http://${SERVER_HOST_FQDN}:${PORT}" | tee "$ENDPOINT_FILE"

# Pin visible GPUs (Slurm usually does this, but we make it explicit)
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))

# ==========================
# Launch SGLang server
# ==========================
# Notes:
#   --tp-size: tensor parallel across NUM_GPUS on a single node
#   If your SGLang build expects a different flag (rare), change to --dp-size/--tp-size accordingly.
python -m sglang.launch_server \
  --model-path "$MODEL" \
  --dp-size "$NUM_GPUS" \
  --mem-fraction-static 0.88 \
  --max-prefill-tokens 16384 \
  --chunked-prefill-size 4096 \
  --enable-metrics \
  --host "$HOST" \
  --port "$PORT" \
  2>&1 | tee -a "$LOG" &

SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
echo "Launched SGLang server (pid=$SERVER_PID); logs: $LOG"

# ==========================
# Readiness check
# ==========================
READY=0
BASE_URL="http://127.0.0.1:${PORT}"
for i in {1..30}; do
  # Try OpenAI-ish endpoints first; fall back to landing/docs
  if curl -fsS "${BASE_URL}/v1/models" >/dev/null \
     || curl -fsS "${BASE_URL}/" >/dev/null \
     || curl -fsS "${BASE_URL}/docs" >/dev/null; then
    echo "SGLang ready at ${BASE_URL}"
    READY=1
    break
  fi
  echo "Not ready yet... (${i}/30). Sleeping 10s."
  sleep 20
done

if [[ "$READY" -ne 1 ]]; then
  echo "ERROR: SGLang did not become ready." >&2
  exit 1
fi

# Keep the server in the foreground for Slurm to track
wait "$SERVER_PID"