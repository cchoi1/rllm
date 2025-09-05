#!/bin/bash
#SBATCH --partition=jag-standard
#SBATCH --account=nlp
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --job-name="cm_solver2"
#SBATCH --output=cm_solver2.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchoi1@stanford.edu
#SBATCH --exclude=jagupard[19-20,26-31]

# ---- Env
source /nlp/scr/cchoi1/miniconda3/etc/profile.d/conda.sh
conda activate rllm
cd /nlp/scr/cchoi1/rllm

set -a
. /nlp/scr/cchoi1/LiveCodeBench/.env
set +a

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

RLLM_DIR=$(python3 -c "import rllm, os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")

# ------------------------------
# Config
# ------------------------------
MODEL="agentica-org/DeepCoder-1.5B-Preview"
PORT=12345
HOST="0.0.0.0"                # bind to all interfaces so other nodes can reach
NUM_GPUS=1

RUN_DIR=/scr/cchoi1/rllm/runs/$(date +%m-%d-%H-%M)
mkdir -p "$RUN_DIR"

LOG="$RUN_DIR/vllm_server.log"

# ------------------------------
# Start vLLM server (no srun)
# ------------------------------
# NOTE: If you only requested 1 GPU above, set --tensor-parallel-size 1 here.
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype auto \
  --tensor-parallel-size "$NUM_GPUS" \
  2>&1 | tee -a "$LOG" &

SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
echo "Launched vLLM server (pid=$SERVER_PID); logs: $LOG"

# ------------------------------
# Readiness check (with API key)
# ------------------------------
READY=0
BASE_URL="http://127.0.0.1:${PORT}/v1"
for i in {1..30}; do
  if curl -sSf "${BASE_URL}/models" >/dev/null; then
    echo "vLLM server ready on ${BASE_URL}"
    READY=1
    break
  fi
  echo "Not ready yet... (${i}/30). Sleeping 10s."
  sleep 10
done
if [[ $READY -ne 1 ]]; then
  echo "ERROR: vLLM server failed to become ready on ${BASE_URL}"
  exit 1
fi

# Keep the server running in foreground of the job
wait "$SERVER_PID"