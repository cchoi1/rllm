#!/bin/bash
#SBATCH --partition=tiger
#SBATCH --account=tiger
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=64
#SBATCH --job-name="deepcoder_solver"
#SBATCH --output=deepcoder_solver.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchoi1@stanford.edu
#SBATCH --exclude=tiger[1-7]

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

# ------------------------------
# Config
# ------------------------------
MODEL="agentica-org/DeepCoder-1.5B-Preview"   # solver model replicated on each GPU
NUM_GPUS=8
BASE_PORT=8000                                      # endpoints will be 8000..8007

rm -rf /scr/cchoi1/rllm/runs/*
RUN_DIR=/scr/cchoi1/rllm/runs/$(date +%m-%d-%H-%M)
mkdir -p "$RUN_DIR"

# ------------------------------
# Launch one vLLM server per GPU
# ------------------------------
PIDS=()
ENDPOINTS=()

for i in $(seq 0 $((NUM_GPUS-1))); do
  PORT=$((BASE_PORT + i))
  LOG="$RUN_DIR/vllm_gpu${i}.log"

  echo "Starting replica on GPU $i -> port $PORT; log: $LOG"

  CUDA_VISIBLE_DEVICES=$i \
  vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --dtype auto \
    2>&1 | tee -a "$LOG" &

  PIDS+=($!)
  ENDPOINTS+=("http://127.0.0.1:${PORT}/v1")
done

cleanup() {
  echo "Stopping replicas..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

# ------------------------------
# Readiness checks for each replica
# ------------------------------
for idx in "${!ENDPOINTS[@]}"; do
  URL="${ENDPOINTS[$idx]}"
  echo "Waiting for replica $idx at $URL ..."
  READY=0
  for attempt in $(seq 1 30); do
    if curl -sSf "${URL}/models" >/dev/null; then
      echo "Replica $idx ready at ${URL}"
      READY=1
      break
    fi
    echo "Replica $idx not ready yet... (${attempt}/30). Sleeping 10s."
    sleep 10
  done
  if [[ $READY -ne 1 ]]; then
    echo "ERROR: Replica $idx failed to become ready at ${URL}"
    exit 1
  fi
done

echo "All replicas are up."
echo "=== Solver endpoints (use these in remote_urls) ==="
for url in "${ENDPOINTS[@]}"; do
  echo "  $url"
done

# ------------------------------
# Keep job alive while servers run
# ------------------------------
wait -n  # if any replica exits, the job ends
