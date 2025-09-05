#!/bin/bash
#SBATCH --partition=tiger
#SBATCH --account=tiger
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=64
#SBATCH --job-name="cm_mc_small_2_steps_tiger"
#SBATCH --output=cm_mc_small_2_steps_tiger.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchoi1@stanford.edu
#SBATCH --exclude=tiger[1-5]

# Load conda environment
source /nlp/scr/cchoi1/miniconda3/etc/profile.d/conda.sh
conda activate rllm
cd /nlp/scr/cchoi1/rllm

set -a
. /nlp/scr/cchoi1/LiveCodeBench/.env
set +a

set -x

# Print GPU info
srun -l bash -c 'echo "Node: $(hostname -s)"; nvidia-smi -L'

# --- vLLM / torch env
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HYDRA_FULL_ERROR=1

# --- scratch; isolate by user+pid or Slurm job id
export RAY_TMPDIR="/tmp/$USER/ray_${SLURM_JOB_ID:-$$}"
export TMPDIR="/tmp/$USER/tmp_${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

# --- guarantee we don't attach to someone else's Ray
unset RAY_ADDRESS RAY_NAMESPACE RAY_RUNTIME_ENV_DIR RAY_HEAD_NODE
export RAY_NAMESPACE="cm-${USER}-$$"

# clean any previous instances and stray shared dirs
ray stop -f || true
rm -rf /tmp/ray /tmp/*/ray 2>/dev/null || true

echo "RAY_TMPDIR=$RAY_TMPDIR  RAY_ADDRESS=${RAY_ADDRESS:-<unset>}  RAY_NAMESPACE=$RAY_NAMESPACE"

RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")

# ------------------------------
# Config
# ------------------------------
MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B-Instruct"

RUN_DIR=/scr/cchoi1/rllm/runs/$(date +%m-%d-%H-%M)
mkdir -p "$RUN_DIR"

# vLLM server settings (GPU 0)
VLLM_HOST=0.0.0.0
VLLM_BIND=127.0.0.1
VLLM_PORT=12345

# ------------------------------
# 1) Start vLLM server on the 3rd GPU (GPU 0)
# ------------------------------
Use a separate env for the server so it sees only GPU 0.
srun --ntasks=1 --gpus-per-task=1 --exclusive -c 16 --mem=64GB \
  --output="$RUN_DIR/vllm_server.log" --error="$RUN_DIR/vllm_server.log" \
  bash -lc "python3 -m vllm.entrypoints.openai.api_server \
    --model '${MODEL_PATH}' \
    --host '${VLLM_HOST}' \
    --port ${VLLM_PORT} \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 32768" &

VLLM_STEP_PID=$!
trap 'kill "$VLLM_STEP_PID" 2>/dev/null || true' EXIT
echo "Launched vLLM server (srun step pid=$VLLM_STEP_PID); logs: $RUN_DIR/vllm_server.log"

# Wait for vLLM to become ready
READY=0
for i in {1..30}; do
  if curl -s "http://$VLLM_BIND:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
    READY=1; break
  fi
  sleep 30
done
if [[ $READY -ne 1 ]]; then
  echo "vLLM server failed to start on port $VLLM_PORT"; exit 1
fi

NUM_GPUS=4

srun --ntasks=1 --gpus-per-task=${NUM_GPUS} --exclusive -c 48 --mem=192GB \
  --output="./cm_mc_small_2steps_tiger.log" --error="./cm_mc_small_2steps_tiger.log" \
  bash -lc "python3 -m examples.context_manager.train_cm \
    agent.max_steps=2 \
    agent.use_stepwise_advantage=True \
    agent.normalize_step_advantage=True \
    agent.stepwise_advantage_mode=mc_return \
    algorithm.gamma=0.95 \
    algorithm.lam=0.95 \
    hydra.run.dir='$RUN_DIR' \
    +trainer.save_dir='$RUN_DIR/checkpoints' \
    ++trainer.resume_mode=resume_path \
    ++trainer.resume_from_path=/nlp/scr/cchoi1/rllm/checkpoints/rllm-deepcoder/cm-qwen1.5b-coder-mc-4k-2steps/global_step_15 \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=16 \
    data.val_batch_size=32 \
    data.max_prompt_length=6400 \
    data.max_response_length=1600 \
    actor_rollout_ref.model.path='$MODEL_PATH' \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    +actor_rollout_ref.model.override_config.torch_dtype=bfloat16 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$NUM_GPUS \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode='async' \
    actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.mask_truncated_samples=True \
    algorithm.clip_advantages=False \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb','console'] \
    trainer.project_name='rllm-deepcoder' \
    trainer.experiment_name='cm-qwen1.5b-coder-mc-4k-2steps-tiger' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node='$NUM_GPUS' \
    trainer.n_training_gpus_per_node='$NUM_GPUS' \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=100"

# Stop vLLM server after training
# kill "$VLLM_STEP_PID" 2>/dev/null || true