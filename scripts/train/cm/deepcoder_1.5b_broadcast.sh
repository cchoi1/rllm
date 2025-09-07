#!/bin/bash 
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000123
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=512GB
#SBATCH --cpus-per-task=100
#SBATCH --job-name="deepcoder_1.5b_cm_broadcast_4_steps"
#SBATCH --output=deepcoder_1.5b_cm_broadcast_4_steps.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchoi1@stanford.edu

# --- Modules / compilers: avoid NVHPC so Triton uses gcc/clang
module load slurm 
module load nvhpc 
module load cudnn/cuda12/9.3.0.75 
export CC=gcc
export CXX=g++

# --- Conda / env
source /scratch/m000123/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000123/envs/rllm 
set -a
. ~/rllm/.env
set +a

# --- vLLM / torch env
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

# ------------------------------
# Config
# ------------------------------
MODEL_PATH="agentica-org/DeepCoder-1.5B-Preview"
RUN_DIR=/scratch/m000123/context_manager_caroline/rllm/runs/$(date +%m-%d-%H-%M)
mkdir -p "$RUN_DIR"

# ---- Remote vLLM endpoint on Node A ----
# Preferred: server writes this path when it starts (see previous reply)
ENDPOINT_FILE=/scratch/m000123/vllm_endpoint.txt

# Fallback hostname if file isn't present
FALLBACK_HOST="n02.marlowe.stanford.edu"
FALLBACK_PORT=12345

if [[ -f "$ENDPOINT_FILE" ]]; then
  VLLM_BASE_URL="$(cat "$ENDPOINT_FILE")"
else
  VLLM_BASE_URL="http://${FALLBACK_HOST}:${FALLBACK_PORT}/v1"
fi

echo "=== vLLM Solver Configuration ==="
echo "Base URL: $VLLM_BASE_URL"
echo "Model:    $MODEL_PATH"
echo "=================================="

# ------------------------------
# Remote readiness probe
# ------------------------------
READY=0
for i in {1..20}; do
  # vLLM ignores API key by default; header kept for compatibility
  if curl -sS -H "Authorization: " "${VLLM_BASE_URL}/models" | grep -q "${MODEL_PATH}"; then
    echo "Remote vLLM is reachable and serving ${MODEL_PATH}"
    READY=1
    break
  fi
  echo "Waiting for remote vLLM at ${VLLM_BASE_URL} ... (${i}/20)"
  sleep 6
done
if [[ $READY -ne 1 ]]; then
  echo "ERROR: Could not reach remote vLLM at ${VLLM_BASE_URL} or model not listed."
  exit 1
fi

# quick ping
curl -sS -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_PATH\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}" \
  "${VLLM_BASE_URL}/chat/completions" || true
echo

NUM_GPUS=8

RLLM_DIR=$(python3 -c "import rllm, os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
cd "$RLLM_DIR"

python3 -m examples.context_manager.train_cm \
    agent.max_steps=4 \
    agent.use_stepwise_advantage=True \
    agent.normalize_step_advantage=True \
    agent.stepwise_advantage_mode=broadcast \
    +agent_args.use_memory=False \
    +agent_args.use_solver_cot=False \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    hydra.run.dir="$RUN_DIR" \
    +trainer.save_dir="$RUN_DIR/checkpoints" \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.val_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    +env_args.solver_remote.temperature=0.0 \
    +env_args.solver_remote.max_tokens=8192 \
    +env_args.solver_remote.base_url="$VLLM_BASE_URL" \
    +env_args.solver_remote.api_key="$VLLM_API_KEY" \
    +env_args.solver_remote.model="$MODEL_PATH" \
    +env_args.reward_kwargs.remote_url="$VLLM_BASE_URL" \
    +env_args.reward_kwargs.remote_api_key="$VLLM_API_KEY" \
    +env_args.reward_kwargs.solver_model_path="$MODEL_PATH" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    +actor_rollout_ref.model.override_config.torch_dtype=bfloat16 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=60000 \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.mask_truncated_samples=True \
    algorithm.clip_advantages=False \
    trainer.critic_warmup=0 \
    trainer.logger="['wandb','console']" \
    trainer.project_name="rllm-deepcoder" \
    trainer.experiment_name="cm-deepcoder1.5b-broadcast-4steps-marlowe" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.n_training_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=100