#!/bin/bash
#SBATCH --partition=sphinx
#SBATCH --account=nlp
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=24
#SBATCH --job-name="qwen1.5b_cm_broadcast_4_steps"
#SBATCH --output=qwen1.5b_cm_broadcast_4_steps.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchoi1@stanford.edu
#SBATCH --exclude=sphinx[1-2,5,7]

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

RUN_DIR=/scr/biggest/cchoi1/rllm/runs/$(date +%m-%d-%H-%M)
mkdir -p "$RUN_DIR"

# ---- Remote vLLM endpoint on Node A ----
# Set these to your Node A host/port and API key used when launching vLLM
VLLM_HOST="jagupard35.stanford.edu"    # e.g., sphinx8.stanford.edu or 10.24.7.52
VLLM_PORT=12345
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

# Optional: quick readiness check for remote server
for i in {1..10}; do
  if curl -s -H "Authorization: " "${VLLM_BASE_URL}/models" | grep -q "${MODEL_PATH}"; then
    echo "Remote vLLM is reachable and serving ${MODEL_PATH}"
    break
  fi
  echo "Waiting for remote vLLM at ${VLLM_BASE_URL} ... (${i}/10)"
  sleep 10
  if [[ $i -eq 10 ]]; then
    echo "ERROR: Could not reach remote vLLM at ${VLLM_BASE_URL} or model not listed."
    exit 1
  fi
done

NUM_GPUS=2

python3 -m examples.context_manager.train_cm \
    agent.max_steps=4 \
    agent.use_stepwise_advantage=True \
    agent.normalize_step_advantage=True \
    agent.stepwise_advantage_mode=broadcast \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    +agent_args.use_memory=False \
    +agent_args.use_solver_cot=False \
    hydra.run.dir="$RUN_DIR" \
    +trainer.save_dir="$RUN_DIR/checkpoints" \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    data.val_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    +env_args.solver_remote.temperature=0.2 \
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
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=11000 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.mask_truncated_samples=True \
    algorithm.clip_advantages=False \
    trainer.critic_warmup=0 \
    trainer.logger="[wandb,console]" \
    trainer.project_name="rllm-deepcoder" \
    trainer.experiment_name="cm-qwen1.5b-coder-broadcast-4k-4steps" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.n_training_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=100
