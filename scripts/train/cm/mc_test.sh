#!/usr/bin/env bash
# set -euo pipefail

unset ROCR_VISIBLE_DEVICES
unset ROCM_VISIBLE_DEVICES    # (sometimes set on some clusters)
unset HIP_VISIBLE_DEVICES     # (if present)

# --- vLLM / torch env
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HYDRA_FULL_ERROR=1
export RAY_DISABLE_DASHBOARD=1
export CUDA_VISIBLE_DEVICES=0,1

# clean any previous instances and stray shared dirs
ray stop -f || true
pkill -9 -f "ray::" || true
rm -rf "/tmp/$USER"/ray_* 2>/dev/null || true

# export RAY_TMPDIR="/scr/biggest/cchoi1/ray"
# export TMPDIR="/scr/biggest/cchoi1/tmp"
# mkdir -p "$RAY_TMPDIR" "$TMPDIR"
# chmod 700 "$RAY_TMPDIR" "$TMPDIR"
export RAY_object_store_allow_fallback_to_memory=1

# ------------------------------
# W&B: persist and resume same run
# ------------------------------

RUN_DIR="/nlp/scr/cchoi1/rllm/runs/cm-deepcoder1.5b-mc-4steps-debug"
mkdir -p "$RUN_DIR"
export WANDB_PROJECT="rllm-deepcoder"
export WANDB_NAME="cm-deepcoder1.5b-mc-4steps-debug-timing"
export WANDB_DIR="$RUN_DIR"

# WANDB_RUN_ID="naypfpto"
# WANDB_RESUME=must

WANDB_ID_FILE="$RUN_DIR/wandb_run_id.txt"

if [[ -s "$WANDB_ID_FILE" ]]; then
  export WANDB_RUN_ID="$(cat "$WANDB_ID_FILE")"
  export WANDB_RESUME=must
else
  unset WANDB_RUN_ID
  export WANDB_RESUME=allow
fi

# After wandb.init runs, ensure the id is persisted for future restarts:
# (If your trainer doesn’t write the file itself, do a best-effort scrape after a few seconds)
# You can also leave this to your code if it writes the id out.
( sleep 10
  if [[ ! -s "$WANDB_ID_FILE" && -d "$WANDB_DIR/wandb" ]]; then
    last_run="$(ls -dt "$WANDB_DIR"/wandb/run-* 2>/dev/null | head -1 || true)"
    if [[ -n "$last_run" ]]; then
      echo "$(basename "$last_run" | awk -F- '{print $NF}')" > "$WANDB_ID_FILE"
    fi
  fi
) >/dev/null 2>&1 &

# ------------------------------
# Config
# ------------------------------
MODEL_PATH="agentica-org/DeepCoder-1.5B-Preview"

# ---- Remote vLLM endpoint on Node A ----
# Set these to your Node A host/port and API key used when launching vLLM
VLLM_HOST="tiger7.stanford.edu"
VLLM_PORT=12345
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

echo "=== vLLM Solver Configuration ==="
echo "Host: $VLLM_HOST"
echo "Port: $VLLM_PORT"
echo "Base URL: $VLLM_BASE_URL"
echo "Model: $MODEL_PATH"
echo "=================================="

# Optional: quick readiness check for remote server
for i in {1..10}; do
  if curl -sS "${VLLM_BASE_URL}/models" | grep -q "${MODEL_PATH}"; then
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

curl -sS -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_PATH\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}" \
  "${VLLM_BASE_URL}/chat/completions"

NUM_GPUS=2

python3 -m examples.context_manager.debug_train_cm \
    rllm.agent.max_steps=4 \
    rllm.stepwise_advantage.enable=True \
    rllm.stepwise_advantage.normalize_by_steps=True \
    rllm.stepwise_advantage.mode=per_step \
    ++rllm.agent.agent_args.use_memory=False \
    ++rllm.agent.agent_args.use_solver_cot=False \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    hydra.run.dir="$RUN_DIR" \
    +trainer.save_dir="$RUN_DIR/checkpoints" \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=1 \
    data.val_batch_size=16 \
    data.max_prompt_length=16384 \
    data.max_response_length=8192 \
    ++rllm.env.env_args.solver_remote.temperature=0.0 \
    ++rllm.env.env_args.solver_remote.max_tokens=16384 \
    ++rllm.env.env_args.solver_remote.base_url="$VLLM_BASE_URL" \
    ++rllm.env.env_args.solver_remote.model="$MODEL_PATH" \
    ++rllm.env.env_args.reward_kwargs.remote_url="$VLLM_BASE_URL" \
    ++rllm.env.env_args.reward_kwargs.solver_model_path="$MODEL_PATH" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    +actor_rollout_ref.model.override_config.torch_dtype=float16 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30000 \
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
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=True \
    trainer.critic_warmup=0 \
    trainer.logger="['wandb','console']" \
    trainer.project_name="rllm-deepcoder" \
    trainer.experiment_name=$WANDB_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=100