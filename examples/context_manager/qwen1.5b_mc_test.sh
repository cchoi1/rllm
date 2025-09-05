#!/bin/bash
set -x

# --- env
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export OPENAI_API_KEY="$TOGETHER_API_KEY"
export OPENAI_BASE_URL="https://api.together.xyz/v1"    # used by many OpenAI clients
# (Some libs also read OPENAI_API_BASE; harmless to set both)
export OPENAI_API_BASE="$OPENAI_BASE_URL"

# node-local scratch for Ray
export RAY_TMPDIR="/tmp/$USER/ray"
export TMPDIR="/tmp/$USER/tmp"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"
ray stop -f || true
rm -rf /tmp/ray /tmp/*/ray 2>/dev/null || true

# --- model / run dir
RLLM_DIR=$(python3 -c "import rllm,os;print(os.path.dirname(os.path.dirname(rllm.__file__)))")
MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B-Instruct"
RUN_DIR="/scr/cchoi1/rllm/runs/$(date +%m-%d-%H-%M)"
mkdir -p "$RUN_DIR"

NUM_GPUS=1

python3 -m examples.context_manager.train_cm \
    agent.max_steps=2 \
    agent.use_stepwise_advantage=True \
    agent.normalize_step_advantage=True \
    agent.stepwise_advantage_mode=mc_return \
    algorithm.gamma=0.95 \
    algorithm.lam=0.95 \
    hydra.run.dir="$RUN_DIR" \
    +rewards.context_assist.solver_args.remote_url="${OPENAI_BASE_URL}" \
    +rewards.context_assist.solver_args.remote_api_key="${OPENAI_API_KEY}" \
    +trainer.save_dir="$RUN_DIR/checkpoints" \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=16 \
    data.val_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=13000 \
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
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.mask_truncated_samples=True \
    algorithm.clip_advantages=False \
    trainer.critic_warmup=0 \
    trainer.logger="['wandb','console']" \
    trainer.project_name="rllm-deepcoder" \
    trainer.experiment_name="cm-qwen1.5b-coder-mc-4k-2steps-debug" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.n_training_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=100