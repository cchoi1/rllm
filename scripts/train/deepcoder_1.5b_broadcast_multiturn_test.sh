export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HYDRA_FULL_ERROR=1

# --- scratch; isolate by user+pid or Slurm job id
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID:-$$}"
export TMPDIR="/tmp/tmp_${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

# --- guarantee we don't attach to someone else's Ray
unset RAY_ADDRESS RAY_NAMESPACE RAY_RUNTIME_ENV_DIR RAY_HEAD_NODE
export RAY_NAMESPACE="cm-${USER}-$$"

# clean any previous instances and stray shared dirs
ray stop -f || true
rm -rf /tmp/ray /tmp/*/ray 2>/dev/null || true

echo "RAY_TMPDIR=$RAY_TMPDIR  RAY_ADDRESS=${RAY_ADDRESS:-<unset>}  RAY_NAMESPACE=$RAY_NAMESPACE"

RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")

MODEL_PATH="agentica-org/DeepCoder-1.5B-Preview"
NUM_GPUS=2

python3 -m examples.deepcoder.train_deepcoder_multiturn \
    agent.max_steps=4 \
    agent.use_stepwise_advantage=True \
    agent.normalize_step_advantage=True \
    agent.stepwise_advantage_mode=broadcast \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    hydra.run.dir="$RUN_DIR" \
    +trainer.save_dir="$RUN_DIR/checkpoints" \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.val_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    +actor_rollout_ref.model.override_config.torch_dtype=bfloat16 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=19000 \
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
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.mask_truncated_samples=True \
    algorithm.clip_advantages=False \
    trainer.critic_warmup=0 \
    trainer.logger="['wandb','console']" \
    trainer.project_name="rllm-deepcoder" \
    trainer.experiment_name="multiturn-deepcoder1.5b-broadcast-4steps" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.n_training_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=100