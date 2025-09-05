#!/bin/bash
#SBATCH --partition=sphinx
#SBATCH --account=nlp
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=512GB
#SBATCH --cpus-per-task=64
#SBATCH --job-name="deepcoder_1.5b_short"
#SBATCH --output=deepcoder_1.5b_short.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchoi1@stanford.edu

# Load conda environment
source /nlp/scr/cchoi1/miniconda3/etc/profile.d/conda.sh
conda activate rllm
cd /nlp/scr/cchoi1/rllm

set -a
. /nlp/scr/cchoi1/LiveCodeBench/.env
set +a

set -x

ray stop -f || true
rm -rf /tmp/ray/* 2>/dev/null || true

# Print GPU info
srun -l bash -c 'echo "Node: $(hostname -s)"; nvidia-smi -L'

# Remove ulimit command that requires root privileges
# ulimit -n 1048576 
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000

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

NUM_GPUS=4
MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B-Instruct"

RUN_DIR=/scr/biggest/cchoi1/rllm/runs/$(date +%m-%d-%H-%M)
echo "RUN_DIR: $RUN_DIR"
mkdir -p $RUN_DIR

python3 -m examples.deepcoder.train_deepcoder \
    hydra.run.dir=$RUN_DIR \
    +trainer.save_dir='$RUN_DIR/checkpoints' \
    ++trainer.resume_mode=resume_path \
    ++trainer.resume_from_path=/nlp/scr/cchoi1/rllm/checkpoints/rllm-deepcoder/deepcoder-1.5b-short/global_step_90 \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=256 \
    data.val_batch_size=512 \
    data.max_prompt_length=7000 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10000 \
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
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.mask_truncated_samples=True \
    algorithm.clip_advantages=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-deepcoder' \
    trainer.experiment_name='deepcoder-1.5b-short' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    agent.max_steps=1 \
    agent.use_stepwise_advantage=False \
    trainer.total_epochs=100 