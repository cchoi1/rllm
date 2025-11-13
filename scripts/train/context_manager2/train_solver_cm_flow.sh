#!/bin/bash
#SBATCH --partition=sphinx
#SBATCH --account=nlp
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=512GB
#SBATCH --cpus-per-task=100
#SBATCH --job-name="deepcoder_solver_cm_flow"
#SBATCH --output=deepcoder_solver_cm_flow.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cchoi1@stanford.edu

# Load conda environment
source /nlp/scr/cchoi1/miniconda3/etc/profile.d/conda.sh
conda activate rllm2
cd /nlp/scr/cchoi1/rllm

set -a
. /nlp/scr/cchoi1/LiveCodeBench/.env
set +a

set -x

# Print GPU info
srun -l bash -c 'echo "Node: $(hostname -s)"; nvidia-smi -L'

# --- vLLM / torch env
unset ROCR_VISIBLE_DEVICES ROCM_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HYDRA_FULL_ERROR=1
export RAY_DISABLE_DASHBOARD=1

# Clean any previous instances and stray shared dirs
ray stop -f || true
pkill -9 -f "ray::" || true
rm -rf "/tmp/$USER"/ray_* 2>/dev/null || true

# Recreate isolated dirs AFTER cleanup
export RAY_TMPDIR="/scr/cchoi1/ray"
export TMPDIR="/scr/cchoi1/tmp"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"
chmod 700 "$RAY_TMPDIR" "$TMPDIR"
export RAY_object_store_allow_fallback_to_memory=1

RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
RUN_NAME="deepcoder-solver-cm-flow-4gpu"
RUN_DIR=/scr/cchoi1/rllm/runs/$RUN_NAME
mkdir -p "$RUN_DIR"

# ------------------------------
# W&B: persist and resume same run
# ------------------------------
export WANDB_PROJECT="rllm-deepcoder"
export WANDB_NAME=$RUN_NAME
export WANDB_DIR=$RUN_DIR
export WANDB_RUN_ID="pl4gpu"
export WANDB_RESUME=must

# ------------------------------
# Config
# ------------------------------
MODEL_PATH="agentica-org/DeepCoder-1.5B-Preview"
NUM_GPUS=4

# Start vLLM server in background for solver (if using remote solver)
# Note: For training, you might want to use a separate solver server or local solver
# Uncomment below if you need a local vLLM server for the solver
# echo "Starting vLLM server for solver on port 30000..."
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$MODEL_PATH" \
#     --host 0.0.0.0 \
#     --port 30000 \
#     --served-model-name "$MODEL_PATH" \
#     --data-parallel-size 4 \
#     --dtype bfloat16 \
#     --gpu-memory-utilization 0.85 \
#     --max-model-len 16384 &
# 
# VLLM_PID=$!
# 
# # Wait for server to be ready
# echo "Waiting for vLLM server to start..."
# for i in {1..60}; do
#     if curl -sS "http://localhost:30000/v1/models" | grep -q "$MODEL_PATH"; then
#         echo "vLLM server is ready!"
#         break
#     fi
#     echo "Waiting for vLLM server... (${i}/60)"
#     sleep 5
#     if [[ $i -eq 60 ]]; then
#         echo "ERROR: vLLM server did not start within 5 minutes"
#         kill $VLLM_PID || true
#         exit 1
#     fi
# done

# Run training
python3 -m examples.context_manager2.train_solver_cm_flow \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.val_batch_size=500 \
    data.max_prompt_length=4096 \
    data.max_response_length=16384 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-deepcoder' \
    trainer.experiment_name="$RUN_NAME" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    rllm.stepwise_advantage.enable=False \
    trainer.total_epochs=100 \
    rllm.workflow.use_workflow=True \
    rllm.workflow.n_parallel_tasks=128 \
    hydra.run.dir="$RUN_DIR" \
    +trainer.save_dir="$RUN_DIR/checkpoints"

# Cleanup
echo "Cleaning up..."
ray stop -f || true
pkill -9 -f "ray::" || true
# Uncomment if using local vLLM server
# kill $VLLM_PID || true
# pkill -9 -f "vllm.serve" || true

LOGDIR=$(ls -td /tmp/$USER/ray_*/session_*/logs 2>/dev/null | head -n1 || echo "")
if [ -n "$LOGDIR" ]; then
    grep -nE "Traceback|ImportError|ModuleNotFoundError|OOM|CUDA" "$LOGDIR"/worker*.log 2>/dev/null | tail -n 60 || true
    grep -nE "Traceback|ImportError|ModuleNotFoundError" "$LOGDIR"/python-core-worker*.log 2>/dev/null | tail -n 60 || true
fi

echo "Job completed!"

