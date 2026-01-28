# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export GPU_COUNT=1
export DATA_DIR='data/nq_search'
# export DATA_DIR='data/nq_hotpotqa_train'
# export DATA_DIR='data/nq_search_1000'
# export DATA_DIR='data/nq_search_512'
# export DATA_DIR='data/nq_search_5000'
export TRAIN_DATA_DIR=$DATA_DIR
export TEST_DATA_DIR=$DATA_DIR
export RETRIEVER_URL='http://0.0.0.0:8000/retrieve'
WAND_PROJECT='Search-R1'


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export BASE_MODEL='meta-llama/Llama-3.2-3B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.2-3b-em
# export BASE_MODEL='meta-llama/Llama-3.2-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.2-3b-it-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.1-8b-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.1-8b-it-em

export BASE_MODEL='Qwen/Qwen2.5-3B'
export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-3b-em

export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-3b-it-em
# export EXPERIMENT_NAME=nq-1000-search-r1-grpo-qwen2.5-3b-it-em
# export EXPERIMENT_NAME=nq-512-search-r1-grpo-qwen2.5-3b-it-em
# export EXPERIMENT_NAME=nq-5000-search-r1-grpo-qwen2.5-3b-it-em
# export EXPERIMENT_NAME=nq-5000-search-r1-grpo-qwen2.5-3b-it-em-bm25

# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-em
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-it-em

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

if [ "$GPU_COUNT" -eq 1 ] || [ "$GPU_COUNT" -eq 2 ]; then
  # Reduced settings for 2 GPUs to avoid OOM
  BATCH_SIZE=256          
  VAL_BATCH_SIZE=128      
  PPO_MINI_BATCH_SIZE=128 
  PPO_MICRO_BATCH_SIZE=32
elif [ "$GPU_COUNT" -eq 4 ] || [ "$GPU_COUNT" -eq 8 ]; then
  # original settings
  BATCH_SIZE=512
  VAL_BATCH_SIZE=256
  PPO_MINI_BATCH_SIZE=256
  PPO_MICRO_BATCH_SIZE=64
elif [ "$GPU_COUNT" -eq 6 ]; then
  # adjusted for divisibility by 6 (510*5=2550 total trajectories)
  BATCH_SIZE=510
  VAL_BATCH_SIZE=255
  PPO_MINI_BATCH_SIZE=1275   # (510 * n_agent) / 2 = (510*5)/2
  PPO_MICRO_BATCH_SIZE=64
else
  echo "Error: GPU_COUNT must be 2, 4, 6, or 8 (got $GPU_COUNT)" >&2
  exit 1
fi





# Clean up any existing Ray processes first
echo "Cleaning up existing Ray processes..."
ray stop --force 2>/dev/null || true
pkill -f ray 2>/dev/null || true
pkill -f raylet 2>/dev/null || true
sleep 3

# Start Ray cluster
echo "Starting Ray cluster..."
ray start --head --num-cpus=64 --num-gpus=$GPU_COUNT --object-store-memory=10000000000

# Wait a moment for Ray to fully initialize
sleep 5

# Set Ray environment variables
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_TMPDIR=/tmp/ray_tmp_$(whoami)
mkdir -p $RAY_TMPDIR
export RAY_memory_monitor_refresh_ms=1000
export RAY_object_spilling_threshold=0.8

# Set RAY_ADDRESS to connect to the cluster we just started
export RAY_ADDRESS="auto"





PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_DATA_DIR/train.parquet \
    data.val_files=$TEST_DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$GPU_COUNT \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=10000 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=2 \
    retriever.url=$RETRIEVER_URL \
    retriever.topk=3 \
    2>&1 | tee ${EXPERIMENT_NAME}.log