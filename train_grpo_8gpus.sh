export VLLM_USE_V1=0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PET_NODE_RANK=0
export NCCL_BLOCKING_WAIT=1  
export NCCL_IB_DISABLE=0  
export NCCL_SOCKET_IFNAME=eth0  
# Use Ray via the active Python interpreter so a stale shebang does not break the run.
python3 -m ray.scripts.scripts start --head || true


export BASE_MODEL='/mnt/finder/qyj/models/Qwen2.5-7B-Instruct'
WAND_PROJECT='EviNoteRAG'
EXPERIMENT_NAME='3_all_process_reward'

# upstream: original Da1yuqin/EviNoteRAG reward; custom: role-aware process reward.
export EVINOTE_REWARD_MODE="${EVINOTE_REWARD_MODE:-custom}"

# Resume actor weights from a previous checkpoint.
# BASE_MODEL must remain the original model because it is used as the ref policy.
# Old checkpoints named global_step_N are supported even if training_state.json is missing.
RESUME_FROM=''

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues
DATE=$(date '+%Y-%m-%d-%H-%M-%S')
OUTPUT_DIR="$(pwd)/outputs/${WAND_PROJECT}/${EXPERIMENT_NAME}"
mkdir -p ${OUTPUT_DIR}
export TRAIN_TRAJECTORY_LOG_FILE="${OUTPUT_DIR}/${DATE}_train_trajectories.jsonl"
export EVAL_TRAJECTORY_LOG_FILE="${OUTPUT_DIR}/${DATE}_eval_trajectories.jsonl"

echo "EVINOTE_REWARD_MODE=${EVINOTE_REWARD_MODE}"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=./data_preprocess/data/m_train_dotraining3.parquet \
    data.val_files=./data_preprocess/data/m_test_dotraining3.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=294 \
    data.val_batch_size=294 \
    data.max_prompt_length=16384 \
    data.max_response_length=1024 \
    data.max_start_length=2048 \
    data.max_obs_length=8192 \
    data.shuffle_train_dataloader=false \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.078 \
    actor_rollout_ref.actor.ppo_mini_batch_size=147 \
    actor_rollout_ref.actor.ppo_micro_batch_size=21 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=60 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=120 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    +trainer.val_only=false \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=false \
    +trainer.resume_from="$RESUME_FROM" \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=7 \
    trainer.nnodes=1 \
    +trainer.use_amp=True \
    +trainer.amp_dtype="bfloat16" \
    trainer.save_freq=15 \
    trainer.test_freq=30 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/mnt/finder/qyj/models/$WAND_PROJECT/$EXPERIMENT_NAME \
    max_turns=4 \
    actor_rollout_ref.rollout.n_agent=5 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    retriever.decompose_claims=true \
    2>&1 | tee ${OUTPUT_DIR}/${DATE}.log 