set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export LAUNCHER=pytorch
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export DS_SKIP_CUDA_CHECK=1

GPUS=${GPUS:-6}
BATCH_SIZE=${BATCH_SIZE:-12}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229

OUTPUT_DIR='./work_dirs/janus_finetune_x2i'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  janus/train_helper/train.py \
  --model_name_or_path "deepseek-ai/Janus-1.3B" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./data/janus_train_gen.json" \
  --overwrite_output_dir True \
  --force_image_size 384 \
  --max_dynamic_patch 1 \
  --pad2square True \
  --freeze_vision True \
  --freeze_gen_vision True \
  --unfreeze_gen_aligner True \
  --unfreeze_gen_head True \
  --unfreeze_gen_embed True \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 2048 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --deepspeed "./shell/zero_stage3_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
