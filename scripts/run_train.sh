#!/bin/bash
GPU_IDS=
HF_DATA_DIRS="
"
PL_DATA_DIR=""

OMP_NUM_THREADS=8 \
CUDA_VISIBLE_DEVICES=$GPU_IDS \
torchrun --standalone --nnodes=1 --nproc_per_node= ../train.py \
    --pl_data_dir=$PL_DATA_DIR \
    --num_shards=20 \
    --model_config="" \
    --vocab_path="" \
    --output_dir="" \
    --seed=42 \
    --num_proc= \
    --per_device_train_batch_size=1 \
    --train_batch_drop_last=false \
    --per_device_eval_batch_size=1 \
    --eval_batch_drop_last=false \
    --val_check_interval=0.1 \
    --accumulate_grad_batches=16 \
    --max_epochs=100 \
    --log_every_n_steps=100 \
    --accelerator=gpu \
    --strategy=ddp \
    --devices= \
    --auto_select_gpus=true \
    --auto_scale_batch_size=false \
    --learning_rate= \
    --max_lr= \
    --precision=16 \
    --weight_decay= \
    --warmup_ratio= \
    --final_div_factor=
