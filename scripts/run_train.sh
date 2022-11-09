#!/bin/bash
GPU_IDS=0,1,2
HF_DATA_DIRS="
/data2/bart/temp_workspace/stt/aihub_datasets_arrow/fine-tuning/42maru/data-KsponSpeech-42maru-not-normal-20
"
PL_DATA_DIR="/data2/bart/temp_workspace/stt/aihub_datasets_arrow/fine-tuning/42maru/logmelspect-KsponSpeech-42maru-normal-20"

export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=$GPU_IDS \
torchrun --standalone --nnodes=1 --nproc_per_node=3 /data2/bart/temp_workspace/stt/RNNTransducer/train.py \
    --hf_data_dirs $HF_DATA_DIRS \
    --pl_data_dir=$PL_DATA_DIR \
    --num_shards=20 \
    --model_config="/data2/bart/temp_workspace/stt/RNNTransducer/config.json" \
    --seed=42 \
    --check_val_every_n_epoch=1 \
    --accumulate_grad_batches=1 \
    --max_epochs=100 \
    --log_every_n_steps=50 \
    --accelerator=gpu \
    --strategy=ddp \
    --devices=3 \
    --auto_scale_batch_size=false \
    --move_metrics_to_cpu=true \
    --learning_rate=0.00001 \
    --max_lr=0.001 \
    --weight_decay=0.0001
