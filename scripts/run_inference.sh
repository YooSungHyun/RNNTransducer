#!/bin/bash
PL_DATA_DIR=""

python3 ../inference.py \
    --pl_data_dir=$PL_DATA_DIR \
    --model_config="" \
    --model_dir="" \
    --vocab_path="" \
    --precision=16 \
    --val_on_cpu=true
