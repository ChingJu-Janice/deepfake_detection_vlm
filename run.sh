#!/usr/bin/env bash
set -e  

DATA_ROOT=$1
# python utils/split_dataset.py --dataset_root "$DATA_ROOT"

# check dataset loaded
# python src/dataset_loader.py --dataset_root "$DATA_ROOT"

# train
python training.py \
    --dataset_root "$DATA_ROOT" \
    --epochs 1 \
    # --train_csv pre_data/train.csv \
    # --batch_size 64 \
    
    # --learning_rate 0.001 \


