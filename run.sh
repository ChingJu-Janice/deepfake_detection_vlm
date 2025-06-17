#!/usr/bin/env bash
set -e  

DATA_ROOT="/home/janice/Documents/datasets"
SEED=42
RANK=8
ALPHA=64
DROPOUT=0.05

# split_dataset 
# python utils/split_dataset.py --dataset_root "$DATA_ROOT"

# train
python training.py \
    --dataset_root "$DATA_ROOT" \
    --epochs 1 \
    --seed 42 \
    --batch_size 32 \
    --lr 0.0001 \
    --model_type 'lora' \
    --save_path './weights/best_model.pth' \
    --lora_rank "$RANK" \
    --lora_alpha "$ALPHA" \
    --lora_dropout "$DROPOUT" 
    

# test
python inference.py \
    --dataset_root "$DATA_ROOT" \
    --seed "$SEED" \
    --model_type 'lora' \
    --checkpoint './weights/best_model.pt' \
    --lora_rank "$RANK" \
    --lora_alpha "$ALPHA" \
    --lora_dropout "$DROPOUT"


# ablation_study
# python utils/ablation_result.py \
