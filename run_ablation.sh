#!/bin/bash

DATA_ROOT="/home/janice/Documents/datasets"
SEED=42
RESULT_DIR="./results/ablation"

mkdir -p "$RESULT_DIR"


for rank in 8 16 32; do
  for alpha in 16 32 64; do
    for dropout in 0.0 0.05 0.1; do
      EXP_ID="rank${rank}_alpha${alpha}_dropout${dropout}"
      LOG_FILE="${RESULT_DIR}/${EXP_ID}.json"

      echo "Running ${EXP_ID}..."

      python training.py \
        --dataset_root "$DATA_ROOT" \
        --epochs 1 \
        --batch_size 32 \
        --model_type 'lora' \
        --seed "$SEED" \
        --lora_rank "$rank" \
        --lora_alpha "$alpha" \
        --lora_dropout "$dropout"

      python inference.py \
        --dataset_root "$DATA_ROOT" \
        --video_level \
        --model_type 'lora' \
        --seed "$SEED" \
        --lora_rank "$rank" \
        --lora_alpha "$alpha" \
        --lora_dropout "$dropout" \
        --ablation_id  "$LOG_FILE"

    done
  done
done





# for dropout in 0.0 0.05 0.1; do
#     EXP_ID="dropout${dropout}"
#     LOG_FILE="${RESULT_DIR}/${EXP_ID}.json"

#     echo "Running ${EXP_ID}..."

#     python training.py \
#     --dataset_root "$DATA_ROOT" \
#     --epochs 1 \
#     --batch_size 32 \
#     --model_type 'lora' \
#     --seed "$SEED" \
#     --lora_dropout "$dropout" \

#     python inference.py \
#     --dataset_root "$DATA_ROOT" \
#     --video_level \
#     --model_type 'lora' \
#     --seed "$SEED" \
#     --lora_dropout "$dropout" \
#     --ablation_id "$LOG_FILE"
# done
