# deepfake_detection_vlm

## Dataset
請先下載經過預處理的資料集： [FaceForensics++ C40](https://www.dropbox.com/t/2Amyu4D5TulaIofv)，建議將資料集下載後放置於合適資料夾，並設為 `$DATA_ROOT` 環境變數使用。

## Setup
使用 Conda 環境：
```bash
conda create -n dfvlm python= 3.10
conda activate dfvlm
pip install -r requirements.txt
```

## Quick Start
直接使用以下指令快速開始：
```
bash run.sh
```

## Training Command
自訂訓練參數範例如下：
```
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
```

## Inference Command
模型推論指令如下，請記得指定訓練後的權重檔：
```
python inference.py \
    --dataset_root "$DATA_ROOT" \
    --seed "$SEED" \
    --model_type 'lora' \
    --checkpoint './weights/best_model.pt' \
    --lora_rank "$RANK" \
    --lora_alpha "$ALPHA" \
    --lora_dropout "$DROPOUT"

```

## Pretrained Weights
[預訓練權重檔](https://github.com/ChingJu-Janice/deepfake_detection_vlm/releases)

## Results
以下是不同模型在測試資料上的效能表現：

| Model                    | ACC    | F1     | EER   | AUC   | Run Time |
|--------------------------|--------|--------|-------|-------|----------|
| frozen CLIP linear probe | 0.8182 | 0.898  | 0.38  | 0.703 | ~ 2 min  |
| CLIP + LoRA              | 0.8909 | 0.9394 | 0.185 | 0.886 | ~ 2 min  |

* ACC: 準確率（Accuracy）
* F1: 衡量精確率與召回率的平衡
* EER: Equal Error Rate，錯誤平衡率越低越好
* AUC: ROC 曲線下面積，越接近 1 越佳
* Run Time: 推論時間（約 2 分鐘）

ROC curve 在 results/video_level/video_lora_roc.png

video level score 在 results/video_level/ video_lora_metrics.json

