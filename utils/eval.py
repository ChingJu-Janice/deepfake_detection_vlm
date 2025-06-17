import numpy as np
import pandas as pd
import argparse
import json
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
import matplotlib.pyplot as plt
import os


def parse_args():
    parser = argparse.ArgumentParser()
 
    parser.add_argument(
        '--result_dir', 
        type=str, default='results', 
        help='Directory to save the evaluation results'
    )
    
    parser.add_argument(
        "--video_level", 
        action="store_true",
        help="video level evaluation, default is frame level"
    )
    return parser.parse_args()
    

def compute_eer(fpr, tpr, thresholds):
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    idx_eer = np.nanargmin(abs_diffs)
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2
    threshold = thresholds[idx_eer]
    return eer, threshold

def evaluate_metrics(labels, logits):
    auc = roc_auc_score(labels, logits)
    fpr, tpr, thresholds = roc_curve(labels, logits)
    eer, eer_threshold = compute_eer(fpr, tpr, thresholds)
    preds = (logits >= 0.5).astype(int)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        "AUC": round(auc, 4),
        "EER": round(eer, 4),
        "F1": round(f1, 4),
        "Accuracy": round(acc, 4),
    }

def save_roc_curve(fpr, tpr, auc,output_path):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()

def get_metrics(args,ratio=None): 
    level_type = "video" if args.video_level else "frame"
    
    df = pd.read_csv(f"{args.result_dir}/{level_type}_level/{level_type}_lora_score.csv")
    labels = df['label'].values
    logits = df['score'].values

    metrics = evaluate_metrics(labels, logits)
    
    fpr, tpr, _ = roc_curve(labels, logits)
    save_roc_curve(
        fpr, 
        tpr,
        metrics["AUC"],
        output_path=f'{args.result_dir}/{level_type}_level/{level_type}_lora_roc.png')
    
    
    with open(f'{args.result_dir}/{level_type}_level/{level_type}_lora_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"saved metrics and roc curve to {args.result_dir}")
    
    if args.ablation_id:
        metrics.update({
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "parameter_count": ratio,
        })
        os.makedirs(os.path.dirname(args.ablation_id), exist_ok=True)
        with open(args.ablation_id, 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    args = parse_args()
    get_metrics(args)