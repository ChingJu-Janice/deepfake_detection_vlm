import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import CLIPLoRAModel, CLIPLinearProbeModel
from src.dataset_loader import DeepfakeDataset, image_collate_fn
import argparse
from sklearn.metrics import roc_auc_score
from utils.utils import set_seed, count_trainable_params
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True
    )
    parser.add_argument(
        '--train_csv',
        type=str,
        default='pre_data/train.csv'
    )
    parser.add_argument(
        '--val_csv',
        type=str, 
        default='pre_data/val.csv'
    )
    parser.add_argument(
        '--epochs', 
        type=int,
        default=3
    )
    parser.add_argument(
        '--batch_size',
        type=int, 
        default=32
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=1e-4
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='weights/best_model.pt'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['lora', 'linear_probe'],
        default='lora',
        help='Choose model type: "lora" for LoRA version, "linear_probe" for linear probe version'
    )
    parser.add_argument(
        '--lora_rank',
        type=int,
        default=8,
        help='Dropout for LoRA layers, only used if model_type is "lora"'
    )
    parser.add_argument(
        '--lora_alpha',
        type=int,
        default=16, 
        help='Alpha for LoRA layers, only used if model_type is "lora"'
    )
    parser.add_argument(
        '--lora_dropout',
        type=float,
        default=0.1,
        help='Dropout for LoRA layers, only used if model_type is "lora"'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.model_type == 'linear_probe':
        model = CLIPLinearProbeModel()
        print("Using Linear Probe model")
    else:
        model = CLIPLoRAModel(
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        print("Using CLIP + LoRA model")
        
    model.to(device)
    # ratio = count_trainable_params(model)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    train_set = DeepfakeDataset(args.train_csv, args.dataset_root)
    val_set = DeepfakeDataset(args.val_csv, args.dataset_root)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=image_collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=image_collate_fn)

    criterion = torch.nn.BCEWithLogitsLoss()
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        #training
        model.train()
        total_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for image_batch, labels, _ in progress:
            labels = labels.to(device)
            logits = model(image_batch, device=device)  
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())
            

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}  - Avg Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for image_batch, labels, _ in tqdm(val_loader,"Val"):
                labels = labels.to(device)
                logits = model(image_batch, device=device)
                all_labels.append(labels)
                all_preds.append(torch.sigmoid(logits))
        all_labels = torch.cat(all_labels).cpu().numpy()
        all_preds = torch.cat(all_preds).cpu().numpy()
        
        
        val_auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch}  - Val AUC: {val_auc:.4f}")

        
        model.save_weights(args.save_path)
        print(f"Model weights saved to {args.save_path}")

if __name__ == "__main__":
    main()