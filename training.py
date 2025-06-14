import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
# from utils.seed import seed_everything
from src.model import CLIPLoRADetector
from src.dataset_loader import DeepfakeDataset, image_collate_fn
import argparse
from sklearn.metrics import roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--train_csv', type=str, default='pre_data/train.csv')
    parser.add_argument('--val_csv', type=str, default='pre_data/val.csv')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='best_model.pt')
    return parser.parse_args()

def evaluate(model, val_loader, device):
    model.eval()
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for image_batch, labels, _ in val_loader:
            labels = labels.to(device)
            logits = model(image_batch, device=device)
            all_labels.append(labels)
            all_logits.append(torch.sigmoid(logits))
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_logits = torch.cat(all_logits).cpu().numpy()
    auc = roc_auc_score(all_labels, all_logits)
    return auc

def main():
    args = parse_args()
    # seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPLoRADetector().to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    train_set = DeepfakeDataset(args.train_csv, args.dataset_root)
    val_set = DeepfakeDataset(args.val_csv, args.dataset_root)
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=4, collate_fn=image_collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=4, collate_fn=image_collate_fn)

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
        val_auc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}  - Val AUC: {val_auc:.4f}")


        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), args.save_path)
            print(f"New best AUC: {best_auc:.4f}, model saved to {args.save_path}")

if __name__ == "__main__":
    main()