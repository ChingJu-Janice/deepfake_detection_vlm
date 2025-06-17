import argparse
import torch
from torch.utils.data import DataLoader
from src.dataset_loader import DeepfakeDataset, image_collate_fn
from models import CLIPLoRAModel, CLIPLinearProbeModel
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils.save_scores import save_frame_scores, save_video_scores
from utils.eval import get_metrics
from utils.utils import set_seed, count_trainable_params



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True
    )
    parser.add_argument(
        '--test_csv',
        type=str,
        default='pre_data/test.csv'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default = 'weights/best_model.pt',
        help='Path to the model checkpoint'
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        default='results',
        help='Path to save the scores per frame'
    )
    parser.add_argument(
        "--video_level", 
        action="store_true",
        help="video level evaluation, default is frame level"
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['lora', 'linear_probe'],
        default='lora',
        help='Choose model type: "lora" for LoRA version, "linear_probe" for linear probe version'
    )
    parser.add_argument(
        '--batch_size',
        type=int, 
        default=32
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42
    )
    parser.add_argument(
        '--lora_rank',
        type=int,
        default=8,
        help='Rank for LoRA layers, only used if model_type is "lora"'
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
    parser.add_argument(
        '--ablation_id',
        type = str,
        default = None,
        help='Ablation ID for tracking experiments'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"  # model =CLIPLoRADetector(lora_rank=args.lora_rank,lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout).to(device)
    
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
    
    model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)
    model.eval()

    test_set = DeepfakeDataset(args.test_csv, args.dataset_root)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=image_collate_fn)

    all_labels = []
    all_preds = []
    all_paths = []

    with torch.no_grad():
        for image_batch, labels, path in tqdm(test_loader, desc="Testing"):
            labels = labels.to(device)
            logits = model(image_batch, device=device)
            all_labels.append(labels)
            all_preds.append(torch.sigmoid(logits))
            all_paths.append(path)

    all_labels = torch.cat(all_labels).cpu().numpy()
    all_preds = torch.cat(all_preds).cpu().numpy()
    flat_paths = sum(all_paths, [])  
    
    save_frame_scores(flat_paths, all_labels, all_preds, args.result_dir)
    save_video_scores(flat_paths, all_labels, all_preds, args.result_dir)
    get_metrics(args, ratio)
    
    print("Inference completed and results saved.")
    

if __name__ == "__main__":
    main()
