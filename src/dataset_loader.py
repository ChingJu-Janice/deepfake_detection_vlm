import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class DeepfakeDataset(Dataset):
    """
    自訂 Dataset 類別：
    回傳 (PIL.Image, label:int, 相對路徑:str)
    """
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file, header=None, names=["rel_path", "label"])
        self.root = Path(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data.iloc[idx]
        img_path = self.root / entry["rel_path"]
        image = Image.open(img_path).convert("RGB")
        label = int(entry["label"])
        return image, label, str(entry["rel_path"])


def image_collate_fn(batch):
    """
    批次整理函數：回傳 image list, label tensor, 路徑 list
    """
    imgs, labels, paths = zip(*batch)
    return list(imgs), torch.tensor(labels, dtype=torch.float32), list(paths)


def test_dataset_loading(dataset_root, pre_data_path, num_samples=5):
    dataset = DeepfakeDataset(pre_data_path, dataset_root)
    loader = DataLoader(dataset, batch_size=num_samples, collate_fn=image_collate_fn)

    imgs, labels, paths = next(iter(loader))
    print(f"Successfully loaded {len(dataset)} samples. Showing the first {num_samples} samples:")
    for i in range(num_samples):
        print(f"  [{i}] label={int(labels[i])} | path={paths[i]} | size={imgs[i].size}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument("--pre_data_path", type=str, default="pre_data/train.csv")
    args = parser.parse_args()

    test_dataset_loading(args.dataset_root, args.pre_data_path)
