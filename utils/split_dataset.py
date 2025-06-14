import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import random

def collect_image_paths(data_dir, label, dataset_root):
    data = []
    for video_dir in Path(data_dir).iterdir():  
        if not video_dir.is_dir():
            continue
        video_id = video_dir.name
        images = [p for p in video_dir.glob("*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]  
        for img in images:
            rel_path = img.relative_to(dataset_root)
            data.append({"video": video_id, "path": str(rel_path), "label": label})
    return pd.DataFrame(data)

def save_csv(df, path):
    df[['path', 'label']].to_csv(path, index=False, header=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--seed', type=int, default=13, help='Random seed for splitting')
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    random.seed(args.seed)
    output_dir = Path("./pre_data")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Collect all image paths
    real_df = collect_image_paths(dataset_root / "Real_youtube", label=0, dataset_root=dataset_root)
    faceswap_df = collect_image_paths(dataset_root / "FaceSwap", label=1, dataset_root=dataset_root)
    neuraltextures_df   = collect_image_paths(dataset_root / "NeuralTextures", label=1, dataset_root=dataset_root)

    # Step 2: Split video-wise
    real_vids = real_df['video'].unique()
    real_train_v, real_temp_v = train_test_split(real_vids, test_size=0.3, random_state=args.seed)  # 70% train
    real_val_v, real_test_v   = train_test_split(real_temp_v, test_size=1/3, random_state=args.seed)  # 20% val, 10% test
    real_train = real_df[real_df['video'].isin(real_train_v)]
    real_val   = real_df[real_df['video'].isin(real_val_v)]
    real_test  = real_df[real_df['video'].isin(real_test_v)]

    fs_vids = faceswap_df['video'].unique()
    fs_train_v, fs_val_v = train_test_split(fs_vids, test_size=0.1, random_state=args.seed)  # 90% train , 10% val
    fs_train = faceswap_df[faceswap_df['video'].isin(fs_train_v)]
    fs_val   = faceswap_df[faceswap_df['video'].isin(fs_val_v)]

    # Step 3: Assemble splits
    train_df = pd.concat([real_train, fs_train], ignore_index=True)
    val_df   = pd.concat([real_val,   fs_val],   ignore_index=True)
    test_df  = pd.concat([real_test,  neuraltextures_df],    ignore_index=True)

    # Step 4: Save to CSV
    save_csv(train_df, output_dir / "train.csv")
    save_csv(val_df,   output_dir / "val.csv")
    save_csv(test_df,  output_dir / "test.csv")

    # print(f"train.csv → total={len(train_df)} | real={(train_df['label']==0).sum()} | fake={(train_df['label']==1).sum()}")
    # print(f"val.csv   → total={len(val_df)}   | real={(val_df['label']==0).sum()} | fake={(val_df['label']==1).sum()}")
    # print(f"test.csv  → total={len(test_df)}  | real={(test_df['label']==0).sum()} | fake={(test_df['label']==1).sum()}")
    print("finished splitting dataset.")
if __name__ == "__main__":

    main()
