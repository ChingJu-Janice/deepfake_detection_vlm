import os
import pandas as pd


def save_frame_scores(paths, labels, preds, result_dir):
    """Saves the inference results to a CSV file.

    Args:
        paths (list): List of file paths.
        labels (np.ndarray): Array of true labels.
        preds (np.ndarray): Array of predicted logits.
        result_dir (str): Directory to save the CSV file.
    """
    df = pd.DataFrame({
        'path': paths,
        'label': labels,
        'score': preds
    })

    frame_specific_dir = os.path.join(result_dir, "frame_level")
    os.makedirs(frame_specific_dir, exist_ok=True)
    df.to_csv(os.path.join(frame_specific_dir, 'frame_lora_score.csv'), index=False)
    
    
def save_video_scores(paths, labels, preds, result_dir):
    """
    用 pandas DataFrame 將 frame level 資料彙總成 video level，
    每個影片對應一筆資料，包含影片路徑、label（取第一個）、平均預測分數。
    
    Args:
        paths (List[str]): frame 相對路徑（如 Real_youtube/550/001.png）
        labels (np.ndarray): 每個 frame 的 label（影片內相同）
        preds (np.ndarray): 每個 frame 的預測分數（已 sigmoid）
        result_dir (str): 儲存 CSV 的目錄
    """

    df = pd.DataFrame({
        "path": paths,
        "label": labels,
        "score": preds  
    })
    df["video"] = df["path"].apply(lambda x: os.path.dirname(x))

    video_df = df.groupby("video").agg({
        "label": "first",
        "score": "mean"
    }).reset_index()

    video_specific_dir = os.path.join(result_dir, "video_level")
    os.makedirs(video_specific_dir, exist_ok=True)
    video_df.to_csv(os.path.join(video_specific_dir, 'video_lora_score.csv'), index=False)


