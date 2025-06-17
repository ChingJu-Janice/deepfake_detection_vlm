import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model

class CLIPLinearProbeModel(nn.Module):
    """
    Baseline 版本：完全凍結 CLIP ViT-B/32，僅訓練線性分類頭
    """
    def __init__(self):
        super().__init__()
        
        # 載入原始 CLIP 模型並凍結所有參數
        self.backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 凍結全部 CLIP 參數（比 LoRA 版本更嚴格）
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 分類頭（與原版相同）
        embed_dim = self.backbone.config.projection_dim  # 512
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, inputs, device=None):
        # 與原版相同，省略 LoRA 相關處理
        if isinstance(inputs, list): 
            processed = self.processor(images=inputs, return_tensors="pt", padding=True)
            pixel_tensor = processed["pixel_values"].to(device)
        else: 
            pixel_tensor = inputs
            
        features = self.backbone.get_image_features(pixel_values=pixel_tensor)
        logits = self.classifier(features)
        return logits.squeeze(1)

    def save_weights(self, path):
        """
        儲存線性分類頭的權重到指定路徑。
        Args:
            path (str): 欲儲存權重的檔案路徑。
        """
        torch.save(self.state_dict(), path)
        print(f"Linear probe weights saved to {path}")