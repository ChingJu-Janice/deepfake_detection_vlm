import torch
import torch.nn as nn
import os
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model


class CLIPLoRAModel(nn.Module):
    """
    使用 CLIP ViT-B/32 作為 backbone，結合 LoRA 進行高效微調，
    搭配線性分類器進行二分類（deepfake / real）。
    """
    def __init__(self, lora_rank=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()

        self.backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # 配置 LoRA 並注入到 vision encoder 的注意力層
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "v_proj"]
        )
        self.backbone.vision_model = get_peft_model(self.backbone.vision_model, lora_config)

        # 凍結所有參數，僅允許 LoRA 權重更新
        for param in self.backbone.parameters():
            param.requires_grad = False
        for name, param in self.backbone.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        # 加上分類頭，將 CLIP image feature → 單一預測值
        embed_dim = self.backbone.config.projection_dim  # 一般為 512
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, inputs, device=None):
        """
        Args:
            inputs: list[PIL.Image] 或已處理 tensor [B, 3, 224, 224]
            device: torch.device，若為 None 則自動偵測模型所在裝置
        Returns:
            logits: Tensor [B]，每張圖片對應一個預測值
        """
        if isinstance(inputs, list): 
            processed = self.processor(images=inputs, return_tensors="pt", padding=True)
            pixel_tensor = processed["pixel_values"].to(device)
        else: 
            pixel_tensor = inputs

        features = self.backbone.get_image_features(pixel_values=pixel_tensor)
        logits = self.classifier(features)
        return logits.squeeze(1)
    
    def save_weights(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
            
        # self.backbone.save_pretrained(f"{os.path.dirname(path)}/lora_adapter")