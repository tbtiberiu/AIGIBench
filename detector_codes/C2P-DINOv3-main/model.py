import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel


class C2P_DINOv3_Model(nn.Module):
    def __init__(
        self,
        model_name='facebook/dinov3-vitl16-pretrain-lvd1689m',
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    ):
        super(C2P_DINOv3_Model, self).__init__()

        # Load the DINOv3 ViT-Large model
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.backbone.requires_grad_(False)  # Freeze by default

        # Configure LoRA for ViT
        # We target q_proj, k_proj, v_proj in the attention blocks based on DINOv3 architecture
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=['q_proj', 'k_proj', 'v_proj'],
            lora_dropout=lora_dropout,
            bias='none',
        )

        # Apply LoRA to the backbone
        self.backbone = get_peft_model(self.backbone, lora_config)

        # Head (ViT-Large has hidden size of 1024)
        hidden_size = self.backbone.config.hidden_size
        self.fc = nn.Linear(hidden_size, 1)
        torch.nn.init.zeros_(self.fc.weight.data)
        torch.nn.init.zeros_(self.fc.bias.data)

    def forward(self, x):
        outputs = self.backbone(x)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.fc(cls_token)

    def detect(self, x):
        with torch.inference_mode():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(1)
