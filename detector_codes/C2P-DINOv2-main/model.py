import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import Dinov2Model


class C2P_DINOv2_Model(nn.Module):
    def __init__(
        self,
        model_name='facebook/dinov2-large',
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    ):
        super(C2P_DINOv2_Model, self).__init__()

        # Load the DINOv2 model
        self.backbone = Dinov2Model.from_pretrained(model_name)
        self.backbone.requires_grad_(False)  # Freeze by default

        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=[
                'query',
                'key',
                'value',
            ],  # DINOv2 in transformers uses query/key/value in attention
            lora_dropout=lora_dropout,
            bias='none',
        )

        # Apply LoRA to the backbone
        self.backbone = get_peft_model(self.backbone, lora_config)

        # Head (DINOv2-Large has hidden size of 1024)
        hidden_size = self.backbone.config.hidden_size
        self.fc = nn.Linear(hidden_size, 1)
        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(self.fc.bias.data, 0.0)

    def forward(self, x):
        outputs = self.backbone(x)
        # We use the [CLS] token representation for classification
        # pooler_output is usually the CLS token in transformers implementation of ViT/DINOv2
        cls_token = outputs.pooler_output
        return self.fc(cls_token)

    def detect(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(1)
