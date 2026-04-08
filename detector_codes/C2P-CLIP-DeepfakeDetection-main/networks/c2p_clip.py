import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel


class C2P_CLIP_Model(nn.Module):
    def __init__(
        self,
        name='openai/clip-vit-large-patch14',
        num_classes=1,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    ):
        super(C2P_CLIP_Model, self).__init__()

        self.model = CLIPModel.from_pretrained(name)
        del self.model.text_model
        del self.model.text_projection
        del self.model.logit_scale

        self.vision_tower = self.model.vision_model
        self.vision_tower.requires_grad_(False)
        self.model.visual_projection.requires_grad_(False)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=['q_proj', 'k_proj', 'v_proj'],
            lora_dropout=lora_dropout,
            bias='none',
        )
        self.vision_tower_lora = get_peft_model(self.vision_tower, lora_config)

        self.model.fc = nn.Linear(768, num_classes)
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, 0.02)

    def encode_image(self, img):
        vision_outputs = self.vision_tower_lora(
            pixel_values=img,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_hidden_states,
            return_dict=self.model.config.return_dict,
        )
        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.model.visual_projection(pooled_output)
        return image_features

    def forward(self, img):
        image_embeds = self.encode_image(img)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return self.model.fc(image_embeds)

    def detect(self, img):
        with torch.no_grad():
            output = self.forward(img)
            return torch.sigmoid(output).squeeze(1)
