import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, dim = x.shape
        query = self.query.expand(batch_size, -1, -1)
        key = self.key(x)
        value = self.value(x)

        attention = torch.matmul(query, key.transpose(-2, -1)) / (dim**0.5)
        attention = self.softmax(attention)
        return torch.matmul(attention, value).squeeze(1)


class ForensicBranch(nn.Module):
    """Forensic Branch — captures high-frequency artifacts, initialized with SRM filters."""

    def __init__(self, out_dim=256):
        super().__init__()
        # 5 SRM filters x 3 channels = 15 output channels
        self.conv1 = nn.Conv2d(3, 15, kernel_size=5, padding=2, bias=False, groups=3)

        kernels = torch.tensor(
            [
                [
                    [0, 0, -1, 0, 0],
                    [0, -1, -2, -1, 0],
                    [-1, -2, 16, -2, -1],
                    [0, -1, -2, -1, 0],
                    [0, 0, -1, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, -2, 1, 0],
                    [0, -2, 4, -2, 0],
                    [0, 1, -2, 1, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, -1, 2, -1, 0],
                    [0, 2, -4, 2, 0],
                    [0, -1, 2, -1, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, -1, 0, 0],
                    [0, 0, 2, 0, 0],
                    [0, 0, -1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [-1, 2, -1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
            ],
            dtype=torch.float32,
        )
        # Initialize conv1 with the SRM kernels (shape 15, 1, 5, 5 for groups=3)
        self.conv1.weight.data = kernels.unsqueeze(1).repeat(3, 1, 1, 1)

        self.encoder = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Residual = image minus low-freq smoothed version
        blur = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        residual = x - blur

        out = self.conv1(residual)
        out = self.encoder(out)
        return self.pool(out).flatten(1)


def _resolve_attr_path(module, attr_path):
    current = module
    for attr in attr_path.split('.'):
        if not hasattr(current, attr):
            return None
        current = getattr(current, attr)
    return current


def _find_transformer_layers(backbone):
    candidate_paths = [
        'encoder.layer',
        'layers',
        'blocks',
        'transformer.layer',
        'transformer.layers',
        'model.encoder.layer',
        'model.layers',
        'backbone.encoder.layer',
        'backbone.layers',
        'backbone.blocks',
    ]
    for path in candidate_paths:
        layers = _resolve_attr_path(backbone, path)
        if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 0:
            return list(layers)

    for _, module in backbone.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 4:
            return list(module)

    return []


class DeForge_AI_Model(nn.Module):
    def __init__(
        self,
        model_name='facebook/dinov3-vitl16-pretrain-lvd1689m',
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.5,
        lora_target_modules=None,
        forensic_dim=256,
        unfreeze_last_blocks=0,
        image_size=256,
    ):
        super().__init__()

        if lora_target_modules is None:
            lora_target_modules = [
                'q_proj',
                'k_proj',
                'v_proj',
                'out_proj',
                'fc1',
                'fc2',
            ]

        self.image_size = image_size
        self.forensic_dim = forensic_dim

        backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        backbone.requires_grad_(False)
        self._unfreeze_last_blocks(backbone, unfreeze_last_blocks)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias='none',
        )
        self.backbone = get_peft_model(backbone, lora_config)

        hidden_size = self.backbone.config.hidden_size
        self.attn_pool = AttentionPooling(hidden_size)
        self.rgb_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.forensic_branch = ForensicBranch(out_dim=forensic_dim)
        self.forensic_gate = nn.Parameter(torch.tensor(0.3))
        self.head = nn.Sequential(
            nn.Linear(hidden_size + forensic_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _unfreeze_last_blocks(self, backbone, unfreeze_last_blocks):
        if unfreeze_last_blocks <= 0:
            return

        layers = _find_transformer_layers(backbone)
        if not layers:
            return

        for block in layers[-unfreeze_last_blocks:]:
            block.requires_grad_(True)

        for attr_path in [
            'layernorm',
            'norm',
            'post_layernorm',
            'ln_post',
            'final_layer_norm',
        ]:
            module = _resolve_attr_path(backbone, attr_path)
            if isinstance(module, nn.Module):
                module.requires_grad_(True)

    def forward(self, x):
        outputs = self.backbone(x)
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        patch_tokens = last_hidden_state[:, 1:, :]

        token_features = self.attn_pool(patch_tokens)
        rgb_features = self.rgb_proj(torch.cat([cls_token, token_features], dim=1))
        forensic_features = self.forensic_branch(x)
        return self.head(
            torch.cat([rgb_features, self.forensic_gate * forensic_features], dim=1)
        )

    def detect(self, x):
        with torch.inference_mode():
            return torch.sigmoid(self.forward(x)).squeeze(1)
