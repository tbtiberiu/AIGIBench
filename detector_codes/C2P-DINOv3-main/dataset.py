import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import functional as tvf


class AIGIBenchDataset(Dataset):
    def __init__(self, hf_data, transform=None):
        self.hf_data = hf_data
        self.transform = transform

    def __len__(self):
        return len(self.hf_data)

    def __getitem__(self, idx):
        item = self.hf_data[idx]
        image = item['image'].convert('RGB')
        label = item['label']  # 0 for Real, 1 for Fake

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


class DownUpResize(nn.Module):
    def __init__(self, scale_range=(0.55, 0.9)):
        super().__init__()
        self.scale_range = scale_range
        self.interpolations = [
            InterpolationMode.BILINEAR,
            InterpolationMode.BICUBIC,
            InterpolationMode.NEAREST_EXACT,
        ]

    def forward(self, image):
        height, width = tvf.get_size(image)
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        down_height = max(32, int(round(height * scale)))
        down_width = max(32, int(round(width * scale)))

        down_interpolation = self.interpolations[
            torch.randint(len(self.interpolations), size=()).item()
        ]
        up_interpolation = self.interpolations[
            torch.randint(len(self.interpolations), size=()).item()
        ]

        image = tvf.resize(
            image,
            [down_height, down_width],
            interpolation=down_interpolation,
            antialias=down_interpolation != InterpolationMode.NEAREST_EXACT,
        )
        return tvf.resize(
            image,
            [height, width],
            interpolation=up_interpolation,
            antialias=up_interpolation != InterpolationMode.NEAREST_EXACT,
        )


def get_train_transforms(size=256):
    resize_size = max(int(round(size * 1.15)), size)
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(resize_size, antialias=True),
            v2.RandomCrop(size),
            v2.RandomHorizontalFlip(),
            v2.RandomApply([DownUpResize()], p=0.25),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.2))], p=0.25),
            v2.RandomAdjustSharpness(sharpness_factor=1.5, p=0.15),
            v2.RandomApply([v2.JPEG(quality=(40, 95))], p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transforms(size=256):
    resize_size = max(int(round(size * 1.15)), size)
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(resize_size, antialias=True),
            v2.CenterCrop(size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
