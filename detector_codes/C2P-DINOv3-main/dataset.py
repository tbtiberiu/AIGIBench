import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


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


def get_train_transforms(size=224):
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size + 32, antialias=True),
            v2.RandomCrop(size),
            v2.RandomHorizontalFlip(),
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.5,
            ),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))], p=0.5),
            v2.RandomApply([v2.JPEG(quality=(50, 95))], p=0.5),
            v2.RandomApply([v2.RandomGrayscale(p=1.0)], p=0.1),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transforms(size=224):
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size + 32, antialias=True),
            v2.CenterCrop(size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
