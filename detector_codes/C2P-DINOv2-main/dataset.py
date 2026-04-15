import torch
from torch.utils.data import Dataset
from torchvision import transforms


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
    return transforms.Compose(
        [
            transforms.Resize(size + 32),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transforms(size=224):
    return transforms.Compose(
        [
            transforms.Resize(size + 32),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
