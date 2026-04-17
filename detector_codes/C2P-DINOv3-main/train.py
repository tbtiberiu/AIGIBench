import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dataset import AIGIBenchDataset, get_train_transforms, get_val_transforms
from model import C2P_DINOv3_Model


def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    seed_everything()

    # Load Environment Variables
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Checkpoints directory
    checkpoints_dir = os.path.join(current_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Load Dataset
    print('Loading AIGIBench dataset from HuggingFace...')
    ds = load_dataset('TheKernel01/AIGIBench', token=HF_TOKEN)

    train_ds = AIGIBenchDataset(ds['train'], transform=get_train_transforms())
    val_ds = AIGIBenchDataset(ds['validation'], transform=get_val_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False, num_workers=8, pin_memory=True
    )

    # Initialize Model
    model = C2P_DINOv3_Model().to(device)

    # Optimizer and Loss
    lr = 1e-4
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 1
    max_steps = 2500
    global_step = 0

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=max_steps,
        pct_start=0.1,
        anneal_strategy='cos',
    )

    print(f'Starting training for {num_epochs} epoch or {max_steps} steps...')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for i, (images, labels) in enumerate(pbar):
            if global_step >= max_steps:
                break

            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_loss += loss.item()
            if i % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(
                    {'loss': running_loss / (i + 1), 'lr': f'{current_lr:.2e}'}
                )

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        print('Running validation...')
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating', leave=False):
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)
        print(
            f'Epoch {epoch + 1} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}'
        )

        # Save Checkpoint
        checkpoint_name = f'dinov3_lora_epoch_{epoch + 1}_acc_{avg_val_acc:.4f}.pth'
        save_path = os.path.join(checkpoints_dir, checkpoint_name)
        torch.save(
            {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_acc': avg_val_acc,
            },
            save_path,
        )
        print(f'Saved checkpoint to {save_path}')

    print('Training Complete!')


if __name__ == '__main__':
    train()
