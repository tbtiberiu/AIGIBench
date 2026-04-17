import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description='Train C2P DINOv3 Model')
    parser.add_argument(
        '--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=8, help='Batch size (default: 8)'
    )
    parser.add_argument(
        '--epochs', type=int, default=1, help='Number of epochs (default: 1)'
    )
    parser.add_argument(
        '--max-steps', type=int, default=2000, help='Max training steps (default: 2000)'
    )
    parser.add_argument(
        '--num-workers', type=int, default=8, help='DataLoader workers (default: 8)'
    )
    parser.add_argument(
        '--seed', type=int, default=123, help='Random seed (default: 123)'
    )
    parser.add_argument(
        '--no-val', action='store_true', help='Skip validation after each epoch'
    )
    parser.add_argument(
        '--val-every',
        type=int,
        default=1,
        help='Run validation every N epochs (default: 1)',
    )
    parser.add_argument(
        '--pct-start',
        type=float,
        default=0.1,
        help='OneCycleLR warmup fraction (default: 0.1)',
    )
    return parser.parse_args()


def run_validation(model, val_loader, criterion, device, epoch):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    print('Running validation...')
    with torch.inference_mode():
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
    print(f'Epoch {epoch} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}')
    return avg_val_loss, avg_val_acc


def train():
    args = parse_args()
    seed_everything(args.seed)

    # Load Environment Variables
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(
        f'Config: lr={args.lr}, batch_size={args.batch_size}, epochs={args.epochs}, '
        f'max_steps={args.max_steps}, val={"disabled" if args.no_val else f"every {args.val_every} epoch(s)"}'
    )

    # Checkpoints directory
    checkpoints_dir = os.path.join(current_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Load Dataset
    print('Loading AIGIBench dataset from HuggingFace...')
    ds = load_dataset('TheKernel01/AIGIBench', token=HF_TOKEN)

    train_ds = AIGIBenchDataset(ds['train'], transform=get_train_transforms())
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = None
    if not args.no_val:
        val_ds = AIGIBenchDataset(ds['validation'], transform=get_val_transforms())
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # Initialize Model
    model = C2P_DINOv3_Model().to(device)

    # Optimizer and Loss
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    criterion = nn.BCEWithLogitsLoss()

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.max_steps,
        pct_start=args.pct_start,
        anneal_strategy='cos',
    )

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(
            train_loader, desc=f'Epoch {epoch}/{args.epochs}', total=args.max_steps
        )
        for i, (images, labels) in enumerate(pbar):
            if global_step >= args.max_steps:
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
        should_validate = (
            not args.no_val and val_loader is not None and epoch % args.val_every == 0
        )

        avg_val_acc, avg_val_loss = None, None
        if should_validate:
            avg_val_loss, avg_val_acc = run_validation(
                model, val_loader, criterion, device, epoch
            )

        # Save Checkpoint
        if avg_val_acc is not None:
            checkpoint_name = f'dinov3_lora_epoch_{epoch}_acc_{avg_val_acc:.4f}.pth'
        else:
            checkpoint_name = f'dinov3_lora_epoch_{epoch}_step_{global_step}.pth'

        save_path = os.path.join(checkpoints_dir, checkpoint_name)
        torch.save(
            {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_acc': avg_val_acc,
                'args': vars(args),
            },
            save_path,
        )
        print(f'Saved checkpoint to {save_path}')

        if global_step >= args.max_steps:
            print(f'Reached max_steps={args.max_steps}, stopping.')
            break

    print('Training Complete!')


if __name__ == '__main__':
    train()
