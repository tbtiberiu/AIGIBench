import argparse
import os
import random
import sys
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import AIGIBenchDataset, get_train_transforms, get_val_transforms
from datasets import load_dataset
from dotenv import load_dotenv
from model import C2P_DINOv3_Model
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train C2P DINOv3 Model')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone-lr-scale', type=float, default=0.25)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=5000)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.5)
    parser.add_argument('--forensic-dim', type=int, default=256)
    parser.add_argument('--unfreeze-last-blocks', type=int, default=2)
    parser.add_argument(
        '--lora-target-modules',
        type=str,
        default='q_proj,k_proj,v_proj,out_proj,fc1,fc2',
    )
    parser.add_argument('--no-val', action='store_true')
    parser.add_argument('--val-every', type=int, default=1)
    parser.add_argument('--pct-start', type=float, default=0.1)
    return parser.parse_args()


def get_amp_context(device):
    if device.type == 'cuda':
        return torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    return nullcontext()


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def build_optimizer(model, args):
    backbone_params = []
    fast_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('backbone') and 'lora_' not in name:
            backbone_params.append(parameter)
        else:
            fast_params.append(parameter)

    parameter_groups = []
    if backbone_params:
        parameter_groups.append(
            {
                'params': backbone_params,
                'lr': args.lr * args.backbone_lr_scale,
            }
        )
    if fast_params:
        parameter_groups.append({'params': fast_params, 'lr': args.lr})

    return optim.AdamW(
        parameter_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def save_checkpoint(path, epoch, global_step, model, optimizer, metrics, args):
    torch.save(
        {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'args': vars(args),
        },
        path,
    )


def run_validation(model, val_loader, criterion, device, epoch):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    real_correct = 0
    fake_correct = 0
    real_total = 0
    fake_total = 0

    print('Running validation...')
    with torch.inference_mode():
        for images, labels in tqdm(val_loader, desc='Validating', leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            with get_amp_context(device):
                logits = model(images)
                loss = criterion(logits, labels)

            batch_size = labels.size(0)
            val_loss += loss.item() * batch_size

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += batch_size

            real_mask = labels == 0
            fake_mask = labels == 1
            if real_mask.any():
                real_correct += (preds[real_mask] == labels[real_mask]).sum().item()
                real_total += real_mask.sum().item()
            if fake_mask.any():
                fake_correct += (preds[fake_mask] == labels[fake_mask]).sum().item()
                fake_total += fake_mask.sum().item()

    metrics = {
        'val_loss': val_loss / max(total, 1),
        'val_acc': correct / max(total, 1),
        'val_real_acc': real_correct / max(real_total, 1),
        'val_fake_acc': fake_correct / max(fake_total, 1),
    }
    metrics['val_balanced_acc'] = 0.5 * (
        metrics['val_real_acc'] + metrics['val_fake_acc']
    )
    print(
        f'Epoch {epoch} | Val Loss: {metrics["val_loss"]:.4f} | '
        f'Val Acc: {metrics["val_acc"]:.4f} | '
        f'Val BAcc: {metrics["val_balanced_acc"]:.4f} | '
        f'Real Acc: {metrics["val_real_acc"]:.4f} | '
        f'Fake Acc: {metrics["val_fake_acc"]:.4f}'
    )
    return metrics


def train():
    args = parse_args()
    seed_everything(args.seed)

    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(
        f'Config: lr={args.lr}, batch_size={args.batch_size}, epochs={args.epochs}, '
        f'max_steps={args.max_steps}, image_size={args.image_size}, '
        f'val={"disabled" if args.no_val else f"every {args.val_every} epoch(s)"}'
    )

    checkpoints_dir = os.path.join(current_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    print('Loading AIGIBench dataset from HuggingFace...')
    dataset = load_dataset('TheKernel01/AIGIBench', token=hf_token)

    train_ds = AIGIBenchDataset(
        dataset['train'],
        transform=get_train_transforms(size=args.image_size),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    val_loader = None
    if not args.no_val:
        val_ds = AIGIBenchDataset(
            dataset['validation'],
            transform=get_val_transforms(
                size=args.image_size,
            ),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == 'cuda',
        )

    model = C2P_DINOv3_Model(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=[
            module.strip()
            for module in args.lora_target_modules.split(',')
            if module.strip()
        ],
        forensic_dim=args.forensic_dim,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
        image_size=args.image_size,
    ).to(device)

    trainable_params, total_params = count_parameters(model)
    print(
        f'Trainable params: {trainable_params / 1e6:.2f}M / '
        f'{total_params / 1e6:.2f}M ({100 * trainable_params / max(total_params, 1):.2f}%)'
    )

    optimizer = build_optimizer(model, args)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group['lr'] for group in optimizer.param_groups],
        total_steps=args.max_steps,
        pct_start=args.pct_start,
        anneal_strategy='cos',
    )

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    best_balanced_acc = float('-inf')
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        remaining_steps = max(args.max_steps - global_step, 0)
        epoch_total = min(len(train_loader), remaining_steps) if remaining_steps else 0
        pbar = tqdm(
            train_loader, desc=f'Epoch {epoch}/{args.epochs}', total=epoch_total
        )

        for step_idx, (images, labels) in enumerate(pbar, start=1):
            if global_step >= args.max_steps:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)

            with get_amp_context(device):
                logits = model(images)
                loss = criterion(logits, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.gradient_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.gradient_clip,
                    )
                optimizer.step()

            scheduler.step()
            global_step += 1
            running_loss += loss.item()

            if step_idx % 10 == 0 or step_idx == 1:
                current_lr = max(group['lr'] for group in optimizer.param_groups)
                pbar.set_postfix(
                    {
                        'loss': f'{running_loss / step_idx:.4f}',
                        'lr': f'{current_lr:.2e}',
                    }
                )

        metrics = {}
        should_validate = (
            not args.no_val and val_loader is not None and epoch % args.val_every == 0
        )
        if should_validate:
            metrics = run_validation(model, val_loader, criterion, device, epoch)

        checkpoint_name = (
            f'dinov3_epoch_{epoch}_bacc_{metrics["val_balanced_acc"]:.4f}.pth'
            if metrics
            else f'dinov3_epoch_{epoch}_step_{global_step}.pth'
        )
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
        save_checkpoint(
            checkpoint_path,
            epoch,
            global_step,
            model,
            optimizer,
            metrics,
            args,
        )
        print(f'Saved checkpoint to {checkpoint_path}')

        if metrics and metrics['val_balanced_acc'] > best_balanced_acc:
            best_balanced_acc = metrics['val_balanced_acc']
            best_path = os.path.join(checkpoints_dir, 'model_epoch_best.pth')
            save_checkpoint(
                best_path,
                epoch,
                global_step,
                model,
                optimizer,
                metrics,
                args,
            )
            print(f'Updated best checkpoint at {best_path}')
        elif not metrics:
            best_path = os.path.join(checkpoints_dir, 'model_epoch_best.pth')
            if not os.path.exists(best_path):
                save_checkpoint(
                    best_path,
                    epoch,
                    global_step,
                    model,
                    optimizer,
                    metrics,
                    args,
                )
                print(f'Initialized best checkpoint at {best_path}')

        latest_path = os.path.join(checkpoints_dir, 'model_epoch_last.pth')
        save_checkpoint(
            latest_path,
            epoch,
            global_step,
            model,
            optimizer,
            metrics,
            args,
        )

        if global_step >= args.max_steps:
            print(f'Reached max_steps={args.max_steps}, stopping.')
            break

    print('Training complete!')


if __name__ == '__main__':
    train()
