import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer

# Add the current directory to sys.path so it can find networks, options, etc.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from networks.trainer import Trainer
from options.train_options import TrainOptions
from utils.util import Logger


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


class AIGIBenchDataset(Dataset):
    def __init__(self, hf_data, opt, transform=None):
        self.hf_data = hf_data
        self.opt = opt
        self.transform = transform

        # Use CLIP model path or HF name for tokenizer
        tokenizer_name = (
            opt.clip if os.path.exists(opt.clip) else 'openai/clip-vit-large-patch14'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, model_max_length=77, padding_side='right', use_fast=False
        )
        self.tokenizer.pad_token_id = 0

    def __len__(self):
        return len(self.hf_data)

    def __getitem__(self, idx):
        item = self.hf_data[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        generator_id = item['generator']

        # Generator mapping based on AIGIBench spec
        gen_names = {0: 'Real', 1: 'ProGAN', 2: 'SD14'}
        gen_name = gen_names.get(generator_id, 'Unknown')

        cates = self.opt.cates
        cates_len = len(cates) // 2

        # Generic image description to guide the contrastive learning
        text = f'An image produced by {gen_name}'

        if label == 1:  # Fake (AI-generated)
            text = (
                f'{" ".join(cates[:cates_len])}. {text} {" ".join(cates[:cates_len])}.'
            )
        else:  # Real (Authentic)
            text = (
                f'{" ".join(cates[cates_len:])}. {text} {" ".join(cates[cates_len:])}.'
            )

        inputs = self.tokenizer(
            [text],
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        if self.transform:
            image = self.transform(image)

        # Format: path, image, text, input_ids, attention_mask, label
        return 'hf_dataset', image, text, input_ids, attention_mask, label


def get_train_transforms(opt):
    return transforms.Compose(
        [
            transforms.Resize(opt.loadSize),
            transforms.RandomCrop(opt.cropSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


def get_val_transforms(opt):
    return transforms.Compose(
        [
            transforms.Resize(opt.loadSize),
            transforms.CenterCrop(opt.cropSize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


if __name__ == '__main__':
    # Parse options
    opt = TrainOptions().parse()
    seed_torch(opt.seed)

    # Handle CLIP path if folder doesn't exist locally
    if not os.path.exists(opt.clip):
        print(
            f"CLIP path {opt.clip} not found locally. Using 'openai/clip-vit-large-patch14' from HuggingFace."
        )
        opt.clip = 'openai/clip-vit-large-patch14'

    # Initialize logger
    log_path = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    Logger(os.path.join(log_path, 'log.log'))

    print("Loading AIGIBench dataset from HuggingFace...")
    ds = load_dataset('TheKernel01/AIGIBench')
    train_data = ds['train']

    val_data = ds['validation']

    train_dataset = AIGIBenchDataset(
        train_data, opt, transform=get_train_transforms(opt)
    )
    val_dataset = AIGIBenchDataset(val_data, opt, transform=get_val_transforms(opt))

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_threads,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_threads,
    )

    # Initialize Trainer
    model = Trainer(opt)

    print(f'Starting training on {len(train_data)} samples...')
    for epoch in range(opt.niter):
        model.train()
        epoch_start_time = time.time()

        for i, data in enumerate(train_loader):
            model.total_steps += 1

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(
                    f'{time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())} Train loss: {model.loss:.4f} loss1: {model.loss1:.4f} loss2-cla: {model.loss2:.4f} at step: {model.total_steps} lr {model.lr}'
                )

            if model.total_steps >= opt.total_steps:
                break

        # Validation at the end of epoch
        print('Running validation...')
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            count = 0
            # Limit validation samples for speed during training
            max_val_samples = 2000

            for i, data in enumerate(val_loader):
                model.set_input(data)
                # Trainer's model is DataParallel, access it directly for inference
                _, classhead = model.model(
                    model.input, model.input_ids, model.attention_mask
                )

                loss = nn.functional.binary_cross_entropy_with_logits(
                    classhead, model.label
                )
                val_loss += loss.item() * len(model.label)

                preds = (torch.sigmoid(classhead) > 0.5).float()
                val_acc += (preds == model.label).sum().item()
                count += len(model.label)

                if count >= max_val_samples:
                    break

            avg_val_loss = val_loss / count
            avg_val_acc = val_acc / count
            print(
                f'Epoch {epoch} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}'
            )

        # Save checkpoint
        model.save_networks(f'epoch_{epoch}_acc_{avg_val_acc:.4f}')

        if model.total_steps >= opt.total_steps:
            print('Reached total_steps limit. Ending training.')
            break

    print('Training complete.')
