#!/bin/bash

# Training script for AIGIBench using C2P-CLIP
# This command reproduces the configuration found in the latest checkpoint.

python ./train_aigibench.py \
    --name aigibench_full_train \
    --arch res50 \
    --batch_size 8 \
    --lr 0.0001 \
    --niter 1 \
    --total_steps 5000 \
    --loadSize 256 \
    --cropSize 224 \
    --seed 123 \
    --clip openai/clip-vit-large-patch14 \
    --claloss 0.5 \
    --cates Deepfake Camera \
    --eval_freq 200 \
    --loss_freq 50 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --num_threads 4
