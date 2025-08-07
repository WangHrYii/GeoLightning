#!/bin/bash
# AutoMAE Fine-tuning Script
# Run from root folder with: bash scripts/finetune_automae.sh PATH_TO_PRETRAINED_CHECKPOINT
# 
# This script fine-tunes a pretrained AutoMAE model on the same dataset
# with different hyperparameters optimized for fine-tuning

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo "Error: Pretrained checkpoint path is required"
    echo "Usage: bash scripts/finetune_automae.sh PATH_TO_PRETRAINED_CHECKPOINT"
    exit 1
fi

# Set common variables
PYTHON_CMD="python src/train_AutoMAE.py"
CHECKPOINT_PATH=$1
GPU_ID=0  # Change this to use different GPU

# Create a fine-tuning specific task name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FINETUNE_TASK="automae_finetune_$TIMESTAMP"

echo "Starting fine-tuning using checkpoint: $CHECKPOINT_PATH"

# Run 1: Basic fine-tuning with lower learning rate
echo "Fine-tuning with lower learning rate..."
$PYTHON_CMD \
    trainer.devices=[$GPU_ID] \
    ckpt_path=$CHECKPOINT_PATH \
    task_name="${FINETUNE_TASK}_basic" \
    model.learning_rate=5e-5 \
    model.weight_decay=0.02 \
    trainer.max_epochs=100 \
    model.warmup_epochs=10 \
    model.mask_ratio=0.5 \
    callbacks.early_stopping.patience=15

# Run 2: Fine-tuning with even lower mask ratio (less masking)
echo "Fine-tuning with lower mask ratio..."
$PYTHON_CMD \
    trainer.devices=[$GPU_ID] \
    ckpt_path=$CHECKPOINT_PATH \
    task_name="${FINETUNE_TASK}_lowmask" \
    model.learning_rate=5e-5 \
    model.mask_ratio=0.4 \
    model.mask_factor=0.4 \
    trainer.max_epochs=100 \
    callbacks.early_stopping.patience=15

# Run 3: Fine-tuning with different adversarial loss weight
echo "Fine-tuning with adjusted adversarial loss weight..."
$PYTHON_CMD \
    trainer.devices=[$GPU_ID] \
    ckpt_path=$CHECKPOINT_PATH \
    task_name="${FINETUNE_TASK}_advloss" \
    model.learning_rate=5e-5 \
    model.loss_g_factor=0.1 \
    trainer.max_epochs=100 \
    callbacks.early_stopping.patience=15

echo "All fine-tuning jobs completed!"