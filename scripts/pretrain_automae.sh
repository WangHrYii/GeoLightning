#!/bin/bash
# AutoMAE Pretraining Script
# Run from root folder with: bash scripts/pretrain_automae.sh
# 
# This script runs multiple pretraining configurations with different hyperparameters
# Each run will create its own output directory with logs and checkpoints

# Set common variables
PYTHON_CMD="python src/train_AutoMAE.py"
BASE_CONFIG="configs/AutoMAE/config.yaml"
GPU_ID=0  # Change this to use different GPU

# Run 1: Default configuration
echo "Starting pretraining with default configuration..."
$PYTHON_CMD trainer.devices=[$GPU_ID]

# Run 2: Different mask ratio
echo "Starting pretraining with mask_ratio=0.65..."
$PYTHON_CMD trainer.devices=[$GPU_ID] model.mask_ratio=0.65 task_name="automae_millionaid_pretrain_mask65"

# Run 3: Different mask factor
echo "Starting pretraining with mask_factor=0.6..."
$PYTHON_CMD trainer.devices=[$GPU_ID] model.mask_factor=0.6 task_name="automae_millionaid_pretrain_maskfactor60"

# Run 4: Different generator loss weight
echo "Starting pretraining with loss_g_factor=0.3..."
$PYTHON_CMD trainer.devices=[$GPU_ID] model.loss_g_factor=0.3 task_name="automae_millionaid_pretrain_gfactor30"

# Run 5: Smaller batch size (for lower memory GPUs)
echo "Starting pretraining with batch_size=128..."
$PYTHON_CMD trainer.devices=[$GPU_ID] data.batch_size=128 task_name="automae_millionaid_pretrain_bs128"

# Run 6: Higher learning rate
echo "Starting pretraining with learning_rate=2e-4..."
$PYTHON_CMD trainer.devices=[$GPU_ID] model.learning_rate=2e-4 task_name="automae_millionaid_pretrain_lr2e-4"

echo "All pretraining jobs completed!"