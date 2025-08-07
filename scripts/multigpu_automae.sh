#!/bin/bash
# AutoMAE Multi-GPU Training Script
# Run from root folder with: bash scripts/multigpu_automae.sh
# 
# This script runs AutoMAE training using multiple GPUs with DDP (Distributed Data Parallel)
# It demonstrates different multi-GPU configurations and strategies

# Set common variables
PYTHON_CMD="python src/train_AutoMAE.py"

# Create a timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MULTIGPU_NAME="automae_multigpu_$TIMESTAMP"

echo "Starting multi-GPU training: $MULTIGPU_NAME"

# Check available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ $NUM_GPUS -lt 2 ]; then
    echo "Warning: Multi-GPU training requires at least 2 GPUs. Falling back to single GPU mode."
    GPU_CONFIG="[0]"
    STRATEGY=""
    BATCH_SIZE=256
else
    # Use all available GPUs
    GPU_IDS=$(seq -s, 0 $(($NUM_GPUS-1)))
    GPU_CONFIG="[$GPU_IDS]"
    STRATEGY="strategy=ddp"
    
    # Scale batch size with number of GPUs
    BATCH_SIZE=$((256 * $NUM_GPUS))
    echo "Scaling batch size to $BATCH_SIZE for $NUM_GPUS GPUs"
fi

# Run 1: Basic multi-GPU training
echo "Running multi-GPU training with DDP strategy..."
$PYTHON_CMD \
    task_name="${MULTIGPU_NAME}_basic" \
    trainer.devices=$GPU_CONFIG \
    $STRATEGY \
    data.batch_size=$BATCH_SIZE

# Run 2: Multi-GPU training with gradient accumulation
# Useful for simulating larger batch sizes when memory is limited
echo "Running multi-GPU training with gradient accumulation..."
$PYTHON_CMD \
    task_name="${MULTIGPU_NAME}_grad_accum" \
    trainer.devices=$GPU_CONFIG \
    $STRATEGY \
    trainer.accumulate_grad_batches=2 \
    data.batch_size=$((BATCH_SIZE / 2))  # Reduce batch size when using gradient accumulation

# Run 3: Multi-GPU training with precision adjustment
# Using mixed precision can speed up training significantly
echo "Running multi-GPU training with mixed precision..."
$PYTHON_CMD \
    task_name="${MULTIGPU_NAME}_mixed_precision" \
    trainer.devices=$GPU_CONFIG \
    $STRATEGY \
    trainer.precision=16-mixed \
    data.batch_size=$BATCH_SIZE

echo "All multi-GPU training jobs completed!"