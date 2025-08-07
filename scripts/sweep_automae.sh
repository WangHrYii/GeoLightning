#!/bin/bash
# AutoMAE Hyperparameter Sweep Script
# Run from root folder with: bash scripts/sweep_automae.sh
# 
# This script performs a systematic hyperparameter sweep using Hydra's multirun functionality
# It will create multiple runs with different combinations of hyperparameters

# Set common variables
PYTHON_CMD="python src/train_AutoMAE.py"
GPU_ID=0  # Change this to use different GPU

# Create a timestamp for this sweep
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SWEEP_NAME="automae_sweep_$TIMESTAMP"

echo "Starting hyperparameter sweep: $SWEEP_NAME"

# Run a sweep over mask_ratio and mask_factor
# This uses Hydra's multirun functionality with the --multirun flag
$PYTHON_CMD \
    --multirun \
    task_name=$SWEEP_NAME \
    trainer.devices=[$GPU_ID] \
    trainer.max_epochs=100 \
    model.mask_ratio=0.65,0.7,0.75,0.8 \
    model.mask_factor=0.4,0.5,0.6 \
    model.learning_rate=1e-4,1.5e-4 \
    hydra.sweep.dir=logs/sweeps/$SWEEP_NAME \
    hydra.job.num_jobs=4  # Run up to 4 jobs in parallel, adjust based on your system

echo "Hyperparameter sweep completed!"

# Note: This will create a grid of all combinations:
# - 4 mask_ratio values
# - 3 mask_factor values
# - 2 learning_rate values
# = 24 total runs
#
# To run a smaller subset or specific combinations, you can modify the parameters or 
# use Hydra's more advanced sweep configuration options in a separate config file.