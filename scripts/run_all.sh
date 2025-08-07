#!/bin/bash
# Master Script for AutoMAE Training Pipeline
# Run from root folder with: bash scripts/run_all.sh
#
# This script provides a complete pipeline for:
# 1. Pretraining AutoMAE models
# 2. Fine-tuning the best model
# 3. Running a hyperparameter sweep (optional)

# Create a timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="automae_pipeline_$TIMESTAMP"
LOG_FILE="logs/pipeline_$TIMESTAMP.log"

# Make sure the log directory exists
mkdir -p logs

# Start logging
echo "Starting AutoMAE training pipeline: $RUN_NAME" | tee $LOG_FILE
echo "Timestamp: $(date)" | tee -a $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE

# Function to ask for confirmation
confirm() {
    read -p "$1 (y/n): " choice
    case "$choice" in 
        y|Y ) return 0;;
        * ) return 1;;
    esac
}

# Step 1: Run pretraining
echo "Step 1: Pretraining AutoMAE models" | tee -a $LOG_FILE
if confirm "Do you want to run the pretraining script?"; then
    echo "Running pretraining..." | tee -a $LOG_FILE
    bash scripts/pretrain_automae.sh 2>&1 | tee -a $LOG_FILE
    
    # Check if pretraining was successful
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error: Pretraining failed. Check the logs for details." | tee -a $LOG_FILE
        exit 1
    fi
else
    echo "Skipping pretraining step." | tee -a $LOG_FILE
fi

# Step 2: Fine-tuning
echo "----------------------------------------" | tee -a $LOG_FILE
echo "Step 2: Fine-tuning the best model" | tee -a $LOG_FILE
if confirm "Do you want to run the fine-tuning script?"; then
    # Ask for checkpoint path
    read -p "Enter the path to the checkpoint for fine-tuning: " CHECKPOINT_PATH
    
    if [ -f "$CHECKPOINT_PATH" ]; then
        echo "Running fine-tuning with checkpoint: $CHECKPOINT_PATH" | tee -a $LOG_FILE
        bash scripts/finetune_automae.sh $CHECKPOINT_PATH 2>&1 | tee -a $LOG_FILE
        
        # Check if fine-tuning was successful
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "Error: Fine-tuning failed. Check the logs for details." | tee -a $LOG_FILE
            exit 1
        fi
    else
        echo "Error: Checkpoint file not found: $CHECKPOINT_PATH" | tee -a $LOG_FILE
        exit 1
    fi
else
    echo "Skipping fine-tuning step." | tee -a $LOG_FILE
fi

# Step 3: Hyperparameter sweep (optional)
echo "----------------------------------------" | tee -a $LOG_FILE
echo "Step 3: Hyperparameter sweep (optional)" | tee -a $LOG_FILE
if confirm "Do you want to run the hyperparameter sweep?"; then
    echo "Running hyperparameter sweep..." | tee -a $LOG_FILE
    bash scripts/sweep_automae.sh 2>&1 | tee -a $LOG_FILE
    
    # Check if sweep was successful
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error: Hyperparameter sweep failed. Check the logs for details." | tee -a $LOG_FILE
        exit 1
    fi
else
    echo "Skipping hyperparameter sweep." | tee -a $LOG_FILE
fi

# Step 4: Multi-GPU training (optional)
echo "----------------------------------------" | tee -a $LOG_FILE
echo "Step 4: Multi-GPU training (optional)" | tee -a $LOG_FILE
if confirm "Do you want to run multi-GPU training?"; then
    echo "Running multi-GPU training..." | tee -a $LOG_FILE
    bash scripts/multigpu_automae.sh 2>&1 | tee -a $LOG_FILE
    
    # Check if multi-GPU training was successful
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error: Multi-GPU training failed. Check the logs for details." | tee -a $LOG_FILE
        exit 1
    fi
else
    echo "Skipping multi-GPU training." | tee -a $LOG_FILE
fi

echo "----------------------------------------" | tee -a $LOG_FILE
echo "AutoMAE training pipeline completed successfully!" | tee -a $LOG_FILE
echo "Timestamp: $(date)" | tee -a $LOG_FILE