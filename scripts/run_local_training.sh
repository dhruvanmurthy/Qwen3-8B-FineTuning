#!/bin/bash
# Local training script for quick testing
# Trains on small dataset locally or on single GPU

set -e

# Always run from the repo root
cd "$(dirname "$0")/.." 

echo "================================================"
echo "Local Training Script"
echo "================================================"

# Configuration
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/local_test}"

echo "Configuration:"
echo "  Num samples: $NUM_SAMPLES"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Output dir: $OUTPUT_DIR"

# Check if datasets exist
if [ ! -d "data/processed" ]; then
    echo "Error: datasets not found in data/processed/"
    echo "Run: bash scripts/prepare_datasets.sh"
    exit 1
fi

# Run training
python3 src/train.py \
    --model_name_or_path "Qwen/Qwen3-8B" \
    --data_dir "./data/processed" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --report_to "wandb" \
    --logging_steps 10 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --bf16 \
    --gradient_checkpointing \
    --seed 42

echo "================================================"
echo "✓ Local training complete"
echo "Output: $OUTPUT_DIR"
echo "================================================"
