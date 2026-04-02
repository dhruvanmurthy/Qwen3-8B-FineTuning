#!/bin/bash
# Local training script — supports single-GPU and multi-GPU (torchrun)
#
# Usage:
#   bash scripts/run_local_training.sh          # auto-detect GPUs
#   GPUS=1 bash scripts/run_local_training.sh   # force single-GPU

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

# GPU detection — default to all visible GPUs
if [ -z "$GPUS" ]; then
    GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
fi

echo "Configuration:"
echo "  Num samples : $NUM_SAMPLES"
echo "  Epochs      : $EPOCHS"
echo "  Batch size  : $BATCH_SIZE"
echo "  Output dir  : $OUTPUT_DIR"
echo "  GPUs        : $GPUS"

# Check if datasets exist
if [ ! -d "data/processed" ]; then
    echo "Error: datasets not found in data/processed/"
    echo "Run: bash scripts/prepare_datasets.sh"
    exit 1
fi

# Build the training arguments
TRAIN_ARGS=(
    src/train.py
    --model_name_or_path "Qwen/Qwen3-8B"
    --data_dir "./data/processed"
    --output_dir "$OUTPUT_DIR"
    --num_train_epochs "$EPOCHS"
    --per_device_train_batch_size "$BATCH_SIZE"
    --per_device_eval_batch_size 16
    --gradient_accumulation_steps 2
    --learning_rate 2e-4
    --warmup_ratio 0.1
    --weight_decay 0.01
    --max_grad_norm 1.0
    --report_to "wandb"
    --logging_steps 10
    --eval_strategy "steps"
    --eval_steps 100
    --save_strategy "steps"
    --save_steps 100
    --save_total_limit 3
    --load_best_model_at_end
    --bf16
    --gradient_checkpointing
    --ddp_find_unused_parameters false
    --ddp_backend nccl
    --seed 42
)

# Launch — torchrun for multi-GPU, plain python for single
if [ "$GPUS" -gt 1 ]; then
    echo ""
    echo "Launching with torchrun (${GPUS} GPUs)..."
    torchrun --nproc_per_node="$GPUS" "${TRAIN_ARGS[@]}"
else
    echo ""
    echo "Launching single-GPU training..."
    python3 "${TRAIN_ARGS[@]}"
fi

echo "================================================"
echo "✓ Local training complete"
echo "Output: $OUTPUT_DIR"
echo "================================================"
