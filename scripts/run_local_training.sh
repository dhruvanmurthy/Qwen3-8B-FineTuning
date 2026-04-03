#!/bin/bash
# Local training script using Tinker remote GPUs
#
# No local GPU required — training runs on Tinker's infrastructure.
# Requires: TINKER_API_KEY environment variable.
# Recommended: WANDB_API_KEY and HF_TOKEN for logging and Hub pushes.
#
# Usage:
#   bash scripts/run_local_training.sh

set -e

# Always run from the repo root
cd "$(dirname "$0")/.."  

# Load .env file if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

echo "================================================"
echo "Tinker SFT Training Script"
echo "================================================"

# Configuration
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LORA_RANK="${LORA_RANK:-64}"
LR="${LR:-2e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/local_test}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen3-8b-tool-use}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
HF_REPO_ID="${HF_REPO_ID:-}"

# Verify Tinker API key
if [ -z "$TINKER_API_KEY" ]; then
    echo "Error: TINKER_API_KEY not set."
    echo "Get one at https://tinker-console.thinkingmachines.ai/"
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: WANDB_API_KEY not set. W&B logging will run in disabled mode."
fi

if [ -n "$HF_REPO_ID" ] && [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_REPO_ID set but HF_TOKEN is missing. Hub upload will be skipped."
fi

if [ -n "$HF_TOKEN" ] && [ -z "$HF_REPO_ID" ]; then
    echo "Warning: HF_TOKEN is set but HF_REPO_ID is empty. Hub upload will be skipped."
fi

echo "Configuration:"
echo "  Base model      : $BASE_MODEL"
echo "  Epochs          : $EPOCHS"
echo "  Batch size      : $BATCH_SIZE"
echo "  LoRA rank       : $LORA_RANK"
echo "  LR              : $LR"
echo "  Output dir      : $OUTPUT_DIR"
echo "  W&B project     : $WANDB_PROJECT"
if [ -n "$WANDB_ENTITY" ]; then echo "  W&B entity      : $WANDB_ENTITY"; fi
if [ -n "$HF_REPO_ID" ]; then echo "  HF repo ID      : $HF_REPO_ID"; fi

# Check if synthetic data exists
if [ ! -d "data/raw/synthetic" ] || [ -z "$(ls data/raw/synthetic/*.jsonl 2>/dev/null)" ]; then
    echo ""
    echo "Synthetic data not found. Generating..."
    python3 scripts/generate_synthetic.py
fi

echo ""
echo "Launching Tinker SFT training..."
python3 src/train.py \
    --base-model "$BASE_MODEL" \
    --synthetic-data-dir "./data/raw/synthetic" \
    --output-dir "$OUTPUT_DIR" \
    --lora-rank "$LORA_RANK" \
    --learning-rate "$LR" \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$EPOCHS" \
    --max-seq-length 2048 \
    --logging-steps 10 \
    --save-steps 100 \
    --seed 42 \
    --wandb-project "$WANDB_PROJECT" \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    ${HF_REPO_ID:+--hf-repo-id "$HF_REPO_ID"}

echo "================================================"
echo "✓ Local training complete"
echo "Output: $OUTPUT_DIR"
echo "================================================"
