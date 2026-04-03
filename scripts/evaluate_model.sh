#!/bin/bash
# Three-stage evaluation: baseline vs SFT vs GRPO

set -e

# Always run from the repo root
cd "$(dirname "$0")/.."  

# Load .env file if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

MODE="${1:-all}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
SFT_ADAPTER="${SFT_ADAPTER:-./outputs/sft}"
GRPO_ADAPTER="${GRPO_ADAPTER:-./outputs/grpo}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen3-8b-tool-use}"

python3 src/evaluate.py \
    --mode "$MODE" \
    --base-model "$BASE_MODEL" \
    --sft-adapter "$SFT_ADAPTER" \
    --grpo-adapter "$GRPO_ADAPTER" \
    --wandb-project "$WANDB_PROJECT" \
    --output outputs/evaluation_results.json
