#!/bin/bash
# Three-stage evaluation: baseline vs SFT vs GRPO

MODE="${1:-all}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
SFT_ADAPTER="${SFT_ADAPTER:-./outputs/sft}"
GRPO_ADAPTER="${GRPO_ADAPTER:-./outputs/grpo}"

python src/evaluate.py \
    --mode "$MODE" \
    --base-model "$BASE_MODEL" \
    --sft-adapter "$SFT_ADAPTER" \
    --grpo-adapter "$GRPO_ADAPTER" \
    --output outputs/evaluation_results.json
