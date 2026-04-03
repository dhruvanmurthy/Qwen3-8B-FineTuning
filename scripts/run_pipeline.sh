#!/bin/bash
# Three-stage training & evaluation pipeline (Tinker)
#   Stage 0: Baseline evaluation (zero-shot)
#   Stage 1: SFT training + evaluation
#   Stage 2: GRPO training + evaluation
#   Stage 3: Cross-stage comparison
#
# Training runs on Tinker's remote GPUs — no local GPU required.
# Requires: TINKER_API_KEY environment variable.
# Recommended: WANDB_API_KEY and HF_TOKEN for logging and Hub pushes.
#
# Usage:
#   bash scripts/run_pipeline.sh           # run everything
#   bash scripts/run_pipeline.sh baseline  # only baseline eval
#   bash scripts/run_pipeline.sh sft       # only SFT stage
#   bash scripts/run_pipeline.sh grpo      # only GRPO stage
#   bash scripts/run_pipeline.sh compare   # only comparison

set -e

# Always run from the repo root
cd "$(dirname "$0")/.."

# Load .env file if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

STAGE="${1:-all}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
SFT_OUTPUT="${SFT_OUTPUT:-./outputs/sft}"
GRPO_OUTPUT="${GRPO_OUTPUT:-./outputs/grpo}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen3-8b-tool-use}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
HF_REPO_ID="${HF_REPO_ID:-}"

# Verify Tinker API key
if [ -z "$TINKER_API_KEY" ]; then
    echo "Error: TINKER_API_KEY not set. Get one at https://tinker-console.thinkingmachines.ai/"
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

echo "============================================="
echo " Qwen3-8B Tool-Use Fine-Tuning Pipeline"
echo " (Tinker Remote Training)"
echo "============================================="
echo "  Base model      : $BASE_MODEL"
echo "  SFT output      : $SFT_OUTPUT"
echo "  GRPO output     : $GRPO_OUTPUT"
echo "  Stage           : $STAGE"
echo "  W&B project     : $WANDB_PROJECT"
if [ -n "$WANDB_ENTITY" ]; then echo "  W&B entity      : $WANDB_ENTITY"; fi
if [ -n "$HF_REPO_ID" ]; then echo "  HF repo ID      : $HF_REPO_ID"; fi
echo "============================================="

# --------------------------------------------------
# Stage 0 — Baseline evaluation
# --------------------------------------------------
if [[ "$STAGE" == "baseline" || "$STAGE" == "all" ]]; then
    echo ""
    echo ">>> Stage 0: Baseline Evaluation"
    python3 src/evaluate.py \
        --mode baseline \
        --base-model "$BASE_MODEL" \
        --wandb-project "$WANDB_PROJECT" \
        --output outputs/eval_baseline.json
fi

# --------------------------------------------------
# Stage 1 — SFT training + evaluation
# --------------------------------------------------
if [[ "$STAGE" == "sft" || "$STAGE" == "all" ]]; then
    echo ""
    echo ">>> Stage 1: SFT Training (Tinker)"

    # Generate synthetic data if not present
    if [ ! -d "data/raw/synthetic" ] || [ -z "$(ls data/raw/synthetic/*.jsonl 2>/dev/null)" ]; then
        echo ">>> Generating synthetic training data..."
        python3 scripts/generate_synthetic.py
    fi

    python3 src/train.py \
        --base-model "$BASE_MODEL" \
        --synthetic-data-dir "./data/raw/synthetic" \
        --output-dir "$SFT_OUTPUT" \
        --lora-rank 64 \
        --learning-rate 2e-4 \
        --batch-size 8 \
        --num-epochs 3 \
        --max-seq-length 2048 \
        --logging-steps 10 \
        --save-steps 100 \
        --wandb-project "$WANDB_PROJECT" \
        ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
        --wandb-run-name "sft-${BASE_MODEL##*/}" \
        ${HF_REPO_ID:+--hf-repo-id "${HF_REPO_ID}-sft"}

    echo ""
    echo ">>> Stage 1: SFT Evaluation"
    python3 src/evaluate.py \
        --mode sft \
        --base-model "$BASE_MODEL" \
        --sft-adapter "$SFT_OUTPUT" \
        --wandb-project "$WANDB_PROJECT" \
        --output outputs/eval_sft.json
fi

# --------------------------------------------------
# Stage 2 — GRPO training + evaluation
# --------------------------------------------------
if [[ "$STAGE" == "grpo" || "$STAGE" == "all" ]]; then
    echo ""
    echo ">>> Stage 2: GRPO Training (Tinker, LoRA r=32, LR 4e-5, group 8, 50 steps)"
    python3 src/train_grpo.py \
        --base-model "$BASE_MODEL" \
        --sft-checkpoint "$SFT_OUTPUT" \
        --output-dir "$GRPO_OUTPUT" \
        --lora-rank 32 \
        --learning-rate 4e-5 \
        --batch-size 16 \
        --group-size 8 \
        --max-steps 50 \
        --save-steps 10 \
        --wandb-project "$WANDB_PROJECT" \
        ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
        --wandb-run-name "grpo-${BASE_MODEL##*/}" \
        ${HF_REPO_ID:+--hf-repo-id "${HF_REPO_ID}-grpo"}

    echo ""
    echo ">>> Stage 2: GRPO Evaluation"
    python3 src/evaluate.py \
        --mode grpo \
        --base-model "$BASE_MODEL" \
        --sft-adapter "$SFT_OUTPUT" \
        --grpo-adapter "$GRPO_OUTPUT" \
        --wandb-project "$WANDB_PROJECT" \
        --output outputs/eval_grpo.json
fi

# --------------------------------------------------
# Stage 3 — Cross-stage comparison
# --------------------------------------------------
if [[ "$STAGE" == "compare" || "$STAGE" == "all" ]]; then
    echo ""
    echo ">>> Stage 3: Cross-Stage Comparison"
    python3 src/evaluate.py \
        --mode compare \
        --base-model "$BASE_MODEL" \
        --sft-adapter "$SFT_OUTPUT" \
        --grpo-adapter "$GRPO_OUTPUT" \
        --wandb-project "$WANDB_PROJECT" \
        --output outputs/eval_comparison.json
fi

echo ""
echo "============================================="
echo " Pipeline complete!"
echo "============================================="
