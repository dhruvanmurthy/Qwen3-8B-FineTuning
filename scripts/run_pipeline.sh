#!/bin/bash
# Three-stage training & evaluation pipeline
#   Stage 0: Baseline evaluation (zero-shot)
#   Stage 1: SFT training + evaluation
#   Stage 2: GRPO training + evaluation
#   Stage 3: Cross-stage comparison
#
# Usage:
#   bash scripts/run_pipeline.sh           # run everything
#   bash scripts/run_pipeline.sh baseline  # only baseline eval
#   bash scripts/run_pipeline.sh sft       # only SFT stage
#   bash scripts/run_pipeline.sh grpo      # only GRPO stage
#   bash scripts/run_pipeline.sh compare   # only comparison
#
# Multi-GPU:
#   GPUS=4 bash scripts/run_pipeline.sh    # use 4 GPUs

set -e

# Always run from the repo root
cd "$(dirname "$0")/.." 

STAGE="${1:-all}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
SFT_OUTPUT="${SFT_OUTPUT:-./outputs/sft}"
GRPO_OUTPUT="${GRPO_OUTPUT:-./outputs/grpo}"

# GPU detection — default to all visible GPUs
if [ -z "$GPUS" ]; then
    GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
fi

# Helper: launch with torchrun when multi-GPU, plain python otherwise
run_train() {
    if [ "$GPUS" -gt 1 ]; then
        torchrun --nproc_per_node="$GPUS" "$@"
    else
        python3 "$@"
    fi
}

echo "============================================="
echo " Qwen3-8B Tool-Use Fine-Tuning Pipeline"
echo "============================================="
echo "  Base model : $BASE_MODEL"
echo "  SFT output : $SFT_OUTPUT"
echo "  GRPO output: $GRPO_OUTPUT"
echo "  Stage      : $STAGE"
echo "  GPUs       : $GPUS"
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
        --output outputs/eval_baseline.json
fi

# --------------------------------------------------
# Stage 1 — SFT training + evaluation
# --------------------------------------------------
if [[ "$STAGE" == "sft" || "$STAGE" == "all" ]]; then
    echo ""
    echo ">>> Stage 1: SFT Training"
    run_train src/train.py \
        --model_name_or_path "$BASE_MODEL" \
        --output_dir "$SFT_OUTPUT" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 2 \
        --learning_rate 2e-4 \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --eval_strategy steps \
        --eval_steps 250 \
        --save_strategy steps \
        --save_steps 250 \
        --save_total_limit 3 \
        --load_best_model_at_end \
        --bf16 \
        --gradient_checkpointing \
        --ddp_find_unused_parameters false \
        --ddp_backend nccl \
        --logging_steps 10 \
        --report_to wandb \
        --seed 42

    echo ""
    echo ">>> Stage 1: SFT Evaluation"
    python3 src/evaluate.py \
        --mode sft \
        --base-model "$BASE_MODEL" \
        --sft-adapter "$SFT_OUTPUT" \
        --output outputs/eval_sft.json
fi

# --------------------------------------------------
# Stage 2 — GRPO training + evaluation
# --------------------------------------------------
if [[ "$STAGE" == "grpo" || "$STAGE" == "all" ]]; then
    echo ""
    echo ">>> Stage 2: GRPO Training (LoRA r=32, LR 3e-5, batch 128, group 16, 50 steps)"
    run_train src/train_grpo.py \
        --sft-adapter-path "$SFT_OUTPUT" \
        --base-model-name "$BASE_MODEL" \
        --output-dir "$GRPO_OUTPUT" \
        --lora-r 32 \
        --learning-rate 3e-5 \
        --max-steps 50 \
        --per-device-train-batch-size 4 \
        --gradient-accumulation-steps 32 \
        --num-generations 16 \
        --report-to wandb

    echo ""
    echo ">>> Stage 2: GRPO Evaluation"
    python3 src/evaluate.py \
        --mode grpo \
        --base-model "$BASE_MODEL" \
        --sft-adapter "$SFT_OUTPUT" \
        --grpo-adapter "$GRPO_OUTPUT" \
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
        --output outputs/eval_comparison.json
fi

echo ""
echo "============================================="
echo " Pipeline complete!"
echo "============================================="
