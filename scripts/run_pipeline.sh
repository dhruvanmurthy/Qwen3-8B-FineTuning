#!/bin/bash
# Three-stage training & evaluation pipeline (Tinker)
#   Stage 0: Baseline evaluation (zero-shot)
#   Stage 1: SFT training + evaluation
#   Stage 2: GRPO training + evaluation
#   Stage 3: Cross-stage comparison
#
# Training runs on Tinker's remote GPUs — no local GPU required.
# Requires: TINKER_API_KEY environment variable.
# Recommended: WANDB_API_KEY and HF_TOKEN for logging and Hub uploads.
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
EVAL_SAMPLES="${EVAL_SAMPLES:-1000}"
BENCHMARK_FILTER="${BENCHMARK_FILTER:-}"

SFT_LORA_RANK="${SFT_LORA_RANK:-64}"
GRPO_LORA_RANK="${GRPO_LORA_RANK:-$SFT_LORA_RANK}"

WANDB_PROJECT="${WANDB_PROJECT:-qwen3-8b-tool-use}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
HF_REPO_ID="${HF_REPO_ID:-}"
LOCAL_VALIDATE="${LOCAL_VALIDATE:-false}"
DRY_RUN_ARGS=""

_fail() {
    echo ""
    echo "[FAIL-FAST] $1"
    exit 1
}

_require_file() {
    local file_path="$1"
    local msg="$2"
    [ -f "$file_path" ] || _fail "$msg (missing: $file_path)"
}

# Verify Tinker API key unless local validation mode is enabled
if [ "$LOCAL_VALIDATE" != "true" ] && [ -z "$TINKER_API_KEY" ]; then
    echo "Error: TINKER_API_KEY not set. Get one at https://tinker-console.thinkingmachines.ai/"
    exit 1
fi

if [ "$LOCAL_VALIDATE" = "true" ]; then
    echo "Info: LOCAL_VALIDATE=true, training stages will run in --dry-run mode."
    if [ "$BASE_MODEL" = "Qwen/Qwen3-8B" ]; then
        BASE_MODEL="sshleifer/tiny-gpt2"
    fi
    HF_REPO_ID=""
    DRY_RUN_ARGS="--dry-run --dry-run-steps 3"
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

# --------------------------------------------------
# Helper: extract last sampler_path from checkpoints.jsonl
# --------------------------------------------------
_last_sampler_path() {
    local log_dir="$1"
    python3 - "$log_dir" <<'PYEOF'
import json, sys
from pathlib import Path
ckpt_file = Path(sys.argv[1]) / "checkpoints.jsonl"
if not ckpt_file.exists():
    sys.exit(0)
last = ""
with open(ckpt_file) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            if e.get("sampler_path"):
                last = e["sampler_path"]
        except Exception:
            pass
print(last)
PYEOF
}

echo "============================================="
echo " Qwen3-8B Tool-Use Fine-Tuning Pipeline"
if [ "$LOCAL_VALIDATE" = "true" ]; then
    echo " (Local Validation Mode)"
else
    echo " (Tinker Remote Training)"
fi
echo "============================================="
echo "  Base model      : $BASE_MODEL"
echo "  SFT output      : $SFT_OUTPUT"
echo "  GRPO output     : $GRPO_OUTPUT"
echo "  Eval samples    : $EVAL_SAMPLES"
echo "  SFT LoRA rank   : $SFT_LORA_RANK"
echo "  GRPO LoRA rank  : $GRPO_LORA_RANK"
echo "  Stage           : $STAGE"
echo "  W&B project     : $WANDB_PROJECT"
if [ -n "$WANDB_ENTITY" ]; then echo "  W&B entity      : $WANDB_ENTITY"; fi
if [ -n "$HF_REPO_ID" ]; then echo "  HF repo ID      : $HF_REPO_ID"; fi
echo "============================================="

# --------------------------------------------------
# Pre-flight: ensure synthetic data exists
# (needed by all eval stages, not just SFT)
# --------------------------------------------------
if [ ! -d "data/raw/synthetic" ] || [ -z "$(ls data/raw/synthetic/*.jsonl 2>/dev/null)" ]; then
    echo ">>> Generating synthetic training data..."
    python3 scripts/generate_synthetic.py
fi

# --------------------------------------------------
# Stage 0 — Baseline evaluation
# --------------------------------------------------
if [[ "$STAGE" == "baseline" || "$STAGE" == "all" ]]; then
    echo ""
    if [ "$LOCAL_VALIDATE" = "true" ]; then
        echo ">>> Stage 0: Baseline Evaluation skipped (LOCAL_VALIDATE — no Tinker inference)"
    else
        echo ">>> Stage 0: Baseline Evaluation"
        python3 src/evaluate.py \
            --mode baseline \
            --base-model "$BASE_MODEL" \
            --max-samples "$EVAL_SAMPLES" \
            --output outputs/eval_baseline.json \
            ${BENCHMARK_FILTER:+--benchmarks $BENCHMARK_FILTER}
    fi
fi

# --------------------------------------------------
# Stage 1 — SFT training + evaluation
# --------------------------------------------------
if [[ "$STAGE" == "sft" || "$STAGE" == "all" ]]; then
    echo ""
    echo ">>> Stage 1: SFT Training (Tinker)"

    python3 src/train.py \
        --base-model "$BASE_MODEL" \
        --synthetic-data-dir "./data/raw/synthetic" \
        --output-dir "$SFT_OUTPUT" \
        --lora-rank "$SFT_LORA_RANK" \
        --learning-rate 2e-4 \
        --batch-size 8 \
        --num-epochs 3 \
        --max-seq-length 2048 \
        --logging-steps 10 \
        --save-steps 25 \
        $DRY_RUN_ARGS

    _require_file "$SFT_OUTPUT/checkpoints.jsonl" "SFT training did not produce checkpoints.jsonl"

    if [ "$LOCAL_VALIDATE" = "true" ]; then
        echo ">>> Stage 1 evaluation skipped in local validation mode"
    else
        echo ""
        echo ">>> Stage 1: SFT Evaluation"
        SFT_SAMPLER=$(_last_sampler_path "$SFT_OUTPUT")
        if [ -z "$SFT_SAMPLER" ]; then
            echo "Warning: No sampler_path found in $SFT_OUTPUT/checkpoints.jsonl — skipping SFT eval."
        else
            python3 src/evaluate.py \
                --mode sft \
                --base-model "$BASE_MODEL" \
                --sft-sampler-path "$SFT_SAMPLER" \
                --max-samples "$EVAL_SAMPLES" \
                --output outputs/eval_sft.json \
                ${BENCHMARK_FILTER:+--benchmarks $BENCHMARK_FILTER}
        fi
    fi
fi

# --------------------------------------------------
# Stage 2 — GRPO training + evaluation
# --------------------------------------------------
if [[ "$STAGE" == "grpo" || "$STAGE" == "all" ]]; then
    _require_file "$SFT_OUTPUT/checkpoints.jsonl" "GRPO requires SFT output; run SFT stage first"

    if [[ "$GRPO_LORA_RANK" != "$SFT_LORA_RANK" ]]; then
        _fail "LoRA rank mismatch: SFT_LORA_RANK=$SFT_LORA_RANK but GRPO_LORA_RANK=$GRPO_LORA_RANK. With latest Tinker checkpoint loading, these must match."
    fi

    echo ""
    echo ">>> Stage 2: GRPO Training (Tinker, LoRA r=$GRPO_LORA_RANK, LR 4e-5, group 8, 50 steps)"
    python3 src/train_grpo.py \
        --base-model "$BASE_MODEL" \
        --sft-checkpoint "$SFT_OUTPUT" \
        --output-dir "$GRPO_OUTPUT" \
        --lora-rank "$GRPO_LORA_RANK" \
        --learning-rate 4e-5 \
        --batch-size 16 \
        --group-size 8 \
        --max-steps 50 \
        --save-steps 10 \
        $DRY_RUN_ARGS

    _require_file "$GRPO_OUTPUT/checkpoints.jsonl" "GRPO training did not produce checkpoints.jsonl"

    if [ "$LOCAL_VALIDATE" = "true" ]; then
        echo ">>> Stage 2 evaluation skipped in local validation mode"
    else
        echo ""
        echo ">>> Stage 2: GRPO Evaluation"
        GRPO_SAMPLER=$(_last_sampler_path "$GRPO_OUTPUT")
        if [ -z "$GRPO_SAMPLER" ]; then
            echo "Warning: No sampler_path found in $GRPO_OUTPUT/checkpoints.jsonl — skipping GRPO eval."
        else
            python3 src/evaluate.py \
                --mode grpo \
                --base-model "$BASE_MODEL" \
                --grpo-sampler-path "$GRPO_SAMPLER" \
                --max-samples "$EVAL_SAMPLES" \
                --output outputs/eval_grpo.json \
                ${BENCHMARK_FILTER:+--benchmarks $BENCHMARK_FILTER}
        fi
    fi
fi

# --------------------------------------------------
# Stage 3 — Cross-stage comparison
# --------------------------------------------------
if [[ "$STAGE" == "compare" || "$STAGE" == "all" ]]; then
    if [ "$LOCAL_VALIDATE" = "true" ]; then
        echo ""
        echo ">>> Stage 3: Cross-stage comparison skipped in local validation mode"
    else
        echo ""
        echo ">>> Stage 3: Cross-Stage Comparison"
        SFT_SAMPLER=$(_last_sampler_path "$SFT_OUTPUT")
        GRPO_SAMPLER=$(_last_sampler_path "$GRPO_OUTPUT")
        python3 src/evaluate.py \
            --mode compare \
            --base-model "$BASE_MODEL" \
            ${SFT_SAMPLER:+--sft-sampler-path "$SFT_SAMPLER"} \
            ${GRPO_SAMPLER:+--grpo-sampler-path "$GRPO_SAMPLER"} \
            --max-samples "$EVAL_SAMPLES" \
            --output outputs/eval_comparison.json \
            ${BENCHMARK_FILTER:+--benchmarks $BENCHMARK_FILTER}
    fi
fi

echo ""
echo "============================================="
echo " Pipeline complete!"
echo "============================================="
