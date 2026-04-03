#!/bin/bash
# End-to-end smoke test using REAL Tinker training + inference.
# Runs the full pipeline with minimal data to catch integration issues
# before committing to a full training run.
#
# Requires: TINKER_API_KEY, WANDB_API_KEY (optional)
# Time: ~10-20 minutes total
#
# Usage:
#   bash scripts/run_smoke_test.sh           # run all stages
#   bash scripts/run_smoke_test.sh baseline  # only baseline
#   bash scripts/run_smoke_test.sh sft       # only SFT
#   bash scripts/run_smoke_test.sh grpo      # only GRPO (needs SFT first)

set -e

cd "$(dirname "$0")/.."

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

STAGE="${1:-all}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
SMOKE_DATA="./data/raw/synthetic_smoke"
SFT_OUT="./outputs/smoke_sft"
GRPO_OUT="./outputs/smoke_grpo"
EVAL_SAMPLES=20   # tiny eval set
RESET_SMOKE_OUTPUTS="${RESET_SMOKE_OUTPUTS:-1}"

case "$STAGE" in
    all|baseline|sft|grpo) ;;
    *)
        echo "Error: invalid stage '$STAGE'. Use one of: all, baseline, sft, grpo"
        exit 1
        ;;
esac

# Don't push to HF Hub during smoke test
unset HF_REPO_ID

if [ -z "$TINKER_API_KEY" ]; then
    echo "Error: TINKER_API_KEY not set. This smoke test uses real Tinker inference."
    exit 1
fi

echo "============================================="

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

_require_sampler_path() {
    local out_dir="$1"
    local stage_label="$2"
    local sampler
    sampler=$(_last_sampler_path "$out_dir")
    [ -n "$sampler" ] || _fail "No sampler_path found for $stage_label in $out_dir/checkpoints.jsonl"
    echo "$sampler"
}

_require_metric() {
    local json_path="$1"
    local metric_key="$2"
    local stage_key="$3"
    python3 - "$json_path" "$metric_key" "$stage_key" <<'PYEOF'
import json, sys
path, metric, stage = sys.argv[1:4]
with open(path, "r", encoding="utf-8") as f:
    obj = json.load(f)
if stage not in obj:
    raise SystemExit(f"[FAIL-FAST] Missing stage key '{stage}' in {path}")
if metric not in obj[stage]:
    raise SystemExit(f"[FAIL-FAST] Missing metric '{metric}' in {path}")
print(f"[OK] {stage}.{metric}={obj[stage][metric]}")
PYEOF
}
echo " Smoke Test (Real Tinker, Minimal Data)"
echo "============================================="
echo "  Base model    : $BASE_MODEL"
echo "  Eval samples  : $EVAL_SAMPLES"
echo "  SFT output    : $SFT_OUT"
echo "  GRPO output   : $GRPO_OUT"
echo "  Reset outputs : $RESET_SMOKE_OUTPUTS"
echo "  Stage         : $STAGE"
echo "============================================="

# Helper
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

# Generate tiny synthetic dataset
mkdir -p "$SMOKE_DATA"
if [ ! -f "$SMOKE_DATA/synthetic_single.jsonl" ]; then
    echo ""
    echo ">>> Generating tiny synthetic dataset (128 samples)..."
    python3 scripts/generate_synthetic.py --num-samples 128 --output-dir "$SMOKE_DATA"
fi

# --------------------------------------------------
# Stage 0 — Baseline eval (real Tinker inference)
# --------------------------------------------------
if [[ "$STAGE" == "baseline" || "$STAGE" == "all" ]]; then
    echo ""
    echo ">>> Stage 0: Baseline Evaluation ($EVAL_SAMPLES samples)"
    python3 src/evaluate.py \
        --mode baseline \
        --base-model "$BASE_MODEL" \
        --max-samples "$EVAL_SAMPLES" \
        --output outputs/smoke_eval_baseline.json
    _require_file "outputs/smoke_eval_baseline.json" "Baseline evaluation did not produce output"
    _require_metric "outputs/smoke_eval_baseline.json" "tool_selection_accuracy" "baseline"
    echo ">>> Baseline results:"
    python3 -m json.tool outputs/smoke_eval_baseline.json
fi

# --------------------------------------------------
# Stage 1 — SFT (real Tinker training, 1 epoch, tiny data)
# --------------------------------------------------
if [[ "$STAGE" == "sft" || "$STAGE" == "all" ]]; then
    if [[ "$RESET_SMOKE_OUTPUTS" == "1" ]]; then
        echo ">>> Resetting SFT smoke output: $SFT_OUT"
        rm -rf "$SFT_OUT"
    fi

    echo ""
    echo ">>> Stage 1: SFT Training (1 epoch, tiny data)"
    python3 src/train.py \
        --base-model "$BASE_MODEL" \
        --synthetic-data-dir "$SMOKE_DATA" \
        --output-dir "$SFT_OUT" \
        --lora-rank 16 \
        --learning-rate 2e-4 \
        --batch-size 4 \
        --num-epochs 1 \
        --max-seq-length 1024 \
        --logging-steps 1 \
        --save-steps 5 \
        --seed 42

    _require_file "$SFT_OUT/checkpoints.jsonl" "SFT training did not produce checkpoints.jsonl"

    echo ""
    echo ">>> Stage 1: SFT Evaluation"
    SFT_SAMPLER=$(_require_sampler_path "$SFT_OUT" "SFT")
    echo "  SFT sampler: $SFT_SAMPLER"
    python3 src/evaluate.py \
        --mode sft \
        --base-model "$BASE_MODEL" \
        --sft-sampler-path "$SFT_SAMPLER" \
        --max-samples "$EVAL_SAMPLES" \
        --output outputs/smoke_eval_sft.json
    _require_file "outputs/smoke_eval_sft.json" "SFT evaluation did not produce output"
    _require_metric "outputs/smoke_eval_sft.json" "tool_selection_accuracy" "sft"
    echo ">>> SFT results:"
    python3 -m json.tool outputs/smoke_eval_sft.json
fi

# --------------------------------------------------
# Stage 2 — GRPO (real Tinker training, 3 steps)
# --------------------------------------------------
if [[ "$STAGE" == "grpo" || "$STAGE" == "all" ]]; then
    if [[ "$RESET_SMOKE_OUTPUTS" == "1" ]]; then
        echo ">>> Resetting GRPO smoke output: $GRPO_OUT"
        rm -rf "$GRPO_OUT"
    fi

    _require_file "$SFT_OUT/checkpoints.jsonl" "GRPO requires SFT output; run 'bash scripts/run_smoke_test.sh sft' first"

    echo ""
    echo ">>> Stage 2: GRPO Training (3 steps, group-size 4)"
    python3 src/train_grpo.py \
        --base-model "$BASE_MODEL" \
        --sft-checkpoint "$SFT_OUT" \
        --output-dir "$GRPO_OUT" \
        --lora-rank 16 \
        --learning-rate 4e-5 \
        --batch-size 4 \
        --group-size 4 \
        --max-steps 3 \
        --max-completion-length 256 \
        --save-steps 3 \
        --log-samples-every 1 \
        --seed 42

    _require_file "$GRPO_OUT/checkpoints.jsonl" "GRPO training did not produce checkpoints.jsonl"

    echo ""
    echo ">>> Stage 2: GRPO Evaluation"
    GRPO_SAMPLER=$(_require_sampler_path "$GRPO_OUT" "GRPO")
    echo "  GRPO sampler: $GRPO_SAMPLER"
    python3 src/evaluate.py \
        --mode grpo \
        --base-model "$BASE_MODEL" \
        --grpo-sampler-path "$GRPO_SAMPLER" \
        --max-samples "$EVAL_SAMPLES" \
        --output outputs/smoke_eval_grpo.json
    _require_file "outputs/smoke_eval_grpo.json" "GRPO evaluation did not produce output"
    _require_metric "outputs/smoke_eval_grpo.json" "tool_selection_accuracy" "grpo"
    echo ">>> GRPO results:"
    python3 -m json.tool outputs/smoke_eval_grpo.json
fi

# --------------------------------------------------
# Stage 3 — Comparison
# --------------------------------------------------
if [[ "$STAGE" == "all" ]]; then
    echo ""
    echo ">>> Stage 3: Cross-Stage Comparison"
    SFT_SAMPLER=$(_last_sampler_path "$SFT_OUT")
    GRPO_SAMPLER=$(_last_sampler_path "$GRPO_OUT")
    python3 src/evaluate.py \
        --mode compare \
        --base-model "$BASE_MODEL" \
        ${SFT_SAMPLER:+--sft-sampler-path "$SFT_SAMPLER"} \
        ${GRPO_SAMPLER:+--grpo-sampler-path "$GRPO_SAMPLER"} \
        --max-samples "$EVAL_SAMPLES" \
        --output outputs/smoke_eval_comparison.json
    _require_file "outputs/smoke_eval_comparison.json" "Comparison stage did not produce output"
fi

echo ""
echo "============================================="
echo " Smoke test complete!"
echo "============================================="
echo " Results:"
echo "   Baseline : outputs/smoke_eval_baseline.json"
echo "   SFT      : outputs/smoke_eval_sft.json"
echo "   GRPO     : outputs/smoke_eval_grpo.json"
echo "   Compare  : outputs/smoke_eval_comparison.json"
echo "============================================="
