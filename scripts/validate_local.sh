#!/bin/bash
# Local zero-cost smoke validation for training/eval scripts.
# Does not call Tinker training APIs.

set -e

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

echo "============================================="
echo " Local Script Validation (No Tinker Cost)"
echo "============================================="

# Disable remote side effects for validation.
unset TINKER_API_KEY
unset HF_REPO_ID

BASE_MODEL="${BASE_MODEL:-sshleifer/tiny-gpt2}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen3-8b-tool-use-local-validate}"
SMOKE_DATA_DIR="./data/raw/synthetic_smoke"
SFT_OUT="./outputs/smoke_sft"
GRPO_OUT="./outputs/smoke_grpo"

mkdir -p "$SMOKE_DATA_DIR" "$SFT_OUT" "$GRPO_OUT"

echo "1) Generating tiny synthetic dataset..."
python3 scripts/generate_synthetic.py --num-samples 64 --output-dir "$SMOKE_DATA_DIR"

echo "2) Validating SFT script with --dry-run..."
python3 src/train.py \
  --base-model "$BASE_MODEL" \
  --synthetic-data-dir "$SMOKE_DATA_DIR" \
  --output-dir "$SFT_OUT" \
  --num-epochs 1 \
  --batch-size 2 \
  --max-seq-length 256 \
  --logging-steps 1 \
  --save-steps 0 \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "sft-local-dryrun" \
  --dry-run \
  --dry-run-steps 3

echo "3) Validating GRPO script with --dry-run..."
python3 src/train_grpo.py \
  --base-model "$BASE_MODEL" \
  --sft-checkpoint "$SFT_OUT" \
  --output-dir "$GRPO_OUT" \
  --batch-size 2 \
  --group-size 2 \
  --max-steps 3 \
  --save-steps 0 \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "grpo-local-dryrun" \
  --dry-run \
  --dry-run-steps 3

echo "4) Validating baseline evaluation with tiny model..."
python3 src/evaluate.py \
  --mode baseline \
  --base-model "$BASE_MODEL" \
  --max-samples 16 \
  --wandb-project "$WANDB_PROJECT" \
  --output outputs/eval_local_smoke.json

echo "============================================="
echo " Local validation complete"
echo "  SFT summary  : $SFT_OUT/dry_run_summary.json"
echo "  GRPO summary : $GRPO_OUT/dry_run_summary.json"
echo "  Eval output  : outputs/eval_local_smoke.json"
echo "============================================="
