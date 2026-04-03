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
SFT_OUTPUT="${SFT_OUTPUT:-./outputs/sft}"
GRPO_OUTPUT="${GRPO_OUTPUT:-./outputs/grpo}"

# Extract last sampler_path from checkpoints.jsonl
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

SFT_SAMPLER=$(_last_sampler_path "$SFT_OUTPUT")
GRPO_SAMPLER=$(_last_sampler_path "$GRPO_OUTPUT")

python3 src/evaluate.py \
    --mode "$MODE" \
    --base-model "$BASE_MODEL" \
    ${SFT_SAMPLER:+--sft-sampler-path "$SFT_SAMPLER"} \
    ${GRPO_SAMPLER:+--grpo-sampler-path "$GRPO_SAMPLER"} \
    --output outputs/evaluation_results.json
