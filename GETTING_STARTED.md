# Getting Started

This guide walks through the current local workflow for the Tinker-based
pipeline.

## Prerequisites

- Python 3.10 or 3.11
- Bash for the shell scripts
  - On Windows, use WSL for `scripts/*.sh`
- 30 GB or more of free disk space
- `TINKER_API_KEY` for real training and evaluation

No local GPU is required for the main pipeline. Training and inference run on
Tinker.

## 1. Create an Environment

### PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Bash or WSL

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure `.env`

```bash
cp .env.example .env
```

Fill in the keys you want to use:

```dotenv
HF_TOKEN=hf_xxx
HF_USER=your_hf_username
HF_REPO_ID=qwen3-8b-tool-use-lora
WANDB_API_KEY=xxx
WANDB_ENTITY=your_wandb_entity
WANDB_PROJECT=qwen3-8b-tool-use
TINKER_API_KEY=xxx
```

## 3. Verify the Environment

```bash
python -c "import transformers, datasets, tinker, wandb; print('Core imports OK')"
python -c "import sys; sys.path.insert(0, 'src'); from data_loader import ToolUseDataLoader; from rewards import compute_rewards; print('Project imports OK')"
```

## 4. Prepare Datasets

```bash
bash scripts/prepare_datasets.sh
```

What this does:

- generates synthetic data into `data/raw/synthetic/`
- loads the configured local source from `configs/dataset_config.yaml`
- applies preprocessing and balancing
- saves a raw structured test split to `data/processed/test_raw.jsonl`
- saves tokenized train/validation/test splits to `data/processed/`

Verify the result:

```bash
python -c "
from datasets import load_from_disk
ds = load_from_disk('data/processed')
print('Train:', len(ds['train']))
print('Validation:', len(ds['validation']))
print('Test:', len(ds['test']))
print('Columns:', ds['train'].column_names)
"
```

Expected shape:

- The exact counts vary because generation, deduplication, and balancing are
  data dependent.
- With the default pipeline, expect roughly twelve thousand processed examples
  before splitting and about an 80/10/10 train/validation/test split.
- Tokenized splits should contain `input_ids`, `attention_mask`, and `labels`.

## 5. Run Baseline Evaluation

```bash
python src/evaluate.py \
  --mode baseline \
  --base-model Qwen/Qwen3-8B \
  --output outputs/eval_baseline.json
```

## 6. Run SFT

### Dry-run validation

```bash
python src/train.py \
  --base-model sshleifer/tiny-gpt2 \
  --output-dir outputs/sft_dry_run \
  --dry-run \
  --dry-run-steps 3
```

### Full SFT run

```bash
bash scripts/run_local_training.sh
```

Or call the training script directly:

```bash
python src/train.py \
  --base-model Qwen/Qwen3-8B \
  --synthetic-data-dir ./data/raw/synthetic \
  --output-dir ./outputs/sft \
  --lora-rank 64 \
  --learning-rate 2e-4 \
  --batch-size 8 \
  --num-epochs 3 \
  --max-seq-length 2048
```

## 7. Run GRPO

### Dry-run validation

```bash
python src/train_grpo.py \
  --base-model sshleifer/tiny-gpt2 \
  --sft-checkpoint ./outputs/sft \
  --output-dir ./outputs/grpo_dry_run \
  --dry-run \
  --dry-run-steps 3
```

### Full GRPO run

Use the pipeline script if you want the canonical defaults:

```bash
bash scripts/run_pipeline.sh grpo
```

Or call the training script directly:

```bash
python src/train_grpo.py \
  --base-model Qwen/Qwen3-8B \
  --sft-checkpoint ./outputs/sft \
  --output-dir ./outputs/grpo \
  --lora-rank 64 \
  --learning-rate 4e-5 \
  --batch-size 16 \
  --group-size 8 \
  --max-steps 50
```

If you start GRPO from an SFT checkpoint, keep the LoRA rank aligned with the
SFT run. `scripts/run_pipeline.sh` enforces that by default.

## 8. Compare Stages

```bash
python src/evaluate.py \
  --mode compare \
  --base-model Qwen/Qwen3-8B \
  --sft-output-dir ./outputs/sft \
  --grpo-output-dir ./outputs/grpo \
  --output outputs/eval_comparison.json
```

Or run all stages through the canonical orchestrator:

```bash
bash scripts/run_pipeline.sh all
```

## 9. Use the Smoke Test

```bash
bash scripts/run_smoke_test.sh
```

The smoke test:

- generates a small synthetic dataset
- runs reduced SFT and GRPO jobs
- evaluates a small benchmark subset
- writes smoke outputs under `outputs/smoke_*`

## 10. Optional Hub Uploads

Automatic upload behavior:

- `src/train.py` uploads the SFT adapter when `HF_TOKEN` and `HF_REPO_ID` are set
- `src/train_grpo.py` uploads the GRPO adapter when `HF_TOKEN` and `HF_REPO_ID` are set

Manual helpers:

```bash
python scripts/push_dataset_to_hub.py --repo-id your-user/qwen3-8b-synthetic-tool-use
python scripts/push_model_to_hub.py --repo-id your-user/qwen3-8b-tool-use-grpo
```

## Recommended First Run

If this is your first pass through the repo:

```bash
bash scripts/prepare_datasets.sh
LOCAL_VALIDATE=true bash scripts/run_pipeline.sh all
```

That confirms the local wiring before you spend time or money on a full remote
run.
