# Execution Plan

This is the current operational checklist for the repository.

## Phase 1: Environment Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Required for real remote work:

- `TINKER_API_KEY`

Optional but recommended:

- `WANDB_API_KEY`
- `HF_TOKEN`
- `HF_REPO_ID`

## Phase 2: Dataset Preparation

```bash
bash scripts/prepare_datasets.sh
```

Verify:

```bash
python -c "
from datasets import load_from_disk
ds = load_from_disk('data/processed')
print(len(ds['train']), len(ds['validation']), len(ds['test']))
print(ds['train'].column_names)
"
```

The exact split sizes depend on the generated data, but the default flow should
produce an approximately 80/10/10 split with tokenized
`input_ids/attention_mask/labels` columns.

## Phase 3: Baseline Evaluation

```bash
bash scripts/run_pipeline.sh baseline
```

This evaluates the base model with the current structured benchmarks from
`src/evaluate.py`.

## Phase 4: SFT

```bash
bash scripts/run_pipeline.sh sft
```

Artifacts:

- `outputs/sft/checkpoints.jsonl`
- checkpoint directories and sampler metadata under `outputs/sft/`

## Phase 5: GRPO

```bash
bash scripts/run_pipeline.sh grpo
```

Artifacts:

- `outputs/grpo/checkpoints.jsonl`
- checkpoint directories and sampler metadata under `outputs/grpo/`

Note: the canonical pipeline uses the same LoRA rank for SFT and GRPO when GRPO
starts from the SFT checkpoint.

## Phase 6: Comparison

```bash
bash scripts/run_pipeline.sh compare
```

This creates a side-by-side comparison from the current baseline, SFT, and GRPO
checkpoints.

## One-Command Paths

### Full remote pipeline

```bash
bash scripts/run_pipeline.sh all
```

### Lower-cost remote pipeline

```bash
bash scripts/run_pipeline.sh all-final-compare
```

This skips the intermediate SFT and GRPO eval passes and relies on one final
comparison run instead.

### Local validation path

```bash
LOCAL_VALIDATE=true bash scripts/run_pipeline.sh all
```

### Real smoke test

```bash
bash scripts/run_smoke_test.sh
```

## Completion Checklist

- Environment created and dependencies installed
- `.env` configured
- `scripts/prepare_datasets.sh` completed successfully
- Baseline evaluation produced `outputs/eval_baseline.json`
- SFT produced `outputs/sft/checkpoints.jsonl`
- GRPO produced `outputs/grpo/checkpoints.jsonl`
- Comparison produced `outputs/eval_comparison.json`
