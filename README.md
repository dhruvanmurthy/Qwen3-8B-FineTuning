# Qwen3-8B Fine-Tuning for Tool Use

This repository fine-tunes `Qwen/Qwen3-8B` for structured tool use with a
Tinker-based pipeline:

1. Baseline evaluation
2. SFT training with LoRA
3. GRPO training on top of the SFT checkpoint
4. Cross-stage evaluation and comparison

The current source of truth is the implementation in:

- `scripts/run_pipeline.sh`
- `src/train.py`
- `src/train_grpo.py`
- `src/evaluate.py`

## What Is In Scope

- Synthetic tool-use data generation
- Dataset preparation and tokenization
- Remote training and evaluation through Tinker
- W&B logging
- Optional Hugging Face Hub uploads for SFT and GRPO adapters

## Repository Layout

```text
Qwen3-8B-FineTuning/
|- README.md
|- GETTING_STARTED.md
|- EXECUTION_PLAN.md
|- CONTRIBUTING.md
|- MODEL_CARD.md
|- DATASET_CARD.md
|- configs/
|  \- dataset_config.yaml
|- docs/
|  |- TRAINING_PLAN.md
|  |- EVALUATION.md
|  |- DATASET_STRATEGY.md
|  \- TROUBLESHOOTING.md
|- scripts/
|  |- prepare_datasets.sh
|  |- run_pipeline.sh
|  |- run_local_training.sh
|  |- run_smoke_test.sh
|  |- evaluate_model.sh
|  |- generate_synthetic.py
|  |- push_dataset_to_hub.py
|  \- push_model_to_hub.py
|- src/
|  |- data_loader.py
|  |- constants.py
|  |- rewards.py
|  |- train.py
|  |- train_grpo.py
|  \- evaluate.py
\- .github/workflows/
   |- test.yml
   \- release.yml
```

## Quick Start

The shell scripts are written for Bash. On Windows, use WSL or another Bash
environment to run the `scripts/*.sh` entrypoints.

```bash
pip install -r requirements.txt
cp .env.example .env

# Generate data and prepare tokenized splits
bash scripts/prepare_datasets.sh

# Run the full remote pipeline
bash scripts/run_pipeline.sh all
```

Useful stage-specific commands:

```bash
bash scripts/run_pipeline.sh baseline
bash scripts/run_pipeline.sh sft
bash scripts/run_pipeline.sh grpo
bash scripts/run_pipeline.sh compare
bash scripts/run_pipeline.sh all-final-compare
```

## Local Validation Without Remote Training

If you want to validate wiring without spending Tinker time, the canonical
pipeline supports a dry-run mode:

```bash
LOCAL_VALIDATE=true bash scripts/run_pipeline.sh all
```

This skips remote inference and training, switches the default base model to
`sshleifer/tiny-gpt2`, and validates the local pipeline structure.

If you want the cheapest full orchestration path on Tinker, use:

```bash
bash scripts/run_pipeline.sh all-final-compare
```

That trains SFT and GRPO, skips the intermediate per-stage eval passes, and
does one final comparison pass at the end.

## Smoke Test

For a smaller real end-to-end integration run against Tinker:

```bash
bash scripts/run_smoke_test.sh
```

The smoke test uses a reduced synthetic dataset and evaluates a small benchmark
subset focused on output shape and multi-step chaining.

## Outputs

The current pipeline writes to:

- `outputs/sft/` for SFT checkpoints
- `outputs/grpo/` for GRPO checkpoints
- `outputs/eval_*.json` for evaluation results
- `data/processed/` for tokenized datasets
- `data/processed/test_raw.jsonl` for structured evaluation examples

## Documentation

- [GETTING_STARTED.md](GETTING_STARTED.md): local setup and first run
- [EXECUTION_PLAN.md](EXECUTION_PLAN.md): phase-by-phase execution checklist
- [docs/TRAINING_PLAN.md](docs/TRAINING_PLAN.md): current training flow and defaults
- [docs/EVALUATION.md](docs/EVALUATION.md): benchmarks, commands, and outputs
- [docs/DATASET_STRATEGY.md](docs/DATASET_STRATEGY.md): data generation and preparation
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md): common issues in the current stack

## Notes

- Real training and evaluation require `TINKER_API_KEY`.
- W&B logging is enabled when `WANDB_API_KEY` is present and otherwise runs in
  disabled mode.
- Hugging Face uploads are attempted only when both `HF_TOKEN` and
  `HF_REPO_ID` are configured.
- The canonical GRPO path starts from an SFT checkpoint and should use the same
  LoRA rank as the SFT stage when resuming that checkpoint.
