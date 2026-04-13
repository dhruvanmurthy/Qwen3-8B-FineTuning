# Training Plan

This document describes the current training implementation in the repository.
The canonical code path is the Tinker-based stack in `src/train.py`,
`src/train_grpo.py`, and `scripts/run_pipeline.sh`.

## Training Stages

1. Baseline evaluation of `Qwen/Qwen3-8B`
2. SFT with LoRA on synthetic tool-use conversations
3. GRPO on top of the SFT checkpoint

## Stage 1: SFT

Entry points:

- `src/train.py`
- `scripts/run_local_training.sh`
- `scripts/run_pipeline.sh sft`

Current canonical defaults in the pipeline script:

- base model: `Qwen/Qwen3-8B`
- LoRA rank: `64`
- learning rate: `2e-4`
- batch size: `8`
- epochs: `3`
- max sequence length: `2048`

Training data source:

- synthetic conversations loaded from `data/raw/synthetic/`
- formatted into system/user/assistant conversations
- rendered through the Tinker renderer selected for the base model

Outputs:

- checkpoints and sampler metadata under `outputs/sft/`
- optional SFT adapter upload to Hugging Face Hub

## Stage 2: GRPO

Entry points:

- `src/train_grpo.py`
- `scripts/run_pipeline.sh grpo`

Current canonical defaults in the pipeline script:

- base model: `Qwen/Qwen3-8B`
- starting checkpoint: latest valid checkpoint under `outputs/sft/`
- LoRA rank: matches the SFT rank used by the pipeline
- learning rate: `4e-5`
- batch size: `16`
- group size: `8`
- max steps: `50`
- max completion length: `512`

The pipeline treats matching LoRA rank as required when GRPO resumes from the
SFT checkpoint. This is intentional and enforced in `scripts/run_pipeline.sh`.

Outputs:

- checkpoints and sampler metadata under `outputs/grpo/`
- `grpo_metrics.json`
- optional GRPO adapter upload to Hugging Face Hub

## Reward Signals

`src/rewards.py` is the live reward implementation used by GRPO.

Current composite reward behavior:

- `schema_validation_reward`: binary validity check for the tool-call structure
- `tool_name_reward`: binary exact tool-name match
- `argument_f1_reward`: partial credit on argument key/value overlap
- `full_chain_reward`: binary exact chain match for multi-step examples
- `chain_partial_reward`: partial credit for ordered chain overlap

The reward mix is fully programmatic. There is no learned reward model in the
current implementation.

## Data Flow

### SFT data flow

1. Generate or refresh synthetic JSONL files
2. Load examples from `data/raw/synthetic/`
3. Convert them into chat conversations
4. Render them into Tinker datums
5. Train LoRA weights remotely through Tinker

### GRPO data flow

1. Load structured examples through `ToolUseDataLoader`
2. Build prompt metadata directly inside `train_grpo.py`
3. Sample groups of completions on Tinker
4. Score those completions locally with `src/rewards.py`
5. Apply GRPO-style updates through the Tinker training client

## Recommended Commands

### SFT

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

### GRPO

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

## Validation Options

### Dry-run

- `src/train.py --dry-run`
- `src/train_grpo.py --dry-run`
- `LOCAL_VALIDATE=true bash scripts/run_pipeline.sh all`

### Smoke test

```bash
bash scripts/run_smoke_test.sh
```

Use the smoke test when you want a smaller real Tinker run that still exercises
baseline, SFT, GRPO, and comparison behavior.
