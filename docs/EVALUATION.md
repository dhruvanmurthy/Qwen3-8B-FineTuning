# Evaluation

The current evaluation implementation lives in `src/evaluate.py`.

It compares three stages:

- `baseline`
- `sft`
- `grpo`

It can also build a comparison table with `compare`, run the full sequence with
`all`, and reconstruct a comparison from finished W&B runs with
`fetch-compare`.

## Benchmark Definitions

### Tool Selection Accuracy

- metric key: `tool_selection_accuracy`
- behavior: exact match on the first extracted tool call

### Argument Accuracy

- metric key: `argument_accuracy`
- behavior: exact dictionary match on the extracted arguments

### Schema Compliance

- metric key: `schema_compliance`
- behavior: output contains a parseable tool call with a tool name and argument
  object

### Multi-Step Success

- metric key: `multi_step_success`
- behavior: expected tool-call chain is reproduced in order

### Latency

- metric keys: `avg_latency_ms`, `ms_per_token`
- behavior: average response latency and normalized token timing

## Data Source for Evaluation

Evaluation examples are loaded in this order:

1. `data/processed/test_raw.jsonl`
2. held-out examples from `data/raw/synthetic/`

`test_raw.jsonl` is the preferred source because it preserves tool metadata and
structured fields needed for evaluation.

## Common Commands

### Baseline

```bash
python src/evaluate.py \
  --mode baseline \
  --base-model Qwen/Qwen3-8B \
  --output outputs/eval_baseline.json
```

### SFT

```bash
python src/evaluate.py \
  --mode sft \
  --base-model Qwen/Qwen3-8B \
  --sft-output-dir outputs/sft \
  --output outputs/eval_sft.json
```

### GRPO

```bash
python src/evaluate.py \
  --mode grpo \
  --base-model Qwen/Qwen3-8B \
  --grpo-output-dir outputs/grpo \
  --output outputs/eval_grpo.json
```

### Comparison

```bash
python src/evaluate.py \
  --mode compare \
  --base-model Qwen/Qwen3-8B \
  --sft-output-dir outputs/sft \
  --grpo-output-dir outputs/grpo \
  --output outputs/eval_comparison.json
```

### Full sequence

```bash
python src/evaluate.py \
  --mode all \
  --base-model Qwen/Qwen3-8B \
  --sft-output-dir outputs/sft \
  --grpo-output-dir outputs/grpo \
  --output outputs/evaluation_results.json
```

### Comparison from completed W&B runs

```bash
python src/evaluate.py \
  --mode fetch-compare \
  --baseline-run-path entity/project/run_id \
  --sft-run-path entity/project/run_id \
  --grpo-run-path entity/project/run_id \
  --output outputs/eval_comparison.json
```

## Benchmark Subsets

You can limit the run to a subset with `--benchmarks`:

```bash
python src/evaluate.py \
  --mode baseline \
  --benchmarks schema_compliance multi_step \
  --output outputs/eval_subset.json
```

This is how the smoke test keeps the evaluation pass small.

## Sampler Resolution

For SFT and GRPO evaluation, the script resolves sampler paths by:

1. using `--sft-sampler-path` or `--grpo-sampler-path` when provided
2. otherwise reading the latest sampler path from `checkpoints.jsonl`

That means `outputs/sft/checkpoints.jsonl` and `outputs/grpo/checkpoints.jsonl`
are the authoritative local metadata for evaluation.

## Outputs

Evaluation writes JSON to the path supplied by `--output`.

Typical files:

- `outputs/eval_baseline.json`
- `outputs/eval_sft.json`
- `outputs/eval_grpo.json`
- `outputs/eval_comparison.json`

When multiple stages are present, the script also prints a markdown comparison
table to stdout and logs a W&B comparison table.
