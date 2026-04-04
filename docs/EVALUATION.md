# Evaluation Strategy & Benchmarks

Three-stage evaluation comparing **Baseline → SFT → GRPO** on tool-use tasks.

## Pipeline Overview

The project evaluates Qwen3-8B at three checkpoints:

| Stage | Model | What it measures |
|-------|-------|------------------|
| **Baseline** | `Qwen/Qwen3-8B` (zero-shot) | Raw model capability before any fine-tuning |
| **SFT** | Base + SFT LoRA adapter | Improvement from supervised fine-tuning |
| **GRPO** | Base + merged-SFT + GRPO LoRA adapter | Improvement from reinforcement learning |

A final **compare** step prints a side-by-side table of all three stages.

## Evaluation Dimensions

### 1. Tool Selection Accuracy

**Metric**: Percentage of correct tool selections from available options
**Test Set**: Synthetic held-out split (10%)
**Target**: Baseline ~65%, SFT ~85%, GRPO >90%

Each test example provides a query and a list of available tools. The evaluator
generates a response and checks whether the first tool call matches the expected tool.

### 2. Argument Generation Accuracy

**Metric**: F1 score for argument names and values
**Test Set**: Synthetic held-out split
**Target**: Baseline ~50%, SFT ~75%, GRPO >85%

Compares predicted argument key-value pairs against expected ones using set-based
precision / recall / F1.

### 3. Schema Compliance

**Metric**: Percentage of outputs that conform to the expected JSON tool-call schema
**Target**: Baseline ~40%, SFT ~80%, GRPO >90%

Checks whether the model output can be parsed as a valid tool call (correct JSON
structure, required fields present, types correct).

### 4. Multi-Step Success Rate

**Metric**: End-to-end success of 2-3 step tool chains
**Test Set**: Synthetic multi-step held-out examples
**Target**: Baseline ~40%, SFT ~70%, GRPO >80%

### 5. Latency

**Metric**: Average generation time per example (seconds)
**Target**: < 2s per response on A100 / < 5s on consumer GPU

## Running Evaluations

All inference runs on Tinker’s remote GPUs (requires `TINKER_API_KEY`).
Sampler paths for SFT/GRPO models are auto-detected from `checkpoints.jsonl`
in the respective output directories.

### CLI usage

```bash
# Evaluate baseline only
python src/evaluate.py \
  --base-model Qwen/Qwen3-8B \
  --mode baseline \
  --output outputs/eval_baseline.json

# Evaluate SFT
python src/evaluate.py \
  --base-model Qwen/Qwen3-8B \
  --mode sft \
  --sft-output-dir outputs/sft \
  --output outputs/eval_sft.json

# Evaluate GRPO
python src/evaluate.py \
  --base-model Qwen/Qwen3-8B \
  --mode grpo \
  --sft-output-dir outputs/sft \
  --grpo-output-dir outputs/grpo \
  --output outputs/eval_grpo.json

# Side-by-side comparison (requires all three results)
python src/evaluate.py \
  --base-model Qwen/Qwen3-8B \
  --mode compare \
  --sft-output-dir outputs/sft \
  --grpo-output-dir outputs/grpo \
  --output outputs/eval_compare.json

# Run everything in sequence
python src/evaluate.py \
  --base-model Qwen/Qwen3-8B \
  --mode all \
  --sft-output-dir outputs/sft \
  --grpo-output-dir outputs/grpo \
  --output outputs/eval_all.json
```

### Via the pipeline script

```bash
# Full pipeline (trains + evaluates all stages)
bash scripts/run_pipeline.sh all

# Or evaluate-only stages
bash scripts/run_pipeline.sh baseline   # baseline eval
bash scripts/run_pipeline.sh compare    # comparison table
```

### Via the evaluate helper script

```bash
bash scripts/evaluate_model.sh all
```

## Expected Results

Target comparison table (actual results will vary):

| Metric | Baseline | SFT | GRPO | Target (GRPO) |
|--------|----------|-----|------|---------------|
| Tool Selection | ~65% | ~85% | **>90%** | >90% |
| Argument F1 | ~50% | ~75% | **>85%** | >85% |
| Schema Compliance | ~40% | ~80% | **>90%** | >90% |
| Multi-Step Success | ~40% | ~70% | **>80%** | >80% |
| Latency (s) | ~1.5 | ~1.8 | ~1.8 | <2.0 |

## How Model Loading Works

All inference runs on Tinker’s remote GPUs via sampling clients:

1. **Baseline**: Tinker loads `Qwen/Qwen3-8B` from HuggingFace Hub.
2. **SFT**: Tinker loads the model from the sampler path saved in `outputs/sft/checkpoints.jsonl`.
3. **GRPO**: Tinker loads the model from the sampler path saved in `outputs/grpo/checkpoints.jsonl`.

Sampler paths (`tinker://` URIs) are auto-detected from the checkpoint logs. You can
also pass them explicitly via `--sft-sampler-path` and `--grpo-sampler-path`.

## GRPO-Specific Metrics

During GRPO training, these additional reward metrics are tracked via W&B:

| Reward function | What it checks |
|-----------------|----------------|
| `tool_name_reward` | Correct tool selected (binary 0/1) |
| `argument_match_reward` | All argument keys & values match (binary 0/1) |
| `schema_validation_reward` | Output is valid parseable JSON tool call (binary 0/1) |
| `full_chain_reward` | Entire multi-step sequence correct (binary 0/1) |

Track the running average of each reward during GRPO training. If all rewards stay near 0 after 10+ steps, the model isn't learning — see [TROUBLESHOOTING.md](TROUBLESHOOTING.md#grpo-rewards-stuck-at-zero).

## Logging to W&B

Evaluation results are automatically logged to W&B:

```
wandb.ai/dhruvanmurthy/qwen3-8b-tool-use
```

Each evaluation run logs:
- `eval/{stage}/tool_selection` — accuracy
- `eval/{stage}/argument_accuracy` — F1
- `eval/{stage}/schema_compliance` — accuracy
- `eval/{stage}/multi_step_success` — accuracy
- `eval/{stage}/latency_seconds` — average

The compare stage logs a W&B Table with all three stages side-by-side.

## Output Files

Each evaluation writes a JSON file to the `--output` path:

```
outputs/evaluation_results.json
```

The JSON contains metric values for each evaluated stage, plus a comparison table
when running in `compare` or `all` mode.

## Next Steps

- Review results in W&B: https://wandb.ai/dhruvanmurthy/qwen3-8b-tool-use
- If GRPO targets are not met, increase `max_steps` beyond 50 (see [TRAINING_PLAN.md](TRAINING_PLAN.md))
- Push final model to HF Hub: `huggingface-cli upload dhruvanmurthy/qwen3-8b-tool-use-lora outputs/grpo/final_adapter`

## Per-Category Breakdown

Evaluate by tool category to identify weaknesses:

```python
def evaluate_by_category(model, dataset):
    """Evaluate accuracy per API category."""

    categories = {}

    for example in dataset:
        category = example["category"]  # e.g., "weather", "payment"

        if category not in categories:
            categories[category] = {"correct": 0, "total": 0}

        # Evaluate
        predicted = model_predict(example)
        is_correct = predicted == example["expected"]

        if is_correct:
            categories[category]["correct"] += 1
        categories[category]["total"] += 1

    # Print breakdown
    for cat in sorted(categories.keys()):
        stats = categories[cat]
        accuracy = 100 * stats["correct"] / stats["total"]
        print(f"{cat:20s}: {accuracy:5.1f}% ({stats['correct']}/{stats['total']})")
```

**Expected Output**:
```
weather          :  94.2% (47/50)
payment          :  89.5% (43/48)
translation      :  91.3% (42/46)
email            :  88.0% (44/50)
calendar         :  86.0% (43/50)
...
```

## Continuous Evaluation

Run evaluation on new models automatically:

```yaml
# .github/workflows/eval.yml
name: Evaluation
on:
  push:
    branches: [main]
    paths:
      - 'outputs/model_final/**'
      - 'src/evaluate.py'

jobs:
  evaluate:
    runs-on: ubuntu-latest-gpu
    steps:
      - uses: actions/checkout@v3
      - name: Evaluate Model
        run: |
          python src/evaluate.py \
            --model outputs/model_final \
            --datasets synthetic \
            --output results.json
      - name: Upload to W&B
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          python -c "
          import json, wandb
          with open('results.json') as f:
            results = json.load(f)
          wandb.log(results)
          "
```

## Cost of Evaluation

| Component | Cost |
|-----------|------|
| Synthetic test set (~1,228 ex) | ~$2 |
| **Total** | **~$2** |

Run evaluations frequently (low cost!)

## Next Steps

1. Complete training ([TRAINING_PLAN.md](TRAINING_PLAN.md))
2. Run evaluation script:
   ```bash
   python src/evaluate.py \
     --model dhruvanmurthy/qwen3-8b-tool-use-lora \
     --datasets synthetic
   ```
3. Compare against baselines
4. Iterate on training hyperparameters based on results
5. Log results to W&B for tracking

---
**Last Updated**: March 2026
