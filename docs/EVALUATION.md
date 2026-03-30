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
**Test Set**: API-Bank + ToolBench test splits
**Target**: Baseline ~65%, SFT ~85%, GRPO >90%

Each test example provides a query and a list of available tools. The evaluator
generates a response and checks whether the first tool call matches the expected tool.

### 2. Argument Generation Accuracy

**Metric**: F1 score for argument names and values
**Test Set**: Gorilla test split
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
**Test Set**: API-Bank multi-step examples
**Target**: Baseline ~40%, SFT ~70%, GRPO >80%

### 5. Latency

**Metric**: Average generation time per example (seconds)
**Target**: < 2s per response on A100 / < 5s on consumer GPU

## Running Evaluations

### CLI usage

```bash
# Evaluate baseline only
python src/evaluate.py \
  --base-model Qwen/Qwen3-8B \
  --mode baseline \
  --output-dir outputs/eval_baseline

# Evaluate SFT
python src/evaluate.py \
  --base-model Qwen/Qwen3-8B \
  --sft-adapter outputs/sft/final_adapter \
  --mode sft \
  --output-dir outputs/eval_sft

# Evaluate GRPO
python src/evaluate.py \
  --base-model Qwen/Qwen3-8B \
  --sft-adapter outputs/sft/final_adapter \
  --grpo-adapter outputs/grpo/final_adapter \
  --mode grpo \
  --output-dir outputs/eval_grpo

# Side-by-side comparison (requires all three results)
python src/evaluate.py \
  --base-model Qwen/Qwen3-8B \
  --sft-adapter outputs/sft/final_adapter \
  --grpo-adapter outputs/grpo/final_adapter \
  --mode compare \
  --output-dir outputs/eval_compare

# Run everything in sequence
python src/evaluate.py \
  --base-model Qwen/Qwen3-8B \
  --sft-adapter outputs/sft/final_adapter \
  --grpo-adapter outputs/grpo/final_adapter \
  --mode all \
  --output-dir outputs/eval_all
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
bash scripts/evaluate_model.sh \
  --base-model Qwen/Qwen3-8B \
  --sft-adapter outputs/sft/final_adapter \
  --grpo-adapter outputs/grpo/final_adapter \
  --mode all
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

Understanding the model loading chain is important for reproducibility:

1. **Baseline**: `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")` in bf16.
2. **SFT**: Same base model + `PeftModel.from_pretrained(base, sft_adapter_path)`.
3. **GRPO**: Base model → merge SFT adapter into weights → `PeftModel.from_pretrained(merged, grpo_adapter_path)`.

The GRPO stage merges SFT weights *first* because GRPO training started from a merged SFT checkpoint with a fresh LoRA on top.

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

Each evaluation writes a JSON file to the output directory:

```
outputs/eval_all/
├── baseline_results.json
├── sft_results.json
├── grpo_results.json
└── comparison.json
```

Each JSON contains the metric values, example-level predictions, and metadata
(model paths, dataset sizes, timestamps).

## Next Steps

- Review results in W&B: https://wandb.ai/dhruvanmurthy/qwen3-8b-tool-use
- If GRPO targets are not met, increase `max_steps` beyond 50 (see [TRAINING_PLAN.md](TRAINING_PLAN.md))
- Push final model to HF Hub: `huggingface-cli upload dhruvanmurthy/qwen3-8b-tool-use-lora outputs/grpo/final_adapter`
        example["query"],
        example["expected_calls"],
        pred,
        pred == example["expected_calls"]
    )
wandb.log({"evaluation_examples": eval_table})
```

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
            --datasets api-bank,toolbench,gorilla \
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
| API-Bank test (500 ex) | ~$1 |
| ToolBench test (1k ex) | ~$2 |
| Gorilla test (500 ex) | ~$1 |
| Adversarial test (500 ex) | ~$1 |
| **Total** | **~$5** |

Run evaluations frequently (low cost!)

## Next Steps

1. Complete training ([TRAINING_PLAN.md](TRAINING_PLAN.md))
2. Run evaluation script:
   ```bash
   python src/evaluate.py \
     --model dhruvanmurthy/qwen3-8b-tool-use-lora \
     --datasets api-bank,toolbench,gorilla
   ```
3. Compare against baselines
4. Iterate on training hyperparameters based on results
5. Log results to W&B for tracking

---
**Last Updated**: March 2026
