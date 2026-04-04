# Dataset Strategy & Pipeline

Complete guide to generating, preprocessing, and versioning synthetic datasets for tool-use fine-tuning.

## Dataset Overview

**Total Generated**: ~40,000 raw JSONL examples ? ~12,280 after dedup ? ~9,824 training
**Strategy**: 100% synthetically generated via `scripts/generate_synthetic.py`
**Format**: Unified `text` column + structured `tool_calls` field
**Time to Prepare**: ~2–5 minutes (automated via `prepare_datasets.sh`)

## Source Datasets

### Synthetic Data (15,000 sampled for training)

**Source**: `scripts/generate_synthetic.py` with template-based generation (seed=42)
**Rationale**: Full control over tool coverage, argument diversity, and schema compliance

**Categories** (12 single-step + 6 multi-step generators):
- Weather lookup, stock prices, currency conversion
- Translation, unit conversion, math operations
- Reminders, music search, email composition
- Multi-step: translate+weather, stock+convert, weather+remind, etc.

**Generation**:
```bash
python scripts/generate_synthetic.py --num-samples 40000 --seed 42
```

**Output**:
- `data/raw/synthetic/synthetic_single.jsonl` (~34,700 examples)
- `data/raw/synthetic/synthetic_multistep.jsonl` (~5,300 examples)
- 15,000 sampled during dataset loading for training

**Quality Control**:
- Schema validation (`instruction`, `tool_calls`, `text` fields required)
- MD5 deduplication
- Deterministic with seed for reproducibility

## Data Format

Every training example has the following schema:

```json
{
  "instruction": "What is the weather in Tokyo?",
  "tool_calls": [
    {
      "tool": "get_weather",
      "arguments": {"city": "Tokyo"}
    }
  ],
  "text": "<|im_start|>user\nWhat is the weather in Tokyo?<|im_end|>\n<|im_start|>assistant\n{\"tool\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}<|im_end|>"
}
```

Multi-step examples include a sequence in `tool_calls`:

```json
{
  "instruction": "Translate 'hello' to Spanish and get the weather in Madrid.",
  "tool_calls": [
    {"tool": "translate_text", "arguments": {"text": "hello", "target_language": "Spanish"}},
    {"tool": "get_weather", "arguments": {"city": "Madrid"}}
  ],
  "text": "..."
}
```

## Preprocessing Pipeline

### Step 1: Generate Synthetic Data

```bash
python scripts/generate_synthetic.py --num-samples 40000 --seed 42
# Writes:
#   data/raw/synthetic/synthetic_single.jsonl
#   data/raw/synthetic/synthetic_multistep.jsonl
```

Or run the full prepare script which handles everything:

```bash
bash scripts/prepare_datasets.sh
```

### Step 2: Clean & Normalize

`data_loader.py` handles cleaning automatically:

- Remove duplicates (MD5 hash of full row)
- Validate completeness (`instruction`, `tool_calls`, `text` fields required)
- Normalize whitespace
- Ensure `text` column is present (already the case for synthetic data)

### Step 3: Split & Tokenize

After normalization, the pipeline:
1. **Splits** into train (80%) / validation (10%) / test (10%)
2. **Tokenizes** to `max_length=2048`

**Output**: HuggingFace Dataset (Arrow format) in `data/processed/`
- `data/processed/train/` (~9,824 samples)
- `data/processed/validation/` (~1,228 samples)
- `data/processed/test/` (~1,228 samples)

### Step 4: Upload to Hub (Optional)

```bash
# Upload raw JSONL files to HF dataset repo
python scripts/push_dataset_to_hub.py \
  --repo-id YOUR_HF_USER/qwen3-8b-tool-use-dataset
```

## Configuration

`configs/dataset_config.yaml`:

```yaml
sources:
  synthetic:
    enabled: true
    name: "Synthetic"
    path: "./data/raw/synthetic/"
    type: "local"
    samples: 15000
    weight: 1.0

preprocessing:
  max_seq_length: 2048
  min_seq_length: 50
  remove_duplicates: true
  dedup_method: "md5"
  remove_incomplete: true

splits:
  train: 0.8
  validation: 0.1
  test: 0.1

seed: 42
```

## Deduplication & Quality

### Deduplication

```python
import hashlib
from collections import defaultdict

seen = defaultdict(list)
duplicates = []

for example in examples:
    text = json.dumps(example["text"], sort_keys=True)
    hash_val = hashlib.md5(text.encode()).hexdigest()

    if hash_val in seen:
        duplicates.append(example["id"])
    else:
        seen[hash_val].append(example["id"])

print(f"Found {len(duplicates)} duplicates")
```

### Quality Metrics

After preprocessing, compute:

```python
print(f"Total examples: {len(dataset)}")
print(f"Avg tokens per example: {dataset.map(count_tokens)['token_count'].mean():.0f}")
print(f"Max tokens: {dataset.map(count_tokens)['token_count'].max()}")
print(f"Tool calls per example: {dataset.map(count_tools)['tools'].mean():.2f}")
```

### Expected Output

```
Total generated:  ~40,000
After dedup:      ~12,280

Source distribution:
- synthetic (single-step): ~9,700 (79%)
- synthetic (multi-step):  ~2,580 (21%)

Final splits:
- Train:      ~9,824 (80%)
- Validation: ~1,228 (10%)
- Test:       ~1,228 (10%)
```

## Accessing in Training

```python
from datasets import load_from_disk

ds = load_from_disk("data/processed")
train = ds["train"]
print(len(train), train.column_names)
# ~9824, ['input_ids', 'attention_mask', 'labels']
```

## GRPO Prompt Preparation

The GRPO stage uses a prompt-only dataset where each row contains:

| Field | Description |
|-------|-------------|
| `prompt` | The user query formatted via `apply_chat_template` (system + user turn only) |
| `expected_tool` | Ground-truth tool name for reward scoring |
| `expected_args` | Ground-truth arguments dict |
| `expected_chain` | Full expected call sequence (for `full_chain_reward`) |

This is handled automatically by `ToolUseDataLoader.prepare_grpo_prompts()`:

```python
from src.data_loader import ToolUseDataLoader

loader = ToolUseDataLoader("configs/dataset_config.yaml")
grpo_dataset = loader.prepare_grpo_prompts(tokenizer)
```

## Common Issues & Fixes

### Issue: Memory Error During Tokenization

```
MemoryError: Unable to allocate 128 GB for an array
```

**Fix**: Tokenize in batches:
```python
dataset.map(tokenize_fn, batched=True, batch_size=1000)
```

### Issue: Duplicate Examples

```python
# Check for near-duplicates (not exact)
from difflib import SequenceMatcher

duplicates = []
for i, ex1 in enumerate(dataset):
    for j, ex2 in enumerate(dataset[i+1:], i+1):
        similarity = SequenceMatcher(None,
            str(ex1["text"]),
            str(ex2["text"])
        ).ratio()
        if similarity > 0.95:
            duplicates.append((i, j, similarity))
```

### Issue: Imbalanced Tool Distribution

Some tools appear in >50% of examples.

**Fix**: Undersample common tools:
```python
from collections import Counter
import random

tool_counts = Counter()
for ex in dataset:
    for call in ex.get("tool_calls", []):
        tool_counts[call["tool"]] += 1

top_tools = [t for t, c in tool_counts.most_common(int(0.1 * len(tool_counts)))]
dataset = dataset.filter(
    lambda x: not any(c["tool"] in top_tools for c in x.get("tool_calls", []))
    or random.random() < 0.5
)
```

## Data Versioning with Git

Track dataset config and generation code with Git (actual data files are gitignored):

```bash
# Tag current state
git tag dataset-v1.0
git push --tags
```

The processed Arrow files in `data/processed/` are regenerated from scratch
by `bash scripts/prepare_datasets.sh` and are gitignored.

## Reproducibility Checklist

? Random seed (42) for all generation and splits
? `generate_synthetic.py` versioned in Git
? `configs/dataset_config.yaml` versioned in Git
? License: MIT (see `DATASET_CARD.md`)
? Sample examples in `DATASET_CARD.md`
? Schema documented above

---
**Last Updated**: March 2026
