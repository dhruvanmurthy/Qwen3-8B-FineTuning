# Dataset Strategy & Pipeline

Complete guide to aggregating, preprocessing, and versioning datasets for tool-use fine-tuning.

## Dataset Overview

**Total Loaded**: ~20,458 raw samples → ~3,043 training after dedup + balancing
**Strategy**: Mix of real HF datasets + locally generated synthetic data
**Format**: Unified `text` column (normalized from heterogeneous schemas)
**Time to Prepare**: ~5 minutes (automated via `prepare_datasets.sh`)

## Source Datasets

### 1. APIBench (5,000 samples)

**Homepage**: https://huggingface.co/datasets/gorilla-llm/APIBench
**Characteristics**:
- Real API documentation from TorchHub, HuggingFace, and TensorFlow
- API call + domain + structured api_data
- Mixed-type `api_arguments` field (requires pandas fallback loader)
- Single API calls with parameter specifications

**Acquisition**:
```bash
# Loaded via data_loader.py with pandas fallback (ArrowInvalid on default loader)
# Individual files: torchhub_train.json, huggingface_train.json, tensorflow_train.json
python -c "
from datasets import load_dataset
ds = load_dataset('gorilla-llm/APIBench', data_files='torchhub_train.json', split='train')
print(ds[0])
"
```

**Schema (columns)**:
- `domain`: API framework (e.g., "Image Classification")
- `api_call`: The API invocation string
- `api_data`: Structured data with `api_name`, `api_arguments`, `description`, etc.

**Example**:
```json
{
  "domain": "Image Classification",
  "api_call": "hub.load('pytorch/vision', 'resnet18', pretrained=True)",
  "api_data": {
    "api_name": "resnet18",
    "api_arguments": {"pretrained": true},
    "description": "ResNet-18 model pre-trained on ImageNet"
  }
}
```

**Selection**: All 5,000 samples across 3 files (torchhub + huggingface + tensorflow)

### 2. ToolBench (200 samples)

**Homepage**: https://huggingface.co/datasets/tuandunghcmut/toolbench-v1
**Characteristics**:
- Tool selection from large API catalogs
- User queries with relevant API lists
- Benchmark config, `g1_instruction` split (200 rows available)
- Diverse categories (data, sports, streaming, etc.)

**Acquisition**:
```bash
python -c "
from datasets import load_dataset
toolbench = load_dataset('tuandunghcmut/toolbench-v1', 'benchmark', split='g1_instruction')
print(len(toolbench), toolbench.column_names)
"
```

**Schema (columns)**:
- `query_id`: Unique identifier
- `query`: User instruction/question
- `api_list`: JSON list of available API tools with descriptions and parameters
- `relevant_apis`: List of [tool_name, api_name] pairs for the correct answer

**Example**:
```json
{
  "query_id": 577,
  "query": "I am a fitness enthusiast and I want to buy a fitness tracker. Can you suggest some top-rated fitness trackers?",
  "api_list": [{"category_name": "Data", "tool_name": "ASIN Data", "api_name": "Category", ...}],
  "relevant_apis": [["ASIN Data", "Search"], ["ASIN Data", "Product"]]
}
```

**Selection**: All 200 available rows from the benchmark/g1_instruction split

> **Note**: The benchmark split only has 200 rows. The `default` config has ~187k rows
> but uses a different schema. Small sources are oversampled during balancing.

### 3. Gorilla BFCL (258 samples)

**Homepage**: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
**Characteristics**:
- Function calling benchmark with parameter correctness focus
- Nested question format (`[[{role, content}]]`) + function definitions
- Mixed-type columns across JSON files (requires pandas fallback loader)
- Single function calls with detailed schemas

**Acquisition**:
```bash
# Loaded via data_loader.py with pandas fallback (ArrowInvalid on default loader)
# Uses specific data_files to avoid mixed-column issues across JSON files
python -c "
from datasets import load_dataset
ds = load_dataset('gorilla-llm/Berkeley-Function-Calling-Leaderboard',
                  data_files='BFCL_v3_live_simple.json', split='train')
print(ds[0])
"
```

**Schema (columns)**:
- `id`: Example identifier (e.g., "live_simple_0-0-0")
- `question`: Nested list `[[{role, content}]]` with user query
- `function`: List of function definitions with name, description, parameters

**Example**:
```json
{
  "id": "live_simple_0-0-0",
  "question": [[{"role": "user", "content": "Can you retrieve the details for user ID 7890?"}]],
  "function": [{"name": "get_user_info", "description": "Retrieve details for a specific user",
    "parameters": {"type": "dict", "required": ["user_id"],
      "properties": {"user_id": {"type": "integer", "description": "The unique identifier"}}}}]
}
```

**Selection**: All 258 rows from `BFCL_v3_live_simple.json`

### 4. Synthetic Data (15,000 samples)

**Source**: Generated via `scripts/generate_synthetic.py` with template-based generation
**Rationale**: Fill domain gaps, provide majority of training data

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
- 15,000 sampled during dataset loading

**Quality Control**:
- Schema validation (text, expected_tool, expected_args fields)
- MD5 deduplication
- Deterministic with seed for reproducibility

## Data Format Standardization

All sources are normalized to a unified `text` column (see Step 3 below).
The original heterogeneous schemas (ChatML messages, API calls, queries, etc.)
are flattened to plain text for tokenization.

## Preprocessing Pipeline\n\n### Step 1: Generate Synthetic Data & Download HF Datasets\n\n```bash\nbash scripts/prepare_datasets.sh\n```\n\nThis runs two steps:\n1. `python scripts/generate_synthetic.py` → generates JSONL files in `data/raw/synthetic/`\n2. `python -c \"...\"` → loads all sources, normalizes, deduplicates, balances, splits, tokenizes

**Output**: `data/raw/` with subdirectories:
- `data/raw/synthetic/` (JSONL files generated by `scripts/generate_synthetic.py`)

HF datasets are downloaded to `hf_cache/` (gitignored) and loaded on-the-fly.

### Step 2: Clean & Normalize

The `prepare_datasets.sh` script handles cleaning automatically via `data_loader.py`.

**Operations**:
- Remove duplicates (MD5 hash of full row)
- Validate completeness (at least one meaningful content field)
- Normalize whitespace
- Convert all schemas to unified `text` column via `_normalize_to_text()`

### Step 3: Normalize to Text

All sources are converted to a unified `text` column (not ChatML messages).
The normalizer handles each source's schema:

| Source | Fields Used | Text Format |
|--------|------------|-------------|
| Synthetic | `text` (already present) | As-is |
| APIBench | `domain`, `api_call`, `api_data` | `domain: ...\napi_call: ...\napi_data: ...` |
| ToolBench | `query`, `api_list`, `relevant_apis` | `query: ...\napi_list: [...]\nrelevant_apis: [...]` |
| BFCL | `question`, `function` | `question: [...]\nfunction: [...]` |

### Step 4: Balance, Split & Tokenize

After normalization, the pipeline:
1. **Balances sources** using median-target resampling (small sources oversampled, large undersampled)
2. **Splits** into train (80%) / validation (10%) / test (10%)
3. **Tokenizes** to `max_length=2048`

**Output**: HuggingFace Dataset (Arrow format) in `data/processed/`
- `data/processed/train/` (~3,043 samples)
- `data/processed/validation/` (~380 samples)
- `data/processed/test/` (~381 samples)

### Step 5: Upload to Hub (Optional)\n\n```bash\nhuggingface-cli upload dhruvanmurthy/qwen3-8b-tool-use-dataset data/processed\n```

## Configuration

Create `configs/dataset_config.yaml`:

```yaml
sources:
  api_bank:
    enabled: true
    name: "API-Bank"
    url: "gorilla-llm/APIBench"
    type: "huggingface"
    split: "train"
    samples: 5000
    config: null
    weight: 1.0
    data_files:
      - "torchhub_train.json"
      - "huggingface_train.json"
      - "tensorflow_train.json"

  toolbench:
    enabled: true
    name: "ToolBench"
    url: "tuandunghcmut/toolbench-v1"
    type: "huggingface"
    split: "g1_instruction"
    config: "benchmark"
    samples: 15000
    weight: 1.0

  gorilla:
    enabled: true
    name: "Gorilla"
    url: "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    type: "huggingface"
    split: "train"
    config: null
    samples: 5000
    weight: 1.0
    data_files:
      - "BFCL_v3_live_simple.json"

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
  remove_non_english: false

splits:
  train: 0.8
  validation: 0.1
  test: 0.1

balance_sources: true
seed: 42
```

## Handling Large Files

### Using Streaming (Don't Download All)

For very large datasets, stream instead of downloading:

```python
from datasets import load_dataset

# Stream without downloading
toolbench = load_dataset("tuandunghcmut/toolbench-v1", "benchmark", split="g1_instruction", streaming=True)

for sample in toolbench:
    # Process one at a time
    process(sample)
```

### Batching Downloads

```bash
# Download in 5,000-sample batches to avoid timeout
bash scripts/prepare_datasets.sh
```

## Data Versioning with Git

Track dataset config and code with Git (actual data files are gitignored):

```bash
# Tag current state
git tag dataset-v1.0
git push --tags
```

The processed Arrow files in `data/processed/` are regenerated from scratch
by `bash scripts/prepare_datasets.sh` and are gitignored.

## Deduplication & Quality

### Deduplication

```python
import hashlib
from collections import defaultdict

seen = defaultdict(list)
duplicates = []

for example in examples:
    text = json.dumps(example["messages"], sort_keys=True)
    hash_val = hashlib.md5(text.encode()).hexdigest()

    if hash_val in seen:
        duplicates.append((example["id"], seen[hash_val]))
    else:
        seen[hash_val].append(example["id"])

print(f"Found {len(duplicates)} duplicates")
# Keep first occurrence, remove rest
```

### Quality Metrics

After preprocessing, compute:

```python
# Dataset statistics
print(f"Total examples: {len(dataset)}")
print(f"Avg tokens per example: {dataset.map(count_tokens)['token_count'].mean():.0f}")
print(f"Max tokens: {dataset.map(count_tokens)['token_count'].max()}")
print(f"Tool calls per example: {dataset.map(count_tools)['tools'].mean():.2f}")

# Source distribution
from collections import Counter
sources = Counter([x["source"] for x in dataset])
for source, count in sources.most_common():
    print(f"{source}: {count} ({100*count/len(dataset):.1f}%)")

# Language distribution
langs = Counter([detect_language(x["messages"]) for x in dataset])
for lang, count in langs.most_common():
    print(f"{lang}: {count} ({100*count/len(dataset):.1f}%)")
```

### Expected Output

```
Total loaded: ~20,458
After dedup: ~14,382

Source distribution (before balancing):
- api_bank: 1,644 (11.4%)
- toolbench: 200 (1.4%)
- gorilla: 258 (1.8%)
- synthetic: 12,280 (85.4%)

After median-target balancing (~951 per source):
- Total: ~3,804

Final splits:
- Train: ~3,043 (80%)
- Validation: ~380 (10%)
- Test: ~381 (10%)
```

## Accessing in Training

### Via HuggingFace Datasets

```python\nfrom datasets import load_from_disk\n\n# Local (Arrow format)\nds = load_from_disk(\"data/processed\")\ntrain = ds[\"train\"]\nprint(len(train), train.column_names)  # ~3043, ['input_ids', 'attention_mask', 'labels']\n```

## Balancing & Sampling Strategies

### Median-Target Resampling (Current Approach)

The `_balance_sources` method uses the **median** source count as the target
(not the minimum), which avoids collapsing the dataset when one source is tiny.

- Sources smaller than the median are **oversampled** (with replacement)
- Sources larger than the median are **undersampled** (without replacement)
- Floor: at least 500 samples per source (or the max if all are smaller)

```python
# From src/data_loader.py — _balance_sources
counts = sorted(source_counts.values())
median = counts[n // 2]
target_count = max(median, min(500, max(counts)))

for source_name in source_counts:
    subset = dataset.filter(lambda x: x["source"] == source_name)
    if len(subset) >= target_count:
        indices = np.random.choice(len(subset), target_count, replace=False)
    else:
        indices = np.random.choice(len(subset), target_count, replace=True)  # oversample
```

This is controlled by `balance_sources: true` in `dataset_config.yaml`.

### Curriculum Learning (Optional)

```python
# Easy → Hard ordering
difficulty_scores = dataset.map(score_difficulty)
dataset = dataset.sort_by("difficulty")

# Train: hard examples later
# Initially train on simple examples, then harder
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
            str(ex1["messages"]),
            str(ex2["messages"])
        ).ratio()
        if similarity > 0.95:
            duplicates.append((i, j, similarity))
```

### Issue: Imbalanced Tool Distribution

Some APIs appear in >50% of examples:

**Fix**: Undersample common tools:
```python
tool_counts = Counter()
for ex in dataset:
    for call in ex.get("tool_calls", []):
        tool_counts[call["tool"]] += 1

# Downsample top 10% of tools
top_tools = [t for t, c in tool_counts.most_common(int(0.1*len(tool_counts)))]
dataset = dataset.filter(
    lambda x: not any(c["tool"] in top_tools
                      for c in x.get("tool_calls", []))
    or random.random() < 0.5  # 50% keep rare tools
)
```

## Reproducibility Checklist

✅ Dataset version pinned (v1.0)
✅ Random seed (42) for all splits
✅ DVC tracked (`.dvc` files in git)
✅ Preprocessing script versioned
✅ License attribution (DATASET_CARD.md)
✅ Sample examples in documentation
✅ Source URLs and hashes recorded

## GRPO Prompt Preparation

The GRPO stage does **not** use the tokenized SFT dataset. Instead it needs a
prompt-only dataset where each row contains:

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
grpo_dataset.save_to_disk("data/processed/grpo_prompts")
```

The `train_grpo.py` script calls this internally — you do not need to run it
separately.

## Next Steps

1. Run data preparation:
   ```bash
   bash scripts/prepare_datasets.sh
   ```

2. Verify output:\n   ```bash\n   python -c \"\n   from datasets import load_from_disk\n   ds = load_from_disk('data/processed')\n   print(len(ds['train']), ds['train'].column_names)\n   \"\n   ```

3. Proceed to [TRAINING_PLAN.md](TRAINING_PLAN.md) for the 3-stage pipeline
