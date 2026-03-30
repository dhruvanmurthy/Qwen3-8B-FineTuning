# Dataset Strategy & Pipeline

Complete guide to aggregating, preprocessing, and versioning datasets for tool-use fine-tuning.

## Dataset Overview

**Total Target**: 40,000 training samples
**Strategy**: Mix of real + synthetic data for diversity
**Format**: Unified ChatML (OpenAI function calling format)
**Time to Prepare**: ~2-3 hours (automated via `prepare_datasets.sh`)

## Source Datasets

### 1. API-Bank (5,000 samples)

**Homepage**: https://huggingface.co/datasets/apibench/api-bank
**Characteristics**:
- Real API calls with execution results
- Multi-step sequences (1-3 API calls)
- Long context (API documentation)
- Diverse APIs (weather, translation, etc.)

**Acquisition**:
```bash
python -c "
from datasets import load_dataset
api_bank = load_dataset('apibench/api-bank')
print(api_bank['train'][0])
"
```

**Example**:
```json
{
  "instruction": "I want to translate 'Good Morning' to Korean and get the current weather in Seoul",
  "tools": [
    {"name": "translate", "description": "Translate text", "params": {"text": "...", "target_lang": "..."}},
    {"name": "get_weather", "description": "Get weather", "params": {"city": "..."}}
  ],
  "api_calls": [
    {"tool": "translate", "params": {"text": "Good Morning", "target_lang": "Korean"}},
    {"tool": "get_weather", "params": {"city": "Seoul"}}
  ],
  "results": [
    "안녕하세요",
    {"temperature": 18, "condition": "cloudy"}
  ],
  "final_response": "The translation is '안녕하세요' and Seoul is 18°C, cloudy."
}
```

**Selection**: Use first 5,000 samples (stratified by API type)

### 2. ToolBench (15,000 samples)

**Homepage**: https://huggingface.co/datasets/ToolBench/toolbench
**Characteristics**:
- Tool selection from large catalogs (100+ tools)
- Multi-step reasoning chains
- User instruction only (no execution results)
- Diverse categories (e-commerce, travel, etc.)

**Acquisition**:
```bash
# Download from source
python -c "
from datasets import load_dataset
toolbench = load_dataset('ToolBench/toolbench', 'G1_instructions')
print(len(toolbench), toolbench['train'][0].keys())
"
```

**Example**:
```json
{
  "user_instruction": "I need to book a round-trip flight from NYC to LA for June 10-15",
  "available_tools": [
    {"name": "search_flights", "description": "...", "params": [...]},
    {"name": "book_flight", "description": "...", "params": [...]},
    {"name": "check_prices", "description": "...", "params": [...]}
  ],
  "expected_calls": [
    {"tool": "search_flights", "params": {"from": "NYC", "to": "LA", "date": "2024-06-10"}},
    {"tool": "check_prices", "params": {...}},
    {"tool": "book_flight", "params": {...}}
  ],
  "answer": "I found flights and booking completed."
}
```

**Selection**: Balanced sample from all categories, ~15,000 total

### 3. Gorilla (5,000 samples)

**Homepage**: https://huggingface.co/datasets/gorilla-llm/gorilla-dataset
**Characteristics**:
- Real API documentation
- Focused on parameter correctness
- Single function calls (not chains)
- Wide API coverage (1,645 unique APIs)

**Acquisition**:
```bash
python -c "
from datasets import load_dataset
gorilla = load_dataset('gorilla-llm/gorilla-dataset', 'api_domain')
print(gorilla['eval'][0])
"
```

**Example**:
```json
{
  "api_name": "stripe.charge.create",
  "api_doc": "Create a charge with the given amount and currency.",
  "user_instruction": "Charge $99.99 to customer cus_12345 with description 'Premium Plan'",
  "correct_api_call": "stripe.charge.create(amount=9999, currency='usd', customer='cus_12345', description='Premium Plan')",
  "parameters": {
    "amount": 9999,
    "currency": "usd",
    "customer": "cus_12345",
    "description": "Premium Plan"
  }
}
```

**Selection**: Stratified by API domain, ~5,000 samples

### 4. Synthetic Data (15,000 samples)

**Source**: Generated via GPT-4 with careful prompting
**Rationale**: Fill domain gaps, edge cases, instruction-following diversity

**Categories**:
- E-commerce (product search, cart, checkout): 3,000
- Travel (booking, itineraries, pricing): 3,000
- Finance (payments, transfers, statements): 3,000
- Content (writing, translation, summarization): 3,000
- Utilities (reminders, scheduling, notifications): 3,000

**Generation Process**:

```python
# Pseudo-code for synthetic data generation
import anthropic

def generate_synthetic_example(category: str) -> dict:
    client = anthropic.Anthropic()
    prompt = f"""
    Generate a tool-use example for {category}. Format as JSON with:
    - instruction: User query
    - tools: Available tools with names, descriptions, parameters
    - tool_calls: Expected function calls
    - results: Simulated execution results
    - response: Final assistant response

    Ensure realistic parameters, diverse tool combinations.
    """

    response = client.messages.create(
        model="claude-3-opus-20250219",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(response.content[0].text)

# Generate 15,000 examples
examples = []
for category in ["ecommerce", "travel", "finance", "content", "utilities"]:
    for _ in range(3000):
        examples.append(generate_synthetic_example(category))
```

**Quality Control**:
- ✅ Schema validation (all required fields)
- ✅ Manual review of 1,000 random samples (10%)
- ✅ Hallucination detection (verify tool names exist)
- ✅ Deduplication (MD5 hash)

## Data Format Standardization

All sources converted to unified ChatML format:

```yaml
# Format:
messages:
  - role: system
    content: "System prompt..."
  - role: user
    content: "User query..."
    tools: [...]  # Optional tool definitions
  - role: assistant
    content: "...Thinking..."
    tool_calls: [...]  # Optional function calls
  - role: user
    content: "[Tool execution results]"
  - role: assistant
    content: "Final response..."
```

### Conversion Function

```python
def convert_to_chatml(example: dict, source: str) -> dict:
    """Convert any source format to ChatML."""
    if source == "api-bank":
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["instruction"],
                 "tools": example["tools"]},
                {"role": "assistant", "content": "",
                 "tool_calls": example["api_calls"]},
                {"role": "user", "content": str(example["results"])},
                {"role": "assistant", "content": example["final_response"]}
            ]
        }
    elif source == "toolbench":
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["user_instruction"],
                 "tools": example["available_tools"]},
                {"role": "assistant", "content": "",
                 "tool_calls": example["expected_calls"]},
                {"role": "assistant", "content": example["answer"]}
            ]
        }
    # ... more sources
```

## Preprocessing Pipeline

### Step 1: Download All Datasets

```bash
bash scripts/prepare_datasets.sh
```

**Output**: `data/raw/` with subdirectories:
- `data/raw/api-bank/`
- `data/raw/toolbench/`
- `data/raw/gorilla/`
- `data/raw/synthetic/`

### Step 2: Clean & Normalize

The `prepare_datasets.sh` script handles cleaning automatically.

**Operations**:
- Remove duplicates (MD5)
- Validate JSON schema
- Remove incomplete examples
- Normalize whitespace
- Extract language (filter non-English if needed)

### Step 3: Convert to ChatML

The pipeline script converts all sources into unified ChatML format.

**Output**: `data/processed/dataset_all.jsonl` (40,000 examples)

Each line is a JSON object:
```json
{"messages": [...], "source": "api-bank", "id": "..."}
```

### Step 4: Tokenize & Create HF Dataset

Tokenization happens automatically during training via `ToolUseDataLoader.tokenize_dataset()`.
The data loader can also save a preprocessed HF Dataset to disk:

**Output**: Hugging Face Dataset object in `data/processed/hf_dataset/`
- `train/` (32,000 samples)
- `validation/` (4,000 samples)
- `test/` (4,000 samples)

All pre-tokenized to max_length=2048

### Step 5: Upload to Hub (Optional)

```bash
huggingface-cli upload dhruvanmurthy/qwen3-8b-tool-use-dataset data/processed/hf_dataset
```

**Output**: Dataset on Hugging Face Hub
- URL: https://huggingface.co/datasets/dhruvanmurthy/qwen3-8b-tool-use
- Compatible with `load_dataset(...)`

## Configuration

Create `configs/dataset_config.yaml`:

```yaml
sources:
  api-bank:
    url: "apibench/api-bank"
    type: huggingface
    split: train
    samples: 5000
    weight: 1.0

  toolbench:
    url: "ToolBench/toolbench"
    type: huggingface
    split: train
    config: "G1_instructions"
    samples: 15000
    weight: 1.0

  gorilla:
    url: "gorilla-llm/gorilla-dataset"
    type: huggingface
    split: eval
    config: "api_domain"
    samples: 5000
    weight: 1.0

  synthetic:
    url: "./data/raw/synthetic/"
    type: local
    samples: 15000
    weight: 1.0

preprocessing:
  max_length: 2048
  min_length: 50
  remove_duplicates: true
  remove_incomplete: true
  remove_non_english: false
  balance_sources: true

splits:
  train: 0.8
  validation: 0.1
  test: 0.1

seed: 42
```

## Handling Large Files

### Using Streaming (Don't Download All)

For very large datasets, stream instead of downloading:

```python
from datasets import load_dataset

# Stream without downloading
toolbench = load_dataset("ToolBench/toolbench", streaming=True)

for sample in toolbench["train"]:
    # Process one at a time
    process(sample)
```

### Batching Downloads

```bash
# Download in 5,000-sample batches to avoid timeout
bash scripts/prepare_datasets.sh
```

## Data Versioning with DVC

Track dataset versions with DVC (Data Version Control):

```bash
# Initialize DVC
dvc init

# Track processed dataset
dvc add data/processed/dataset_all.jsonl
git add data/processed/dataset_all.jsonl.dvc
git commit -m "Add full dataset v1.0"

# Create version tag
git tag dataset-v1.0
git push --tags
```

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
Total examples: 40000
Avg tokens per example: 487
Max tokens: 2048
Tool calls per example: 2.31

Source distribution:
- api-bank: 5000 (12.5%)
- toolbench: 15000 (37.5%)
- gorilla: 5000 (12.5%)
- synthetic: 15000 (37.5%)

Language distribution:
- en: 38800 (97%)
- es: 600 (1.5%)
- fr: 400 (1%)
- other: 200 (0.5%)
```

## Accessing in Training

### Via HuggingFace Datasets

```python
from datasets import load_dataset

# Local
dataset = load_dataset("parquet", data_files="data/processed/hf_dataset/train-00000-of-00001.parquet")

# Or from Hub
dataset = load_dataset("dhruvanmurthy/qwen3-8b-tool-use", split="train")
```

### Via PyArrow

```python
import pyarrow.parquet as pq

table = pq.read_table("data/processed/hf_dataset/train-00000-of-00001.parquet")
dataset = table.to_pandas()
```

## Balancing & Sampling Strategies

### Stratified Sampling (by source)

```python
from datasets import load_dataset

dataset = load_dataset("dhruvanmurthy/qwen3-8b-tool-use", split="train")

# Equal contribution from each source
for source in ["api-bank", "toolbench", "gorilla", "synthetic"]:
    subset = dataset.filter(lambda x: x["source"] == source)
    print(f"{source}: {len(subset)}")

# Balance to smallest source (5,000)
balanced = []
for source in ["api-bank", "toolbench", "gorilla", "synthetic"]:
    subset = dataset.filter(lambda x: x["source"] == source)
    balanced.extend(subset.select(range(min(5000, len(subset)))))
```

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

2. Verify output:
   ```bash
   python -c "
   from datasets import load_dataset
   ds = load_dataset('parquet', data_files='data/processed/hf_dataset/train-*.parquet')
   print(len(ds['train']), ds['train'][0].keys())
   "
   ```

3. Proceed to [TRAINING_PLAN.md](TRAINING_PLAN.md) for the 3-stage pipeline
