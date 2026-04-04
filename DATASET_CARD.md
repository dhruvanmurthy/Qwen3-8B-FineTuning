# Dataset Card: Qwen3-8B Tool Use Training Data

## Dataset Summary

Synthetic dataset for training Qwen3-8B on multi-step tool use, function calling, and API orchestration. Data is generated via template-based synthesis, deduplicated, and split for training and evaluation.

### Dataset Composition

| Source | Raw Count | After Dedup | Purpose |
|--------|-----------|-------------|----------|
| **Synthetic** | 15,000 | ~12,280 | Tool-use: single-step & multi-step calls |
| **Total** | **15,000** | **~12,280** | **Multi-domain tool use** |

### Final Splits

| Split | Count |
|-------|-------|
| Train | ~9,824 |
| Validation | ~1,228 |
| Test | ~1,228 |

## Data Sources

### Synthetic Data
- **Source**: Generated via `scripts/generate_synthetic.py` with template-based generation
- **Samples**: 15,000 (deterministic with seed=42)
- **Focus**: Domain-specific scenarios, edge cases, tool-use diversity
- **Categories** (12 single-step + 6 multi-step generators):
  - Weather lookup, stock prices, currency conversion
  - Translation, unit conversion, math operations
  - Reminders, music search, email composition
  - Multi-step: translate+weather, stock+convert, weather+remind, etc.
- **Quality**: MD5 deduplication, schema validation, deterministic with seed

**Example**:
```json
{
  "text": "[weather] USER: What is the weather like in Tokyo?\nASSISTANT: <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\", \"units\": \"C\"}}\n</tool_call>",
  "instruction": "What is the weather like in Tokyo?",
  "tool_calls": [{"name": "get_weather", "arguments": {"city": "Tokyo", "units": "C"}}],
  "category": "weather",
  "num_steps": 1
}
```

**Generation**:
```bash
python scripts/generate_synthetic.py --num-samples 15000 --seed 42
```

## Data Format

All data uses the native synthetic schema with `instruction`, `tool_calls`, `tools`, `category`, `num_steps`, and `text` fields. The `text` field is formatted as:

```
[category] USER: <instruction>
ASSISTANT: <tool_call>
{"name": ..., "arguments": {...}}
</tool_call>
```

Tokenization produces `input_ids`, `attention_mask`, and `labels`.

### Tokenized Format (Arrow files in data/processed/)
```json
{
  "input_ids": [151643, 2610, ...],
  "attention_mask": [1, 1, ...],
  "labels": [151643, 2610, ...]
}
```

## Statistics

### Data Distribution
```
Raw Generated:  15,000 samples
After Dedup:   ~12,280 samples

Training Set:   ~9,824 samples (80%)
Validation Set: ~1,228 samples (10%)
Test Set:       ~1,228 samples (10%)
```

### Sequence Length
```
Max Token Length: 2048 (truncated at tokenization)
```

### Language
- **Primary**: English (100%)

## Data Processing Pipeline

### Steps
1. **Generate Synthetic**: `python scripts/generate_synthetic.py` → JSONL files in `data/raw/synthetic/`
2. **Load**: Local JSONL files read from `data/raw/synthetic/`
3. **Deduplicate**: MD5 hash of full row, keep first occurrence
4. **Filter**: Remove incomplete examples (no meaningful content fields)
5. **Split**: 80/10/10 train/validation/test
6. **Tokenize**: Qwen3-8B tokenizer, max_length=2048, padding to max_length
7. **Save**: Arrow format in `data/processed/{train,validation,test}/`

### Quality Checks
✅ No duplicate examples (MD5 hash)
✅ All rows have at least one meaningful content field
✅ Whitespace normalized
✅ Deterministic with seed=42

## Data Access

### Load Processed Dataset
```python
from datasets import load_from_disk

ds = load_from_disk("data/processed")
print(ds["train"].column_names)  # ['input_ids', 'attention_mask', 'labels']
print(len(ds["train"]))          # ~9,824
```

### Regenerate from Scratch
```bash
rm -rf data/processed data/raw/synthetic/*.jsonl
bash scripts/prepare_datasets.sh
```

### Dataset Composition

| Source | Raw Count | After Dedup | Purpose |
|--------|-----------|-------------|---------|
| **Synthetic** | 15,000 | ~12,280 | Tool-use: single-step & multi-step calls |
| **Total** | **15,000** | **~12,280** | **Multi-domain tool use** |

### Final Splits

| Split | Count |
|-------|-------|
| Train | ~9,824 |
| Validation | ~1,228 |
| Test | ~1,228 |

## Data Sources

### Synthetic Data
- **Source**: Generated via `scripts/generate_synthetic.py` with template-based generation
- **Samples**: 15,000 (~10,500 single-step + ~4,500 multi-step)
- **Focus**: Domain-specific scenarios, edge cases, tool-use diversity
- **Categories** (12 single-step + 6 multi-step generators):
  - Weather lookup, stock prices, currency conversion
  - Translation, unit conversion, math operations
  - Reminders, music search, email composition
  - Multi-step: translate+weather, stock+convert, weather+remind, etc.
- **Quality**: MD5 deduplication, schema validation, deterministic with seed

**Example**:
```json
{
  "text": "User: What's the weather like in Tokyo right now?\nExpected tool: get_weather\nExpected args: {\"city\": \"Tokyo\"}\nAssistant: I'll check the current weather in Tokyo for you.",
  "expected_tool": "get_weather",
  "expected_args": {"city": "Tokyo"}
}
```

**Generation**:
```bash
python scripts/generate_synthetic.py --num-samples 40000 --seed 42
```

## Data Format

All data uses the native synthetic schema. The `text` field is the primary training signal; `instruction`, `tool_calls`, `tools`, `category`, and `num_steps` are retained as metadata.

Tokenization produces `input_ids`, `attention_mask`, and `labels`.

### Tokenized Format (Arrow files in data/processed/)
```json
{
  "input_ids": [151643, 2610, ...],
  "attention_mask": [1, 1, ...],
  "labels": [151643, 2610, ...]
}
```

## Statistics

### Data Distribution
```
Raw Generated:  15,000 samples
After Dedup:   ~12,280 samples

Training Set:   ~9,824 samples (80%)
Validation Set: ~1,228 samples (10%)
Test Set:       ~1,228 samples (10%)
```

### Sequence Length
```
Max Token Length: 2048 (truncated at tokenization)
```

### Language
- **Primary**: English (100% — all sources are English)

## Data Processing Pipeline

### Steps
1. **Generate Synthetic**: `python scripts/generate_synthetic.py` → JSONL files in `data/raw/synthetic/`
2. **Load**: Local JSONL files read from `data/raw/synthetic/`
3. **Deduplicate**: MD5 hash of full row, keep first occurrence
4. **Filter**: Remove incomplete examples (no meaningful content fields)
5. **Split**: 80/10/10 train/validation/test
6. **Tokenize**: Qwen3-8B tokenizer, max_length=2048, padding to max_length
7. **Save**: Arrow format in `data/processed/{train,validation,test}/`

### Quality Checks
✅ No duplicate examples (MD5 hash)
✅ All rows have at least one meaningful content field
✅ Whitespace normalized
✅ Deterministic with seed=42

## Data Access

### Load Processed Dataset
```python
from datasets import load_from_disk

ds = load_from_disk("data/processed")
print(ds["train"].column_names)  # ['input_ids', 'attention_mask', 'labels']
print(len(ds["train"]))          # ~3,043
```

### Regenerate from Scratch
```bash
rm -rf data/processed data/raw/synthetic/*.jsonl
bash scripts/prepare_datasets.sh
```

## Licensing

| Source | License |
|--------|---------|
| Synthetic | MIT |

All training data is self-generated. No third-party dataset licenses apply.

## Ethical Considerations

### Data Collection
- ✅ Datasets sourced from public, publicly-licensed sources
- ✅ No personally identifiable information (PII)
- ✅ No sensitive credentials in examples

### Potential Harms
1. **Synthetic Data Artifacts**: Template-generated examples may not reflect real user behavior
   - Mitigation: Diverse generators (15 tools, 18 generator functions, varied phrasing)
2. **Language Bias**: English only
   - Mitigation: Flagged in model card; future work planned

### Bias Analysis
- ✅ Tool category distribution analyzed (no extreme skew)
- ✅ Geographic representation in examples (if applicable)
- ✅ Function complexity distribution balanced

## Data Versioning & Reproducibility

### Version Control
- **Git**: Code and configs in Git
- **Gitignored**: `hf_cache/`, `data/processed/`, `data/raw/synthetic/` outputs

### Reproducibility Checklist
✅ All preprocessing in `scripts/prepare_datasets.sh` + `src/data_loader.py`
✅ Synthetic generation in `scripts/generate_synthetic.py` with `--seed 42`
✅ Random seed fixed (42) for all sampling, splitting, and balancing
✅ Dataset source URLs and configs pinned in `configs/dataset_config.yaml`
✅ License attribution preserved

## Known Limitations

1. **Synthetic Only**: All data is template-generated; real-world API distributions may differ
2. **Domain Coverage**: 15 tools (weather, stocks, translation, etc.); does not cover all possible APIs
3. **English Only**: No multilingual examples
4. **No Error Cases**: Limited error handling / negative examples

## Future Improvements

- [ ] Add more multilingual examples (Spanish, Mandarin, French)
- [ ] Expand error handling & edge cases
- [ ] Add streaming/real-time API examples
- [ ] Include authentication + credential handling patterns
- [ ] Cross-validate with additional tool-use benchmarks

## Contact

**Dataset Maintainer**: Dhruva N
**HuggingFace**: [dhruvanmurthy/qwen3-8b-tool-use](https://huggingface.co/datasets/dhruvanmurthy/qwen3-8b-tool-use)
**Issues**: GitHub issue tracker

---
**Last Updated**: March 2026
**Version**: 1.0
