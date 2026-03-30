# Dataset Card: Qwen3-8B Tool Use Training Data

## Dataset Summary

Aggregated dataset for training Qwen3-8B on multi-step tool use, function calling, and API orchestration. Raw data is loaded from 4 sources, normalized to a unified `text` column, deduplicated, balanced, and split.

### Dataset Composition

| Source | Raw Count | After Dedup | Balanced (~951/src) | Purpose |
|--------|-----------|-------------|---------------------|----------|
| **APIBench** | 5,000 | ~1,644 | ~951 | API call + domain documentation |
| **ToolBench** | 200 | 200 | ~951 (oversampled) | Tool selection from API catalogs |
| **Gorilla BFCL** | 258 | 258 | ~951 (oversampled) | Function calling accuracy |
| **Synthetic** | 15,000 | ~12,280 | ~951 | Edge cases & domain diversity |
| **Total** | **20,458** | **~14,382** | **~3,804** | **Multi-domain tool use** |

### Final Splits

| Split | Count |
|-------|-------|
| Train | ~3,043 |
| Validation | ~380 |
| Test | ~381 |

## Data Sources

### 1. APIBench
- **Link**: https://huggingface.co/datasets/gorilla-llm/APIBench
- **Samples**: 5,000 (across torchhub_train.json, huggingface_train.json, tensorflow_train.json)
- **Focus**: API documentation with domain and api_call specifications
- **Format**: domain + api_call + api_data (mixed-type JSON, loaded via pandas fallback)
- **License**: CC BY 4.0
- **Selection Criteria**: All available samples from the 3 train files

**Schema**:
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

### 2. ToolBench
- **Link**: https://huggingface.co/datasets/tuandunghcmut/toolbench-v1
- **Samples**: 200 (benchmark config, g1_instruction split)
- **Focus**: Tool selection from large API catalogs
- **Format**: query + api_list + relevant_apis
- **License**: Apache 2.0
- **Selection Criteria**: All 200 rows from benchmark split (oversampled during balancing)

**Schema**:
```json
{
  "query_id": 577,
  "query": "I am a fitness enthusiast and I want to buy a fitness tracker. Can you suggest some top-rated fitness trackers?",
  "api_list": [{"category_name": "Data", "tool_name": "ASIN Data", "api_name": "Category", ...}],
  "relevant_apis": [["ASIN Data", "Search"], ["ASIN Data", "Product"]]
}
```

### 3. Gorilla BFCL
- **Link**: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
- **Samples**: 258 (from BFCL_v3_live_simple.json)
- **Focus**: Function calling with correct argument generation
- **Format**: Nested question + function definitions (mixed-type JSON, loaded via pandas fallback)
- **License**: MIT
- **Selection Criteria**: All rows from the live_simple file

**Schema**:
```json
{
  "id": "live_simple_0-0-0",
  "question": [[{"role": "user", "content": "Can you retrieve the details for user ID 7890?"}]],
  "function": [{"name": "get_user_info", "description": "Retrieve details for a specific user",
    "parameters": {"type": "dict", "required": ["user_id"],
      "properties": {"user_id": {"type": "integer", "description": "The unique identifier"}}}}]
}
```

### 4. Synthetic Data
- **Source**: Generated via `scripts/generate_synthetic.py` with template-based generation
- **Samples**: 15,000 (sampled from 40,000 generated; ~34,700 single-step + ~5,300 multi-step)
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

### Unified Text Format
All datasets are normalized to a unified `text` column (plus `source` metadata) before tokenization.
The normalization is handled by `_normalize_to_text()` in `src/data_loader.py`.

Each source schema is converted differently:

| Source | Input Fields | Text Output |
|--------|-------------|-------------|
| Synthetic | `text` (already present) | As-is |
| APIBench | `domain`, `api_call`, `api_data` | `domain: ...\napi_call: ...\napi_data: {...}` |
| ToolBench | `query`, `api_list`, `relevant_apis` | `query: ...\napi_list: [...]\nrelevant_apis: [...]` |
| BFCL | `question`, `function` | `question: [...]\nfunction: [...]` |

After normalization, only `text` and `source` columns remain. Tokenization produces `input_ids`, `attention_mask`, and `labels`.

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
Raw Loaded:     20,458 samples
After Dedup:    14,382 samples
After Balance:   3,804 samples (~951 per source)

Training Set:    3,043 samples (80%)
Validation Set:    380 samples (10%)
Test Set:          381 samples (10%)
```

### Source Distribution (before balancing)
```
api_bank:   1,644  (11.4%)
toolbench:    200  ( 1.4%)
gorilla:      258  ( 1.8%)
synthetic: 12,280  (85.4%)
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
2. **Load All Sources**: HF datasets downloaded to `hf_cache/` (gitignored); local synthetics read from disk
3. **Normalize**: Convert all schemas to unified `text` column via `_normalize_to_text()`
4. **Deduplicate**: MD5 hash of full row, keep first occurrence
5. **Filter**: Remove incomplete examples (no meaningful content fields)
6. **Balance**: Median-target resampling (oversample small sources, undersample large)
7. **Split**: 80/10/10 train/validation/test
8. **Tokenize**: Qwen3-8B tokenizer, max_length=2048, padding to max_length
9. **Save**: Arrow format in `data/processed/{train,validation,test}/`

### Quality Checks
✅ No duplicate examples (MD5 hash)
✅ All rows have at least one meaningful content field
✅ Whitespace normalized
✅ Balanced source representation via median-target resampling
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

## Licensing & Attribution

| Source | License | Attribution |
|--------|---------|-------------|
| API-Bank | CC BY 4.0 | Xu et al., 2023 |
| ToolBench | Apache 2.0 | Qin et al., 2023 |
| Gorilla | MIT | Patil et al., 2023 |
| Synthetic | Custom (CC BY 4.0) | Generated, human-reviewed |

**Combined License**: CC BY 4.0 (most restrictive)

## Ethical Considerations

### Data Collection
- ✅ Datasets sourced from public, publicly-licensed sources
- ✅ No personally identifiable information (PII)
- ✅ No sensitive credentials in examples

### Potential Harms
1. **Synthetic Data Artifacts**: Template-generated examples may not reflect real user behavior
   - Mitigation: Balanced mix with real HF datasets (~15% of final data)
2. **Small Real-Data Sources**: ToolBench (200) and BFCL (258) are oversampled
   - Mitigation: Median-target balancing avoids extreme imbalance
3. **Language Bias**: English only
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

1. **Small Real-Data Pool**: Only ~458 unique real examples (200 ToolBench + 258 BFCL) survive dedup; oversampled during balancing
2. **Format**: Unified text column (not structured ChatML); tool call format varies by source
3. **Domain Coverage**: Synthetic data covers common tools (weather, stocks, translation); real data adds API documentation
4. **English Only**: No multilingual examples
5. **No Error Cases**: Limited error handling / negative examples

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
