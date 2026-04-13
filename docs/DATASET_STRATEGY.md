# Dataset Strategy

The active data pipeline in this repository is fully synthetic and local-first.

## Canonical Flow

The current dataset path is:

1. generate synthetic examples with `scripts/generate_synthetic.py`
2. load them through `ToolUseDataLoader`
3. preprocess and balance the examples
4. split them into train/validation/test
5. save:
   - tokenized splits under `data/processed/`
   - structured evaluation rows under `data/processed/test_raw.jsonl`

The canonical shell entrypoint is:

```bash
bash scripts/prepare_datasets.sh
```

## Synthetic Generation

Generator script:

```bash
python scripts/generate_synthetic.py --num-samples 40000 --output-dir data/raw/synthetic
```

By default, `scripts/prepare_datasets.sh` generates `40000` raw examples.

The generator writes:

- `data/raw/synthetic/synthetic_single.jsonl`
- `data/raw/synthetic/synthetic_multistep.jsonl`

Each example contains structured fields such as:

- `instruction`
- `tools`
- `tool_calls`
- `category`
- `num_steps`
- `text`

## Loader Configuration

`configs/dataset_config.yaml` is the live configuration file.

Key defaults:

- source path: `./data/raw/synthetic/`
- source type: `local`
- sample cap: `15000`
- split ratios: `0.8 / 0.1 / 0.1`
- source balancing: enabled

That means the raw generator can create more examples than the loader keeps for
the current training run.

## Preprocessing

`ToolUseDataLoader` currently performs:

- duplicate removal
- incomplete-example filtering
- whitespace normalization
- optional source balancing
- train/validation/test splitting
- tokenization for training outputs

## Expected Sizes

Exact sizes vary from run to run because of generation, deduplication, and
balancing. With the default configuration:

- raw generation starts at `40000` examples
- the loader samples up to `15000`
- preprocessing often lands near the low tens of thousands
- the final tokenized dataset is split approximately `80/10/10`

Treat the split sizes as approximate rather than fixed constants.

## Why `test_raw.jsonl` Exists

The evaluation pipeline needs more than token IDs. It needs:

- prompt text
- expected tool calls
- tool metadata
- source labels

`scripts/prepare_datasets.sh` therefore saves a raw structured test split to
`data/processed/test_raw.jsonl` before text normalization and tokenization.

This file is the preferred evaluation source for `src/evaluate.py`.

## Inspecting the Prepared Dataset

```bash
python -c "
from datasets import load_from_disk
ds = load_from_disk('data/processed')
print(ds)
print(ds['train'].column_names)
"
```

## Optional Upload

If you want to publish the raw synthetic dataset:

```bash
python scripts/push_dataset_to_hub.py \
  --repo-id your-user/qwen3-8b-synthetic-tool-use \
  --data-dir data/raw/synthetic
```
