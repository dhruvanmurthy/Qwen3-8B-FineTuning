"""
Data loader and preprocessing utilities for Qwen3-8B fine-tuning.
Handles dataset loading, tokenization, and formatting.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from constants import TOOL_USE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ToolUseDataLoader:
    """Load and preprocess tool-use datasets."""

    def __init__(self, config_path: str = "configs/dataset_config.yaml"):
        """Initialize data loader."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.cache_dir = self.config.get("cache_dir", "./hf_cache")
        self.seed = self.config.get("seed", 42)

    def load_all_datasets(self) -> Dataset:
        """Load all configured datasets and concatenate."""
        datasets = []

        for source_name, source_config in self.config["sources"].items():
            if not source_config.get("enabled", True):
                logger.info(f"Skipping {source_name}")
                continue

            logger.info(f"Loading {source_name}...")
            try:
                dataset = self.load_single_dataset(source_name, source_config)

                # Add source metadata
                _sname = source_name  # capture for lambda closure
                dataset = dataset.map(
                    lambda x, s=_sname: {**x, "source": s},
                    desc=f"Adding source metadata for {source_name}"
                )

                datasets.append(dataset)
                logger.info(f"✓ Loaded {source_name}: {len(dataset)} examples")
            except Exception as exc:
                logger.warning(
                    f"⚠ Skipping {source_name}: {exc}. "
                    "Check HF access / network / local path and rerun if needed."
                )

        if not datasets:
            raise RuntimeError(
                "No datasets were successfully loaded. "
                "Check HF_TOKEN, network connectivity, and data/raw/synthetic/ contents."
            )

        # Concatenate
        logger.info("Concatenating datasets...")
        combined = concatenate_datasets(datasets)

        return combined

    def load_single_dataset(self, name: str, config: Dict) -> Dataset:
        """Load a single dataset from HF or local path."""
        if config["type"] == "huggingface":
            return self._load_from_hub(config)
        elif config["type"] == "local":
            return self._load_from_local(config)
        else:
            raise ValueError(f"Unknown dataset type: {config['type']}")

    def _load_from_hub(self, config: Dict) -> Dataset:
        """Load dataset from Hugging Face Hub."""
        url = config["url"]
        split = config.get("split", "train")
        dataset_config = config.get("config", None)
        samples = config.get("samples", None)
        data_files = config.get("data_files", None)

        logger.info(f"Loading {url} ({split})...")

        hf_token = os.getenv("HF_TOKEN") or None

        kwargs = dict(
            cache_dir=self.cache_dir,
            token=hf_token,
        )
        if data_files:
            kwargs["data_files"] = data_files
        if dataset_config:
            args = (url, dataset_config)
        else:
            args = (url,)

        try:
            dataset = load_dataset(*args, split=split, **kwargs)
        except Exception as first_err:
            # Fallback for mixed-schema JSON: load each file individually
            # via pandas which is more lenient, then combine.
            if data_files:
                logger.info(
                    f"Default loader failed ({first_err}); "
                    "falling back to per-file pandas load..."
                )
                dataset = self._load_files_via_pandas(url, data_files, hf_token)
            else:
                raise first_err

        # Limit samples
        if samples and len(dataset) > samples:
            indices = np.random.choice(len(dataset), samples, replace=False)
            dataset = dataset.select(indices)

        logger.info(f"Loaded {len(dataset)} examples from {url}")
        return dataset

    def _load_files_via_pandas(
        self, repo_id: str, data_files: List[str], token: str | None
    ) -> Dataset:
        """Download individual files from a HF dataset repo and load via
        pandas to tolerate mixed-schema JSON fields."""
        import pandas as pd
        from huggingface_hub import hf_hub_download

        frames = []
        for fname in data_files:
            local = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                repo_type="dataset",
                cache_dir=self.cache_dir,
                token=token,
            )
            # Read as line-delimited JSON (each line is one object)
            try:
                df = pd.read_json(local, lines=True)
            except ValueError:
                # Some files are regular JSON arrays
                df = pd.read_json(local)
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        # Convert every column to string to avoid Arrow type conflicts
        for col in combined.columns:
            combined[col] = combined[col].apply(
                lambda v: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            )
        return Dataset.from_pandas(combined, preserve_index=False)

    def _load_from_local(self, config: Dict) -> Dataset:
        """Load dataset from local JSONL/JSON files."""
        path = Path(config["path"])
        samples = config.get("samples", None)

        logger.info(f"Loading from {path}...")

        data = []
        if path.is_file():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        elif path.is_dir():
            jsonl_files = list(path.glob("*.jsonl")) + list(path.glob("*.json"))
            for file_path in jsonl_files:
                with open(file_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
        else:
            raise FileNotFoundError(f"Local dataset path not found: {path}")

        if not data:
            raise ValueError(
                f"No data found in {path}. "
                "Run 'python scripts/generate_synthetic.py' to create synthetic data first."
            )

        # Limit samples (random for consistency with hub loader)
        if samples and len(data) > samples:
            indices = np.random.choice(len(data), samples, replace=False)
            data = [data[i] for i in indices]

        logger.info(f"Loaded {len(data)} examples from {path}")

        # Build column-oriented dict preserving all structured fields
        all_keys = set()
        for d in data:
            all_keys.update(d.keys())

        columns = {key: [] for key in sorted(all_keys)}
        for d in data:
            for key in columns:
                val = d.get(key)
                # Serialize complex types to strings to avoid Arrow type conflicts
                if isinstance(val, (dict, list)):
                    val = json.dumps(val)
                columns[key].append(val if val is not None else "")

        return Dataset.from_dict(columns)

    def preprocess(self, dataset: Dataset) -> Dataset:
        """Apply preprocessing operations."""

        # Deduplicate
        if self.config["preprocessing"].get("remove_duplicates", True):
            logger.info("Deduplicating...")
            dataset = self._deduplicate(dataset)

        # Remove incomplete
        if self.config["preprocessing"].get("remove_incomplete", True):
            logger.info("Removing incomplete examples...")
            dataset = dataset.filter(self._is_complete)

        # Normalize whitespace
        dataset = dataset.map(
            self._normalize, desc="Normalizing whitespace"
        )

        return dataset

    def _deduplicate(self, dataset: Dataset) -> Dataset:
        """Remove duplicate examples."""
        import hashlib

        seen_hashes = set()
        indices_to_keep = []

        for idx, example in enumerate(dataset):
            text = json.dumps(example, sort_keys=True)
            hash_val = hashlib.md5(text.encode()).hexdigest()

            if hash_val not in seen_hashes:
                seen_hashes.add(hash_val)
                indices_to_keep.append(idx)

        logger.info(f"Removed {len(dataset) - len(indices_to_keep)} duplicates")
        return dataset.select(indices_to_keep)

    def _is_complete(self, example: Dict) -> bool:
        """Check if example has at least one meaningful content field."""
        content_fields = ["text", "messages", "conversations", "id", "instruction",
                          "question", "domain", "api_call", "function"]
        return any(
            field in example and example[field]
            for field in content_fields
        )

    def _normalize(self, example: Dict) -> Dict:
        """Normalize whitespace in text."""
        if "text" in example and isinstance(example["text"], str):
            example["text"] = " ".join(example["text"].split())
        return example

    def _normalize_to_text(self, dataset: Dataset) -> Dataset:
        """Convert any dataset schema into a unified 'text' column.

        Handles:
          - Already has 'text' column → keep as-is
          - ToolBench 'conversations' (list of {from, value}) → render chat
          - APIBench 'api_call' + 'domain' → render tool call
          - BFCL 'question' + 'function' → render function call
          - Messages [{role, content}] → render chat
          - Fallback: JSON-dump entire row
        """
        cols = set(dataset.column_names)

        def _row_to_text(example: Dict) -> Dict:
            # 1. Already has text
            if example.get("text"):
                return example

            # 2. ToolBench conversations: dict with 'from'/'value' or list of turns
            if example.get("conversations"):
                convs = example["conversations"]
                if isinstance(convs, dict) and "from" in convs and "value" in convs:
                    parts = []
                    for role, val in zip(convs["from"], convs["value"]):
                        parts.append(f"{role}: {val}")
                    example["text"] = "\n".join(parts)
                    return example
                if isinstance(convs, list):
                    parts = []
                    for turn in convs:
                        r = turn.get("from", turn.get("role", ""))
                        v = turn.get("value", turn.get("content", ""))
                        parts.append(f"{r}: {v}")
                    example["text"] = "\n".join(parts)
                    return example

            # 3. ToolBench benchmark: query + api_list + relevant_apis
            if example.get("query") and example.get("api_list"):
                parts = [f"query: {example['query']}"]
                api_list = example["api_list"]
                if isinstance(api_list, str):
                    parts.append(f"api_list: {api_list}")
                else:
                    parts.append(f"api_list: {json.dumps(api_list, default=str)}")
                if example.get("relevant_apis"):
                    ra = example["relevant_apis"]
                    if isinstance(ra, str):
                        parts.append(f"relevant_apis: {ra}")
                    else:
                        parts.append(f"relevant_apis: {json.dumps(ra, default=str)}")
                example["text"] = "\n".join(parts)
                return example

            # 4. APIBench: domain + api_call + optional api_data
            if example.get("api_call"):
                parts = []
                if example.get("domain"):
                    parts.append(f"domain: {example['domain']}")
                parts.append(f"api_call: {example['api_call']}")
                if example.get("api_data"):
                    api_data = example["api_data"]
                    if isinstance(api_data, str):
                        parts.append(f"api_data: {api_data}")
                    else:
                        parts.append(f"api_data: {json.dumps(api_data)}")
                example["text"] = "\n".join(parts)
                return example

            # 5. BFCL: question + function list
            if example.get("question"):
                parts = [f"question: {json.dumps(example['question'], default=str)}"]
                if example.get("function"):
                    fn = example["function"]
                    if isinstance(fn, str):
                        parts.append(f"function: {fn}")
                    else:
                        parts.append(f"function: {json.dumps(fn, default=str)}")
                example["text"] = "\n".join(parts)
                return example

            # 6. messages list [{role, content}]
            if isinstance(example.get("messages"), list):
                parts = []
                for msg in example["messages"]:
                    r = msg.get("role", "")
                    c = msg.get("content", "")
                    parts.append(f"{r}: {c}")
                example["text"] = "\n".join(parts)
                return example

            # 7. instruction field
            if example.get("instruction"):
                example["text"] = example["instruction"]
                return example

            # 8. Fallback: dump all non-null fields
            example["text"] = json.dumps(
                {k: v for k, v in example.items()
                 if k != "source" and v is not None},
                default=str,
            )
            return example

        logger.info("Normalizing all rows to unified 'text' column...")
        dataset = dataset.map(_row_to_text, desc="Normalizing to text")

        # Keep only text + source for a clean tokenization input
        keep_cols = {"text", "source"}
        drop_cols = [c for c in dataset.column_names if c not in keep_cols]
        if drop_cols:
            dataset = dataset.remove_columns(drop_cols)

        return dataset

    def split_dataset(
        self, dataset: Dataset
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Split into train/val/test."""
        split_config = self.config["splits"]
        train_ratio = split_config["train"]
        val_ratio = split_config["validation"]

        # Shuffle
        dataset = dataset.shuffle(seed=self.seed)

        total = len(dataset)
        # Need at least 1 sample per split — fallback for tiny datasets
        if total < 3:
            logger.warning(f"Dataset too small ({total}) for 3-way split. Using all as train.")
            empty = Dataset.from_dict({c: [] for c in dataset.column_names})
            return dataset, empty, empty

        # Split
        split_data = dataset.train_test_split(
            test_size=max(1 - train_ratio, 1 / total),
            seed=self.seed
        )

        train_data = split_data["train"]
        remaining = split_data["test"]

        if len(remaining) < 2:
            logger.warning("Not enough data for val/test split. Using remaining as val only.")
            empty = Dataset.from_dict({c: [] for c in dataset.column_names})
            return train_data, remaining, empty

        # Remaining split
        val_test_ratio = val_ratio / (1 - train_ratio)
        split_data = remaining.train_test_split(
            test_size=max(1 - val_test_ratio, 1 / len(remaining)),
            seed=self.seed
        )

        val_data = split_data["train"]
        test_data = split_data["test"]

        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        return train_data, val_data, test_data

    def tokenize_dataset(
        self,
        dataset: Dataset,
        tokenizer,
        max_length: int = 2048
    ) -> Dataset:
        """Tokenize dataset."""

        def tokenize_function(examples):
            # Handle both "text" and "messages" formats
            texts = []
            for item in examples.get("text", examples.get("messages", [])):
                if isinstance(item, list):  # Messages format
                    texts.append(self._format_messages(item))
                else:
                    texts.append(item)

            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )

            # Labels are a copy of input_ids (causal LM)
            tokenized["labels"] = [
                [(t if t != tokenizer.pad_token_id else -100) for t in ids]
                for ids in tokenized["input_ids"]
            ]

            return tokenized

        logger.info("Tokenizing dataset...")
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )

        return dataset

    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages list to string."""
        text_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            text_parts.append(f"{role}: {content}")
        return "\n".join(text_parts)

    def prepare_datasets(
        self,
        tokenizer,
        max_length: int = 2048
    ) -> DatasetDict:
        """Load, preprocess, and tokenize all datasets."""

        # Load
        dataset = self.load_all_datasets()
        logger.info(f"Total: {len(dataset)} examples")

        # Normalize every row to a unified 'text' column
        dataset = self._normalize_to_text(dataset)

        # Preprocess
        dataset = self.preprocess(dataset)

        # Balance
        if self.config.get("balance_sources", True):
            dataset = self._balance_sources(dataset)

        # Split
        train_data, val_data, test_data = self.split_dataset(dataset)

        # Tokenize
        train_data = self.tokenize_dataset(train_data, tokenizer, max_length)
        val_data = self.tokenize_dataset(val_data, tokenizer, max_length)
        test_data = self.tokenize_dataset(test_data, tokenizer, max_length)

        return DatasetDict({
            "train": train_data,
            "validation": val_data,
            "test": test_data
        })

    def _balance_sources(self, dataset: Dataset) -> Dataset:
        """Balance dataset across sources using median-target resampling.

        Uses the median source count as target instead of the minimum,
        which avoids collapsing the entire dataset when one source is tiny.
        Small sources are oversampled (with replacement); large sources are
        undersampled (without replacement).
        """
        from collections import Counter

        source_counts = Counter(dataset["source"])
        logger.info(f"Source distribution before balancing: {dict(source_counts)}")

        counts = sorted(source_counts.values())
        n = len(counts)
        if n == 0:
            return dataset

        # Median as target — keeps more data than min
        median = counts[n // 2] if n % 2 else (counts[n // 2 - 1] + counts[n // 2]) // 2
        # Floor: at least 500 (or the largest source if all are smaller)
        target_count = max(median, min(500, max(counts)))

        balanced = []
        for source_name in source_counts:
            subset = dataset.filter(lambda x, s=source_name: x["source"] == s)
            src_len = len(subset)
            if src_len >= target_count:
                indices = np.random.choice(src_len, target_count, replace=False)
            else:
                # Oversample small sources with replacement
                indices = np.random.choice(src_len, target_count, replace=True)
            balanced.append(subset.select(indices))

        result = concatenate_datasets(balanced)
        logger.info(f"Balanced to ~{target_count} per source, total: {len(result)}")
        return result

    # ------------------------------------------------------------------
    # GRPO prompt preparation
    # ------------------------------------------------------------------

    def prepare_grpo_prompts(self, tokenizer, system_prompt: str = None) -> Dataset:
        """Prepare a prompt-only dataset for GRPO training.

        Returns a Dataset with columns:
          prompt, expected_tool, expected_args, expected_chain, source
        """
        dataset = self.load_all_datasets()
        dataset = self.preprocess(dataset)

        if self.config.get("balance_sources", True):
            dataset = self._balance_sources(dataset)

        if system_prompt is None:
            system_prompt = TOOL_USE_SYSTEM_PROMPT

        chat_template_available = (
            hasattr(tokenizer, "apply_chat_template")
            and tokenizer.chat_template is not None
        )

        def _format_example(example):
            prompt_text = (
                example.get("instruction")
                or example.get("user_instruction")
                or example.get("query")
                or example.get("text", "")
            )

            # --- expected tool ---
            expected_tool = example.get("expected_tool", "")
            if not expected_tool:
                for key in ("expected_calls", "api_calls"):
                    calls = example.get(key)
                    if isinstance(calls, list) and calls:
                        first = calls[0]
                        expected_tool = (
                            first.get("tool")
                            or first.get("name", "")
                        )
                        break

            # --- expected args ---
            expected_args = ""
            for key in ("expected_calls", "api_calls"):
                calls = example.get(key)
                if isinstance(calls, list) and calls:
                    args = calls[0].get("arguments", calls[0].get("parameters", {}))
                    expected_args = json.dumps(args) if isinstance(args, dict) else str(args)
                    break

            # --- expected chain ---
            expected_chain = "[]"
            calls = example.get("api_calls")
            if isinstance(calls, list) and len(calls) > 1:
                chain = [
                    c.get("tool") or c.get("name", "") for c in calls
                ]
                expected_chain = json.dumps(chain)

            # --- formatted prompt ---
            if chat_template_available:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )

            return {
                "prompt": prompt,
                "expected_tool": expected_tool,
                "expected_args": expected_args,
                "expected_chain": expected_chain,
            }

        dataset = dataset.map(_format_example, desc="Formatting GRPO prompts")
        keep = {"prompt", "expected_tool", "expected_args", "expected_chain", "source"}
        drop = [c for c in dataset.column_names if c not in keep]
        if drop:
            dataset = dataset.remove_columns(drop)

        logger.info("GRPO prompt dataset: %d examples", len(dataset))
        return dataset


def create_data_loader(config_path: str = "configs/dataset_config.yaml") -> ToolUseDataLoader:
    """Factory function to create data loader."""
    return ToolUseDataLoader(config_path)
