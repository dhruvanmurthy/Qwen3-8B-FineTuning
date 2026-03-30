"""
Data loader and preprocessing utilities for Qwen3-8B fine-tuning.
Handles dataset loading, tokenization, and formatting.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

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
            dataset = self.load_single_dataset(source_name, source_config)

            # Add source metadata
            dataset = dataset.map(
                lambda x: {**x, "source": source_name},
                desc=f"Adding source metadata for {source_name}"
            )

            datasets.append(dataset)

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

        logger.info(f"Loading {url} ({split})...")

        if dataset_config:
            dataset = load_dataset(
                url,
                dataset_config,
                split=split,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
        else:
            dataset = load_dataset(
                url,
                split=split,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )

        # Limit samples
        if samples and len(dataset) > samples:
            indices = np.random.choice(len(dataset), samples, replace=False)
            dataset = dataset.select(indices)

        logger.info(f"Loaded {len(dataset)} examples from {url}")
        return dataset

    def _load_from_local(self, config: Dict) -> Dataset:
        """Load dataset from local JSONL/JSON files."""
        path = Path(config["path"])
        samples = config.get("samples", None)

        logger.info(f"Loading from {path}...")

        data = []
        if path.is_file():
            with open(path) as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            for file_path in path.glob("*.jsonl"):
                with open(file_path) as f:
                    for line in f:
                        data.append(json.loads(line))

        # Limit samples (random for consistency with hub loader)
        if samples and len(data) > samples:
            indices = np.random.choice(len(data), samples, replace=False)
            data = [data[i] for i in indices]

        logger.info(f"Loaded {len(data)} examples from {path}")
        return Dataset.from_dict({
            "text": [json.dumps(d) for d in data]
        })

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
        """Check if example is complete."""
        # Check required fields based on source
        required = ["messages"] if isinstance(example.get("messages"), list) else ["text"]
        return all(field in example and example[field] for field in required)

    def _normalize(self, example: Dict) -> Dict:
        """Normalize whitespace in text."""
        if "text" in example and isinstance(example["text"], str):
            example["text"] = " ".join(example["text"].split())
        return example

    def split_dataset(
        self, dataset: Dataset
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Split into train/val/test."""
        split_config = self.config["splits"]
        train_ratio = split_config["train"]
        val_ratio = split_config["validation"]

        # Shuffle
        dataset = dataset.shuffle(seed=self.seed)

        # Split
        split_data = dataset.train_test_split(
            test_size=1 - train_ratio,
            seed=self.seed
        )

        train_data = split_data["train"]
        remaining = split_data["test"]

        # Remaining split
        val_test_ratio = val_ratio / (1 - train_ratio)
        split_data = remaining.train_test_split(
            test_size=1 - val_test_ratio,
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
        """Balance dataset across sources."""
        from collections import Counter

        source_counts = Counter(dataset["source"])
        min_count = min(source_counts.values())

        balanced = []
        for source_name in source_counts:
            subset = dataset.filter(lambda x: x["source"] == source_name)
            indices = np.random.choice(len(subset), min_count, replace=False)
            balanced.append(subset.select(indices))

        return concatenate_datasets(balanced)

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
            system_prompt = (
                "You are a helpful assistant that can use tools. "
                "When you need to use a tool, respond with a JSON object "
                'containing "name" and "arguments" fields.'
            )

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
