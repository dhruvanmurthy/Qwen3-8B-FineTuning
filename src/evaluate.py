"""
Three-stage evaluation for Qwen3-8B tool-use fine-tuning.

Compares: Baseline (zero-shot) -> SFT -> GRPO
Metrics: tool selection accuracy, argument accuracy, multi-step success, latency.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------------------
# Model loading helpers
# -----------------------------------------------------------------------

def _load_base_model(model_id: str):
    """Load base model in bf16 for evaluation."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _load_adapter_model(model_id: str, adapter_path: str):
    """Load base model + LoRA adapter for evaluation."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# -----------------------------------------------------------------------
# Core evaluator
# -----------------------------------------------------------------------

class ToolUseEvaluator:
    """Evaluate a single model on tool-use benchmarks."""

    def __init__(self, model, tokenizer, label: str = "model"):
        self.model = model
        self.tokenizer = tokenizer
        self.label = label
        self.results: Dict[str, float] = {}

    # --- generation ---

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- tool extraction ---

    @staticmethod
    def extract_tool_name(text: str) -> Optional[str]:
        import re
        patterns = [
            r'"name":\s*"(\w+)"',
            r'\[(\w+)\]',
            r'use\s+(\w+)',
            r'(\w+)\(',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    # --- benchmarks ---

    def evaluate_tool_selection(
        self, dataset_name: str = "api-bank", num_samples: int = 500
    ) -> float:
        logger.info("[%s] Evaluating tool selection (%s)…", self.label, dataset_name)

        if dataset_name == "api-bank":
            dataset = load_dataset(
                "gorilla-llm/APIBench",
                data_files="torchhub_train.json",
                split="train",
                trust_remote_code=True,
            )
        elif dataset_name == "toolbench":
            dataset = load_dataset(
                "tuandunghcmut/toolbench-v1", "benchmark",
                split="g1_instruction",
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        if len(dataset) > num_samples:
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            dataset = dataset.select(indices)

        correct = total = 0
        for i, example in enumerate(dataset):
            if i % 50 == 0:
                logger.info("  progress: %d/%d", i, len(dataset))

            expected_tool = example.get("expected_tool")
            if not expected_tool and "expected_calls" in example:
                calls = example["expected_calls"]
                if isinstance(calls, list) and calls:
                    expected_tool = calls[0].get("tool")
            if not expected_tool:
                continue

            prompt = example.get("instruction") or example.get("user_instruction")
            if not prompt:
                continue

            output = self.generate(prompt, max_new_tokens=256)
            predicted = self.extract_tool_name(output)

            if predicted and predicted.lower() == expected_tool.lower():
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        key = f"tool_selection_accuracy_{dataset_name}"
        self.results[key] = accuracy
        logger.info("[%s] Tool Selection (%s): %.1f%%", self.label, dataset_name, 100 * accuracy)
        return accuracy

    def evaluate_argument_accuracy(self, num_samples: int = 500) -> float:
        logger.info("[%s] Evaluating argument accuracy…", self.label)
        dataset = load_dataset(
            "gorilla-llm/APIBench",
            data_files="torchhub_train.json",
            split="train",
            trust_remote_code=True,
        )

        if len(dataset) > num_samples:
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            dataset = dataset.select(indices)

        correct = total = 0
        for example in dataset:
            expected_calls = example.get("expected_calls") or example.get("api_calls")
            if not isinstance(expected_calls, list) or not expected_calls:
                continue
            expected_args = expected_calls[0].get("arguments", {})
            prompt = example.get("instruction") or example.get("user_instruction")
            if not prompt:
                continue

            output = self.generate(prompt, max_new_tokens=256)

            from rewards import extract_tool_call
            call = extract_tool_call(output)
            if call:
                pred_args = call.get("arguments", call.get("parameters", {}))
                if pred_args == expected_args:
                    correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        self.results["argument_accuracy"] = accuracy
        logger.info("[%s] Argument Accuracy: %.1f%%", self.label, 100 * accuracy)
        return accuracy

    def evaluate_schema_compliance(self, num_samples: int = 500) -> float:
        logger.info("[%s] Evaluating schema compliance…", self.label)
        dataset = load_dataset(
            "gorilla-llm/APIBench",
            data_files="torchhub_train.json",
            split="train",
            trust_remote_code=True,
        )

        if len(dataset) > num_samples:
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            dataset = dataset.select(indices)

        valid = total = 0
        for example in dataset:
            prompt = example.get("instruction") or example.get("user_instruction")
            if not prompt:
                continue
            output = self.generate(prompt, max_new_tokens=256)

            from rewards import extract_tool_call
            call = extract_tool_call(output)
            if (
                call
                and isinstance(call.get("name"), str)
                and isinstance(call.get("arguments", call.get("parameters", {})), dict)
            ):
                valid += 1
            total += 1

        rate = valid / total if total > 0 else 0
        self.results["schema_compliance"] = rate
        logger.info("[%s] Schema Compliance: %.1f%%", self.label, 100 * rate)
        return rate

    def evaluate_multi_step(self, num_samples: int = 500) -> float:
        logger.info("[%s] Evaluating multi-step chains…", self.label)
        dataset = load_dataset(
            "gorilla-llm/APIBench",
            data_files="torchhub_train.json",
            split="train",
            trust_remote_code=True,
        )

        multi_step = dataset.filter(lambda x: len(x.get("api_calls", [])) > 1)
        if len(multi_step) > num_samples:
            indices = np.random.choice(len(multi_step), num_samples, replace=False)
            multi_step = multi_step.select(indices)

        correct = total = 0
        for example in multi_step:
            prompt = example["instruction"]
            expected_calls = [c.get("tool") for c in example.get("api_calls", [])]
            if not expected_calls or len(expected_calls) < 2:
                continue

            output = self.generate(prompt, max_new_tokens=512)
            found = sum(1 for t in expected_calls if t.lower() in output.lower())
            if found == len(expected_calls):
                correct += 1
            total += 1

        rate = correct / total if total > 0 else 0
        self.results["multi_step_success"] = rate
        logger.info("[%s] Multi-Step Success: %.1f%%", self.label, 100 * rate)
        return rate

    def evaluate_latency(self, num_samples: int = 100) -> float:
        logger.info("[%s] Evaluating latency…", self.label)
        dataset = load_dataset(
            "gorilla-llm/APIBench",
            data_files="torchhub_train.json",
            split="train",
            trust_remote_code=True,
        )
        if len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))

        times, total_tokens = [], 0
        for example in dataset:
            prompt = example["instruction"]
            start = time.time()
            output = self.generate(prompt, max_new_tokens=256)
            elapsed = time.time() - start
            n_tok = len(self.tokenizer(output)["input_ids"])
            times.append(elapsed)
            total_tokens += n_tok

        avg_time = float(np.mean(times))
        ms_per_token = 1000 * avg_time / (total_tokens / len(times))
        self.results["avg_latency_ms"] = avg_time * 1000
        self.results["ms_per_token"] = ms_per_token
        logger.info("[%s] Latency: %.3fs (%.1f ms/token)", self.label, avg_time, ms_per_token)
        return ms_per_token

    def evaluate_all(self) -> Dict[str, float]:
        self.evaluate_tool_selection("api-bank", num_samples=500)
        self.evaluate_tool_selection("toolbench", num_samples=1000)
        self.evaluate_argument_accuracy(num_samples=500)
        self.evaluate_schema_compliance(num_samples=500)
        self.evaluate_multi_step(num_samples=500)
        self.evaluate_latency(num_samples=100)
        return self.results


# -----------------------------------------------------------------------
# Comparison
# -----------------------------------------------------------------------

def compare_stages(all_results: Dict[str, Dict[str, float]]) -> str:
    """Generate a markdown comparison table from {stage: results} dict."""

    stages = list(all_results.keys())
    all_keys = []
    for res in all_results.values():
        for k in res:
            if k not in all_keys:
                all_keys.append(k)

    # Header
    header = "| Metric | " + " | ".join(stages) + " |"
    sep = "|---|" + "|".join(["---"] * len(stages)) + "|"
    rows = [header, sep]

    for key in all_keys:
        vals = []
        for stage in stages:
            v = all_results[stage].get(key)
            if v is None:
                vals.append("—")
            elif "accuracy" in key or "success" in key or "compliance" in key:
                vals.append(f"{100 * v:.1f}%")
            else:
                vals.append(f"{v:.2f}")
        rows.append(f"| {key} | " + " | ".join(vals) + " |")

    table = "\n".join(rows)
    print("\n" + "=" * 70)
    print("STAGE COMPARISON")
    print("=" * 70)
    print(table)
    print("=" * 70 + "\n")
    return table


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Three-stage evaluation")
    parser.add_argument(
        "--mode",
        choices=["baseline", "sft", "grpo", "compare", "all"],
        default="all",
        help="Which stage(s) to evaluate",
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--sft-adapter", default="./outputs/sft")
    parser.add_argument("--grpo-adapter", default="./outputs/grpo")
    parser.add_argument("--output", default="outputs/evaluation_results.json")
    args = parser.parse_args()

    all_results = {}

    # --- baseline ---
    if args.mode in ("baseline", "compare", "all"):
        logger.info(">>> Baseline evaluation")
        model, tok = _load_base_model(args.base_model)
        ev = ToolUseEvaluator(model, tok, label="baseline")
        all_results["baseline"] = ev.evaluate_all()
        del model, tok
        torch.cuda.empty_cache()

    # --- SFT ---
    if args.mode in ("sft", "compare", "all"):
        logger.info(">>> SFT evaluation")
        model, tok = _load_adapter_model(args.base_model, args.sft_adapter)
        ev = ToolUseEvaluator(model, tok, label="sft")
        all_results["sft"] = ev.evaluate_all()
        del model, tok
        torch.cuda.empty_cache()

    # --- GRPO ---
    if args.mode in ("grpo", "compare", "all"):
        logger.info(">>> GRPO evaluation")
        # GRPO adapter sits on top of SFT-merged weights.
        # Load base -> merge SFT -> load GRPO adapter.
        model, tok = _load_base_model(args.base_model)
        if os.path.isdir(args.sft_adapter):
            model = PeftModel.from_pretrained(model, args.sft_adapter)
            model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, args.grpo_adapter)
        ev = ToolUseEvaluator(model, tok, label="grpo")
        all_results["grpo"] = ev.evaluate_all()
        del model, tok
        torch.cuda.empty_cache()

    # --- comparison table ---
    if len(all_results) > 1:
        compare_stages(all_results)

    # --- save ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
