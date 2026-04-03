"""Three-stage evaluation for Qwen3-8B tool-use fine-tuning.

Compares: Baseline (zero-shot) -> SFT -> GRPO
Metrics: tool selection accuracy, argument accuracy, schema compliance,
multi-step success, and latency.

Key fixes:
    - Uses held-out evaluation data (no train-split leakage)
    - Normalizes heterogeneous dataset schemas before scoring
    - Extracts only generated tokens (not prompt+generation concatenation)
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
        from peft import PeftModel
except ImportError:  # Optional at runtime for base-only evaluation
        PeftModel = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------------------
# Data model and normalization
# -----------------------------------------------------------------------


@dataclass
class EvalExample:
    """Normalized evaluation record across all source schemas."""

    prompt: str
    expected_tool: Optional[str]
    expected_args: Optional[Dict[str, Any]]
    expected_chain: List[str]
    source: str


def _stable_test_split(rows: List[Dict[str, Any]], ratio: float = 0.1) -> List[Dict[str, Any]]:
    """Deterministic holdout split without requiring upstream test split."""
    heldout: List[Dict[str, Any]] = []
    threshold = int(ratio * 100)
    for row in rows:
        key = json.dumps(row, sort_keys=True, default=str)
        bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 100
        if bucket < threshold:
            heldout.append(row)
    return heldout


def _json_load_maybe(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _extract_tool_from_api_call(api_call: Any) -> Optional[str]:
    if not api_call:
        return None
    if isinstance(api_call, dict):
        return api_call.get("name") or api_call.get("tool")
    if isinstance(api_call, str):
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\(", api_call)
        if m:
            return m.group(1)
    return None


def _normalize_row(row: Dict[str, Any], source: str) -> Optional[EvalExample]:
    prompt = (
        row.get("instruction")
        or row.get("user_instruction")
        or row.get("query")
        or row.get("question")
        or row.get("text")
    )
    if not isinstance(prompt, str) or not prompt.strip():
        return None

    expected_tool: Optional[str] = None
    expected_args: Optional[Dict[str, Any]] = None
    expected_chain: List[str] = []

    # Synthetic format
    tool_calls = _json_load_maybe(row.get("tool_calls"))
    if isinstance(tool_calls, list) and tool_calls:
        first = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
        expected_tool = first.get("name")
        raw_args = first.get("arguments", first.get("parameters", {}))
        if isinstance(raw_args, dict):
            expected_args = raw_args
        expected_chain = [c.get("name", "") for c in tool_calls if isinstance(c, dict)]

    # Older expected_calls / api_calls schema
    if not expected_tool:
        expected_calls = _json_load_maybe(row.get("expected_calls"))
        if isinstance(expected_calls, list) and expected_calls:
            first = expected_calls[0] if isinstance(expected_calls[0], dict) else {}
            expected_tool = first.get("tool") or first.get("name")
            raw_args = first.get("arguments", first.get("parameters", {}))
            if isinstance(raw_args, dict):
                expected_args = raw_args
            expected_chain = [
                (c.get("tool") or c.get("name") or "")
                for c in expected_calls
                if isinstance(c, dict)
            ]

    if not expected_tool:
        api_calls = _json_load_maybe(row.get("api_calls"))
        if isinstance(api_calls, list) and api_calls:
            first = api_calls[0] if isinstance(api_calls[0], dict) else {}
            expected_tool = first.get("tool") or first.get("name")
            raw_args = first.get("arguments", first.get("parameters", {}))
            if isinstance(raw_args, dict):
                expected_args = raw_args
            expected_chain = [
                (c.get("tool") or c.get("name") or "")
                for c in api_calls
                if isinstance(c, dict)
            ]

    # APIBench schema: api_call string
    if not expected_tool:
        expected_tool = _extract_tool_from_api_call(row.get("api_call"))

    # ToolBench fallback: relevant_apis is often a list of candidate tools.
    # Use the first relevant API as a weak single-tool target if needed.
    if not expected_tool:
        relevant_apis = _json_load_maybe(row.get("relevant_apis"))
        if isinstance(relevant_apis, list) and relevant_apis:
            first = relevant_apis[0]
            if isinstance(first, str):
                expected_tool = first
            elif isinstance(first, dict):
                expected_tool = first.get("name") or first.get("api")

    return EvalExample(
        prompt=prompt,
        expected_tool=expected_tool,
        expected_args=expected_args,
        expected_chain=[c for c in expected_chain if c],
        source=source,
    )


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_eval_examples(max_samples: int = 1000) -> List[EvalExample]:
    """Load held-out evaluation examples from local artifacts first, then HF.

    Priority:
      1) `data/processed/test_raw.jsonl` (if present)
      2) local synthetic data held-out split (10%)
      3) HF datasets held-out split (10%)
    """

    examples: List[EvalExample] = []

    # 1) Preferred local raw test set artifact
    test_raw = Path("data/processed/test_raw.jsonl")
    if test_raw.exists():
        rows = _load_jsonl_rows(test_raw)
        for row in rows:
            ex = _normalize_row(row, source=str(row.get("source", "processed_test")))
            if ex:
                examples.append(ex)
        if examples:
            logger.info("Loaded %d eval examples from %s", len(examples), test_raw)

    # 2) Local synthetic fallback
    if not examples:
        synth_dir = Path("data/raw/synthetic")
        if synth_dir.exists():
            synth_rows: List[Dict[str, Any]] = []
            for p in list(synth_dir.glob("*.jsonl")) + list(synth_dir.glob("*.json")):
                synth_rows.extend(_load_jsonl_rows(p))
            heldout = _stable_test_split(synth_rows, ratio=0.1)
            for row in heldout:
                ex = _normalize_row(row, source="synthetic")
                if ex:
                    examples.append(ex)
            if examples:
                logger.info("Loaded %d held-out synthetic eval examples", len(examples))

    # 3) HF fallback held-out
    if not examples:
        hf_rows: List[Dict[str, Any]] = []

        api_ds = load_dataset(
            "gorilla-llm/APIBench",
            data_files="torchhub_train.json",
            split="train",
        )
        hf_rows.extend([dict(r, source="api-bank") for r in api_ds])

        tb_ds = load_dataset(
            "tuandunghcmut/toolbench-v1",
            "benchmark",
            split="g1_instruction",
        )
        hf_rows.extend([dict(r, source="toolbench") for r in tb_ds])

        heldout = _stable_test_split(hf_rows, ratio=0.1)
        for row in heldout:
            ex = _normalize_row(row, source=str(row.get("source", "hf")))
            if ex:
                examples.append(ex)

        logger.info("Loaded %d held-out HF eval examples", len(examples))

    if len(examples) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(examples), max_samples, replace=False)
        examples = [examples[i] for i in idx]

    return examples


# -----------------------------------------------------------------------
# Model loading helpers
# -----------------------------------------------------------------------

def _load_base_model(model_id: str):
    """Load base model in bf16 for evaluation."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _load_adapter_model(model_id: str, adapter_path: str):
    """Load base model + LoRA adapter for evaluation."""
    if PeftModel is None:
        raise RuntimeError(
            "peft is not installed. Install it to evaluate adapter checkpoints."
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
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
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_ids = outputs[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # --- tool extraction ---

    @staticmethod
    def extract_tool_name(text: str) -> Optional[str]:
        patterns = [
            r'"name"\s*:\s*"([A-Za-z_][A-Za-z0-9_.-]*)"',
            r"<tool_call>.*?\"name\"\s*:\s*\"([A-Za-z_][A-Za-z0-9_.-]*)\".*?</tool_call>",
            r'\[([A-Za-z_][A-Za-z0-9_.-]*)\]',
            r'use\s+([A-Za-z_][A-Za-z0-9_.-]*)',
            r'([A-Za-z_][A-Za-z0-9_.-]*)\(',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    # --- benchmarks ---

    def evaluate_tool_selection(self, examples: List[EvalExample]) -> float:
        logger.info("[%s] Evaluating tool selection…", self.label)

        correct = total = 0
        for i, example in enumerate(examples):
            if i > 0 and i % 50 == 0:
                logger.info("  progress: %d/%d", i, len(examples))

            if not example.expected_tool:
                continue

            output = self.generate(example.prompt, max_new_tokens=256)
            predicted = self.extract_tool_name(output)

            if predicted and predicted.lower() == example.expected_tool.lower():
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        self.results["tool_selection_accuracy"] = accuracy
        logger.info("[%s] Tool Selection: %.1f%%", self.label, 100 * accuracy)
        return accuracy

    def evaluate_argument_accuracy(self, examples: List[EvalExample]) -> float:
        logger.info("[%s] Evaluating argument accuracy…", self.label)

        correct = total = 0
        for example in examples:
            if not example.expected_args:
                continue

            output = self.generate(example.prompt, max_new_tokens=256)

            from rewards import extract_tool_call
            call = extract_tool_call(output)
            if call:
                pred_args = call.get("arguments", call.get("parameters", {}))
                if pred_args == example.expected_args:
                    correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        self.results["argument_accuracy"] = accuracy
        logger.info("[%s] Argument Accuracy: %.1f%%", self.label, 100 * accuracy)
        return accuracy

    def evaluate_schema_compliance(self, examples: List[EvalExample]) -> float:
        logger.info("[%s] Evaluating schema compliance…", self.label)

        valid = total = 0
        for example in examples:
            output = self.generate(example.prompt, max_new_tokens=256)

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

    def evaluate_multi_step(self, examples: List[EvalExample]) -> float:
        logger.info("[%s] Evaluating multi-step chains…", self.label)
        multi_step = [e for e in examples if len(e.expected_chain) > 1]

        correct = total = 0
        for example in multi_step:
            output = self.generate(example.prompt, max_new_tokens=512)
            found = sum(1 for t in example.expected_chain if t.lower() in output.lower())
            if found == len(example.expected_chain):
                correct += 1
            total += 1

        rate = correct / total if total > 0 else 0
        self.results["multi_step_success"] = rate
        logger.info("[%s] Multi-Step Success: %.1f%%", self.label, 100 * rate)
        return rate

    def evaluate_latency(self, examples: List[EvalExample], num_samples: int = 100) -> float:
        logger.info("[%s] Evaluating latency…", self.label)
        subset = examples[:num_samples]

        times, total_tokens = [], 0
        for example in subset:
            start = time.time()
            output = self.generate(example.prompt, max_new_tokens=256)
            elapsed = time.time() - start
            n_tok = len(self.tokenizer(output)["input_ids"])
            times.append(elapsed)
            total_tokens += n_tok

        if not times or total_tokens == 0:
            self.results["avg_latency_ms"] = 0.0
            self.results["ms_per_token"] = 0.0
            return 0.0

        avg_time = float(np.mean(times))
        ms_per_token = 1000 * avg_time / (total_tokens / len(times))
        self.results["avg_latency_ms"] = avg_time * 1000
        self.results["ms_per_token"] = ms_per_token
        logger.info("[%s] Latency: %.3fs (%.1f ms/token)", self.label, avg_time, ms_per_token)
        return ms_per_token

    def evaluate_all(self, examples: List[EvalExample]) -> Dict[str, float]:
        self.evaluate_tool_selection(examples)
        self.evaluate_argument_accuracy(examples)
        self.evaluate_schema_compliance(examples)
        self.evaluate_multi_step(examples)
        self.evaluate_latency(examples, num_samples=100)
        # Log all aggregate metrics to W&B summary for this stage
        for k, v in self.results.items():
            wandb.summary[f"{self.label}/{k}"] = v
        wandb.log({f"{self.label}/{k}": v for k, v in self.results.items()})
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
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--wandb-project", default="qwen3-8b-tool-use",
                        help="W&B project name (requires WANDB_API_KEY env var)")
    args = parser.parse_args()

    examples = load_eval_examples(max_samples=args.max_samples)
    if not examples:
        raise RuntimeError("No evaluation examples available.")
    logger.info("Using %d held-out evaluation examples", len(examples))

    wandb.init(
        project=args.wandb_project,
        name=f"eval-{args.mode}-{args.base_model.split('/')[-1]}",
        tags=["evaluation", args.mode, "tool-use"],
        config={
            "base_model": args.base_model,
            "mode": args.mode,
            "max_samples": args.max_samples,
            "sft_adapter": args.sft_adapter,
            "grpo_adapter": args.grpo_adapter,
            "n_eval_examples": len(examples),
        },
        mode="disabled" if not os.getenv("WANDB_API_KEY") else "online",
    )

    all_results = {}

    # --- baseline ---
    if args.mode in ("baseline", "compare", "all"):
        logger.info(">>> Baseline evaluation")
        model, tok = _load_base_model(args.base_model)
        ev = ToolUseEvaluator(model, tok, label="baseline")
        all_results["baseline"] = ev.evaluate_all(examples)
        del model, tok
        torch.cuda.empty_cache()

    # --- SFT ---
    if args.mode in ("sft", "compare", "all"):
        logger.info(">>> SFT evaluation")
        model, tok = _load_adapter_model(args.base_model, args.sft_adapter)
        ev = ToolUseEvaluator(model, tok, label="sft")
        all_results["sft"] = ev.evaluate_all(examples)
        del model, tok
        torch.cuda.empty_cache()

    # --- GRPO ---
    if args.mode in ("grpo", "compare", "all"):
        logger.info(">>> GRPO evaluation")
        # GRPO adapter sits on top of SFT-merged weights.
        # Load base -> merge SFT -> load GRPO adapter.
        model, tok = _load_base_model(args.base_model)
        if PeftModel is None:
            raise RuntimeError("peft is required for adapter evaluation (sft/grpo modes).")
        if os.path.isdir(args.sft_adapter):
            model = PeftModel.from_pretrained(model, args.sft_adapter)
            model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, args.grpo_adapter)
        ev = ToolUseEvaluator(model, tok, label="grpo")
        all_results["grpo"] = ev.evaluate_all(examples)
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

    # --- W&B comparison table and artifact ---
    if len(all_results) > 1:
        metrics = list(next(iter(all_results.values())).keys())
        stages = list(all_results.keys())
        comparison_table = wandb.Table(columns=["metric"] + stages)
        for metric in metrics:
            row = [metric] + [
                all_results[stage].get(metric, float("nan")) for stage in stages
            ]
            comparison_table.add_data(*row)
        wandb.log({"eval/comparison_table": comparison_table})
    wandb.save(args.output)
    wandb.finish()


if __name__ == "__main__":
    main()
