"""Three-stage evaluation for Qwen3-8B tool-use fine-tuning.

Compares: Baseline (zero-shot) -> SFT -> GRPO
Metrics: tool selection accuracy, argument accuracy, schema compliance,
multi-step success, and latency.

All inference runs on Tinker (remote GPU) — no local model loading required.
Sampling clients are created from:
  - base model:   create_sampling_client_async(base_model="Qwen/Qwen3-8B")
  - SFT/GRPO:     create_sampling_client_async(model_path=<tinker:// URI>)
"""

import asyncio
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
import tinker
import wandb
from datasets import load_dataset
from tinker import ServiceClient
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer, get_text_content
from rewards import extract_tool_call

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
# Checkpoint helper
# -----------------------------------------------------------------------

def _read_last_sampler_path(log_dir: str) -> Optional[str]:
    """Return the sampler_path from the last valid checkpoint in checkpoints.jsonl."""
    ckpt_file = Path(log_dir) / "checkpoints.jsonl"
    if not ckpt_file.exists():
        return None
    last: Optional[str] = None
    with open(ckpt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("sampler_path"):
                    last = entry["sampler_path"]
            except json.JSONDecodeError:
                continue
    return last


# -----------------------------------------------------------------------
# Core evaluator  (async — all inference on Tinker)
# -----------------------------------------------------------------------

class ToolUseEvaluator:
    """Evaluate a single model on tool-use benchmarks via Tinker inference."""

    def __init__(self, sampling_client, renderer, label: str = "model"):
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.label = label
        self.results: Dict[str, float] = {}

    SYSTEM_PROMPT = (
        "You are a helpful assistant that can use tools. "
        "When you need to use a tool, respond with a JSON tool call "
        "inside <tool_call> tags, like:\n"
        "<tool_call>\n"
        '{"name": "tool_name", "arguments": {"arg": "value"}}\n'
        "</tool_call>"
    )

    # --- generation ---

    async def generate(self, prompt: str, max_new_tokens: int = 512) -> tuple:
        """Return (text, n_output_tokens) via Tinker sample_async."""
        convo = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        model_input = self.renderer.build_generation_prompt(convo)
        result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=max_new_tokens,
                stop=self.renderer.get_stop_sequences(),
            ),
        )
        seq = result.sequences[0]
        parsed_msg, _ = self.renderer.parse_response(seq.tokens)
        text = get_text_content(parsed_msg)
        return text, len(seq.tokens)

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

    async def evaluate_tool_selection(self, examples: List[EvalExample]) -> float:
        logger.info("[%s] Evaluating tool selection…", self.label)

        correct = total = 0
        sample_rows = []  # For W&B table logging
        for i, example in enumerate(examples):
            if i > 0 and i % 50 == 0:
                logger.info("  progress: %d/%d (acc so far: %.1f%%)",
                            i, len(examples), 100 * correct / total if total else 0)

            if not example.expected_tool:
                continue

            output, _ = await self.generate(example.prompt, max_new_tokens=256)
            predicted = self.extract_tool_name(output)
            match = bool(predicted and predicted.lower() == example.expected_tool.lower())

            if match:
                correct += 1
            total += 1

            # Log first 20 examples and periodic mismatches for visibility
            if total <= 20 or (not match and len(sample_rows) < 100):
                status = "✓" if match else "✗"
                logger.info(
                    "  [%s] %s  expected=%s  predicted=%s  |  prompt=%.120s…",
                    self.label, status, example.expected_tool, predicted,
                    example.prompt.replace("\n", " "),
                )
                sample_rows.append({
                    "prompt": example.prompt[:300],
                    "expected_tool": example.expected_tool,
                    "predicted_tool": predicted or "(none)",
                    "match": match,
                    "model_output": output[:500],
                })

        # Log sample table to W&B
        if sample_rows:
            table = wandb.Table(
                columns=["prompt", "expected_tool", "predicted_tool", "match", "model_output"],
                data=[[r["prompt"], r["expected_tool"], r["predicted_tool"], r["match"], r["model_output"]] for r in sample_rows],
            )
            wandb.log({f"{self.label}/tool_selection_samples": table})

        accuracy = correct / total if total > 0 else 0
        self.results["tool_selection_accuracy"] = accuracy
        logger.info("[%s] Tool Selection: %.1f%% (%d/%d)", self.label, 100 * accuracy, correct, total)
        return accuracy

    async def evaluate_argument_accuracy(self, examples: List[EvalExample]) -> float:
        logger.info("[%s] Evaluating argument accuracy…", self.label)

        correct = total = 0
        sample_rows = []
        for i, example in enumerate(examples):
            if not example.expected_args:
                continue

            output, _ = await self.generate(example.prompt, max_new_tokens=256)

            call = extract_tool_call(output)
            pred_args = None
            match = False
            if call:
                pred_args = call.get("arguments", call.get("parameters", {}))
                if pred_args == example.expected_args:
                    correct += 1
                    match = True
            total += 1

            if total <= 10 or (not match and len(sample_rows) < 50):
                status = "✓" if match else "✗"
                logger.info(
                    "  [%s] %s  expected_args=%s  predicted_args=%s",
                    self.label, status,
                    json.dumps(example.expected_args)[:120],
                    json.dumps(pred_args)[:120] if pred_args else "(none)",
                )
                sample_rows.append({
                    "prompt": example.prompt[:300],
                    "expected_args": json.dumps(example.expected_args)[:300],
                    "predicted_args": json.dumps(pred_args)[:300] if pred_args else "(none)",
                    "match": match,
                    "model_output": output[:500],
                })

        if sample_rows:
            table = wandb.Table(
                columns=["prompt", "expected_args", "predicted_args", "match", "model_output"],
                data=[[r["prompt"], r["expected_args"], r["predicted_args"], r["match"], r["model_output"]] for r in sample_rows],
            )
            wandb.log({f"{self.label}/argument_accuracy_samples": table})

        accuracy = correct / total if total > 0 else 0
        self.results["argument_accuracy"] = accuracy
        logger.info("[%s] Argument Accuracy: %.1f%%", self.label, 100 * accuracy)
        return accuracy

    async def evaluate_schema_compliance(self, examples: List[EvalExample]) -> float:
        logger.info("[%s] Evaluating schema compliance…", self.label)

        valid = total = 0
        invalid_samples = []
        for example in examples:
            output, _ = await self.generate(example.prompt, max_new_tokens=256)

            call = extract_tool_call(output)
            is_valid = bool(
                call
                and isinstance(call.get("name"), str)
                and isinstance(call.get("arguments", call.get("parameters", {})), dict)
            )
            if is_valid:
                valid += 1
            elif len(invalid_samples) < 20:
                logger.info(
                    "  [%s] ✗ invalid schema  |  output=%.200s…",
                    self.label, output.replace("\n", " "),
                )
                invalid_samples.append({
                    "prompt": example.prompt[:300],
                    "model_output": output[:500],
                })
            total += 1

        if invalid_samples:
            table = wandb.Table(
                columns=["prompt", "model_output"],
                data=[[r["prompt"], r["model_output"]] for r in invalid_samples],
            )
            wandb.log({f"{self.label}/schema_invalid_samples": table})

        rate = valid / total if total > 0 else 0
        self.results["schema_compliance"] = rate
        logger.info("[%s] Schema Compliance: %.1f%% (%d/%d)", self.label, 100 * rate, valid, total)
        return rate

    async def evaluate_multi_step(self, examples: List[EvalExample]) -> float:
        logger.info("[%s] Evaluating multi-step chains…", self.label)
        multi_step = [e for e in examples if len(e.expected_chain) > 1]
        logger.info("  [%s] Found %d multi-step examples", self.label, len(multi_step))

        correct = total = 0
        sample_rows = []
        for example in multi_step:
            output, _ = await self.generate(example.prompt, max_new_tokens=512)
            found = sum(1 for t in example.expected_chain if t.lower() in output.lower())
            match = found == len(example.expected_chain)
            if match:
                correct += 1
            total += 1

            if total <= 10 or (not match and len(sample_rows) < 30):
                status = "✓" if match else "✗"
                logger.info(
                    "  [%s] %s  chain=%s  found=%d/%d  |  output=%.150s…",
                    self.label, status, example.expected_chain,
                    found, len(example.expected_chain),
                    output.replace("\n", " "),
                )
                sample_rows.append({
                    "prompt": example.prompt[:300],
                    "expected_chain": str(example.expected_chain),
                    "found": f"{found}/{len(example.expected_chain)}",
                    "match": match,
                    "model_output": output[:500],
                })

        if sample_rows:
            table = wandb.Table(
                columns=["prompt", "expected_chain", "found", "match", "model_output"],
                data=[[r["prompt"], r["expected_chain"], r["found"], r["match"], r["model_output"]] for r in sample_rows],
            )
            wandb.log({f"{self.label}/multi_step_samples": table})

        rate = correct / total if total > 0 else 0
        self.results["multi_step_success"] = rate
        logger.info("[%s] Multi-Step Success: %.1f%% (%d/%d)", self.label, 100 * rate, correct, total)
        return rate

    async def evaluate_latency(self, examples: List[EvalExample], num_samples: int = 100) -> float:
        logger.info("[%s] Evaluating latency…", self.label)
        subset = examples[:num_samples]

        times, total_tokens = [], 0
        for example in subset:
            start = time.time()
            output, n_tokens = await self.generate(example.prompt, max_new_tokens=256)
            elapsed = time.time() - start
            times.append(elapsed)
            total_tokens += n_tokens

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

    async def evaluate_all(self, examples: List[EvalExample]) -> Dict[str, float]:
        await self.evaluate_tool_selection(examples)
        await self.evaluate_argument_accuracy(examples)
        await self.evaluate_schema_compliance(examples)
        await self.evaluate_multi_step(examples)
        await self.evaluate_latency(examples, num_samples=100)
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

    parser = argparse.ArgumentParser(description="Three-stage evaluation (Tinker inference)")
    parser.add_argument(
        "--mode",
        choices=["baseline", "sft", "grpo", "compare", "all"],
        default="all",
        help="Which stage(s) to evaluate",
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B",
                        help="HF model ID used for baseline Tinker inference")
    parser.add_argument(
        "--sft-sampler-path",
        default=None,
        help="tinker:// URI for SFT sampler weights. "
             "Falls back to reading the last sampler_path in outputs/sft/checkpoints.jsonl",
    )
    parser.add_argument(
        "--grpo-sampler-path",
        default=None,
        help="tinker:// URI for GRPO sampler weights. "
             "Falls back to reading the last sampler_path in outputs/grpo/checkpoints.jsonl",
    )
    parser.add_argument("--sft-output-dir", default="./outputs/sft",
                        help="Local dir containing SFT checkpoints.jsonl (auto-detect fallback)")
    parser.add_argument("--grpo-output-dir", default="./outputs/grpo",
                        help="Local dir containing GRPO checkpoints.jsonl (auto-detect fallback)")
    parser.add_argument("--output", default="outputs/evaluation_results.json")
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()

    asyncio.run(_run_eval(args))


async def _run_eval(args) -> None:
    """Async evaluation loop — all inference on Tinker."""

    # ---- Tinker service client ----
    if not os.getenv("TINKER_API_KEY"):
        raise RuntimeError(
            "TINKER_API_KEY environment variable is required for Tinker-based evaluation."
        )
    service_client = ServiceClient()

    # ---- Resolve sampler paths ----
    sft_sampler = args.sft_sampler_path or _read_last_sampler_path(args.sft_output_dir)
    grpo_sampler = args.grpo_sampler_path or _read_last_sampler_path(args.grpo_output_dir)

    if args.mode in ("sft", "compare", "all") and not sft_sampler:
        raise RuntimeError(
            f"SFT sampler path not provided and not found in {args.sft_output_dir}/checkpoints.jsonl. "
            "Run SFT training first, or pass --sft-sampler-path explicitly."
        )
    if args.mode in ("grpo", "compare", "all") and not grpo_sampler:
        raise RuntimeError(
            f"GRPO sampler path not provided and not found in {args.grpo_output_dir}/checkpoints.jsonl. "
            "Run GRPO training first, or pass --grpo-sampler-path explicitly."
        )

    # ---- W&B init ----
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "qwen3-8b-tool-use"),
        entity=os.getenv("WANDB_ENTITY") or None,
        name=f"eval-{args.mode}-{args.base_model.split('/')[-1]}",
        tags=["evaluation", args.mode, "tool-use"],
        config={
            "base_model": args.base_model,
            "mode": args.mode,
            "max_samples": args.max_samples,
            "sft_sampler_path": sft_sampler,
            "grpo_sampler_path": grpo_sampler,
        },
        mode="disabled" if not os.getenv("WANDB_API_KEY") else "online",
    )

    examples = load_eval_examples(max_samples=args.max_samples)
    if not examples:
        raise RuntimeError("No evaluation examples available.")
    logger.info("Using %d held-out evaluation examples", len(examples))
    wandb.config.update({"n_eval_examples": len(examples)}, allow_val_change=True)

    # ---- Helper: build (sampling_client, renderer) for a stage ----
    async def _make_evaluator(label: str, model_path: Optional[str] = None) -> ToolUseEvaluator:
        logger.info("Creating Tinker sampling client for stage: %s", label)
        if model_path:
            sc = await service_client.create_sampling_client_async(model_path=model_path)
        else:
            sc = await service_client.create_sampling_client_async(base_model=args.base_model)
        tokenizer = sc.get_tokenizer()
        renderer_name = get_recommended_renderer_name(args.base_model)
        renderer = get_renderer(renderer_name, tokenizer)
        return ToolUseEvaluator(sc, renderer, label=label)

    all_results: Dict[str, Dict[str, float]] = {}

    # --- baseline ---
    if args.mode in ("baseline", "compare", "all"):
        logger.info(">>> Baseline evaluation")
        ev = await _make_evaluator("baseline")
        all_results["baseline"] = await ev.evaluate_all(examples)

    # --- SFT ---
    if args.mode in ("sft", "compare", "all"):
        logger.info(">>> SFT evaluation  (sampler: %s)", sft_sampler)
        ev = await _make_evaluator("sft", model_path=sft_sampler)
        all_results["sft"] = await ev.evaluate_all(examples)

    # --- GRPO ---
    if args.mode in ("grpo", "compare", "all"):
        logger.info(">>> GRPO evaluation  (sampler: %s)", grpo_sampler)
        ev = await _make_evaluator("grpo", model_path=grpo_sampler)
        all_results["grpo"] = await ev.evaluate_all(examples)

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
