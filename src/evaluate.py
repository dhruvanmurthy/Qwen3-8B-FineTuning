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
from tinker import ServiceClient
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer, get_text_content
from constants import TOOL_USE_SYSTEM_PROMPT
from rewards import extract_tool_call, extract_tool_calls

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
    tools_context: Optional[str] = None


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


def _extract_tools_context(row: Dict[str, Any]) -> Optional[str]:
    """Build a tools context string for evaluation prompts.

    Training includes full tool definitions (name, description, parameters) in
    the system prompt; evaluation must provide the same level of detail to
    avoid a prompt-format mismatch.
    """
    tools = _json_load_maybe(row.get("tools"))
    if isinstance(tools, list) and tools:
        defs = [t for t in tools[:20] if isinstance(t, dict) and t.get("name")]
        if defs:
            return json.dumps(defs, indent=2, ensure_ascii=False)

    api_list = _json_load_maybe(row.get("api_list"))
    if isinstance(api_list, list) and api_list:
        defs = [t for t in api_list[:20] if isinstance(t, dict) and (t.get("name") or t.get("tool_name") or t.get("api_name"))]
        if defs:
            return json.dumps(defs, indent=2, ensure_ascii=False)

    functions = _json_load_maybe(row.get("function"))
    if isinstance(functions, list) and functions:
        defs = [fn for fn in functions[:20] if isinstance(fn, dict) and fn.get("name")]
        if defs:
            return json.dumps(defs, indent=2, ensure_ascii=False)

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

    return EvalExample(
        prompt=prompt,
        expected_tool=expected_tool,
        expected_args=expected_args,
        expected_chain=[c for c in expected_chain if c],
        source=source,
        tools_context=_extract_tools_context(row),
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
    """Load held-out evaluation examples from prepared dataset or synthetic data.

    Priority:
      1) `data/processed/test_raw.jsonl` (raw test split from prepare_datasets.sh — all sources)
      2) local synthetic data held-out split (10%)
      3) HF datasets held-out split (10%)
    """

    examples: List[EvalExample] = []

    # 1) Preferred: raw test JSONL from prepare_datasets.sh (preserves source + tool metadata)
    test_raw = Path("data/processed/test_raw.jsonl")
    if test_raw.exists():
        rows = _load_jsonl_rows(test_raw)
        for row in rows:
            ex = _normalize_row(row, source=str(row.get("source", "prepared")))
            if ex:
                examples.append(ex)
        if examples:
            logger.info("Loaded %d prepared test examples from %s", len(examples), test_raw)

    # 2) Local synthetic fallback (if prepared dataset unavailable)
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


def _save_checkpoint(checkpoint_file: str, data: Dict[str, Any]) -> None:
    """Atomically write checkpoint data to JSON so partial results survive a crash."""
    path = Path(checkpoint_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)
    logger.debug("Checkpoint saved to %s", checkpoint_file)


# -----------------------------------------------------------------------
# Core evaluator  (async — all inference on Tinker)
# -----------------------------------------------------------------------

class ToolUseEvaluator:
    """Evaluate a single model on tool-use benchmarks via Tinker inference."""

    def __init__(self, sampling_client, renderer, label: str = "model",
                 checkpoint_file: Optional[str] = None,
                 checkpoint_data: Optional[Dict[str, Any]] = None):
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.label = label
        self.results: Dict[str, float] = {}
        self.checkpoint_file = checkpoint_file
        self._all_checkpoint_data: Dict[str, Any] = checkpoint_data if checkpoint_data is not None else {}
        # Pre-populate results from checkpoint if this stage was partially completed
        if label in self._all_checkpoint_data:
            self.results = dict(self._all_checkpoint_data[label])
            logger.info("[%s] Resumed %d benchmark result(s) from checkpoint", label, len(self.results))

    SYSTEM_PROMPT = TOOL_USE_SYSTEM_PROMPT

    # Maps benchmark name -> primary result key used to detect completion in checkpoints
    _BENCHMARK_RESULT_KEYS: Dict[str, str] = {
        "tool_selection": "tool_selection_accuracy",
        "argument_accuracy": "argument_accuracy",
        "schema_compliance": "schema_compliance",
        "multi_step": "multi_step_success",
        "latency": "avg_latency_ms",
    }

    # --- generation ---

    @staticmethod
    def _build_system_prompt(example: EvalExample) -> str:
        if example.tools_context:
            return f"{ToolUseEvaluator.SYSTEM_PROMPT}\n\nAvailable tools:\n{example.tools_context}"
        return ToolUseEvaluator.SYSTEM_PROMPT

    async def generate(self, prompt: str, max_new_tokens: int = 512,
                       system_prompt: Optional[str] = None) -> tuple:
        """Return (text, n_output_tokens) via Tinker sample_async."""
        sys_prompt = system_prompt or self.SYSTEM_PROMPT
        convo = [
            {"role": "system", "content": sys_prompt},
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
        # Decode raw tokens directly so we get the full output including
        # <think>...</think> blocks. get_text_content() strips thinking content
        # and returns empty for Qwen3 responses where tool calls follow thinking.
        text = self.renderer.tokenizer.decode(seq.tokens, skip_special_tokens=True)
        if not text:
            # Final fallback: parse_response path
            parsed_msg, _ = self.renderer.parse_response(seq.tokens)
            text = get_text_content(parsed_msg) or ""
        return text, len(seq.tokens)

    # --- tool extraction ---

    @staticmethod
    def extract_tool_name(text: str) -> Optional[str]:
        """Extract tool name using the same parser as GRPO rewards.

        Uses rewards.extract_tool_call for consistency with training-time
        reward grading.  Avoids false positives from loose regex patterns.
        """
        call = extract_tool_call(text)
        if call and isinstance(call.get("name"), str):
            return call["name"]
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

            output, _ = await self.generate(
                example.prompt,
                max_new_tokens=4096,
                system_prompt=self._build_system_prompt(example),
            )
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

            output, _ = await self.generate(
                example.prompt,
                max_new_tokens=1024,
                system_prompt=self._build_system_prompt(example),
            )

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
            output, _ = await self.generate(
                example.prompt,
                max_new_tokens=1024,
                system_prompt=self._build_system_prompt(example),
            )

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
        logger.info("  [%s] Found %d multi-step examples (out of %d total)",
                    self.label, len(multi_step), len(examples))

        if not multi_step:
            logger.warning("  [%s] NO multi-step examples in eval set — check data/processed or synthetic JSONL.", self.label)
            self.results["multi_step_success"] = 0.0
            return 0.0

        correct = total = 0
        # Diagnostic counters
        count_zero_calls = 0
        count_one_call = 0
        count_partial = 0
        count_wrong_order = 0
        sample_rows = []

        for i, example in enumerate(multi_step):
            output, _ = await self.generate(
                example.prompt,
                max_new_tokens=2048,  # more room for multi-call outputs
                system_prompt=self._build_system_prompt(example),
            )
            predicted_calls = extract_tool_calls(output)
            pred_names = [c.get("name", "").lower().strip() for c in predicted_calls]
            exp_names = [t.lower().strip() for t in example.expected_chain]
            match = pred_names == exp_names

            found_count = sum(1 for t in exp_names if t in pred_names)
            n_pred = len(pred_names)
            n_exp = len(exp_names)

            # Diagnose failure mode
            if n_pred == 0:
                failure_mode = "zero_calls"
                count_zero_calls += 1
            elif n_pred == 1 and n_exp > 1:
                failure_mode = "only_one_call"
                count_one_call += 1
            elif found_count == n_exp and not match:
                failure_mode = "wrong_order"
                count_wrong_order += 1
            elif found_count > 0 and found_count < n_exp:
                failure_mode = "partial"
                count_partial += 1
            elif match:
                failure_mode = "correct"
            else:
                failure_mode = f"wrong_tools(pred={pred_names})"

            if match:
                correct += 1
            total += 1

            # Always log first 5 in detail; log all mismatches up to 50
            log_this = (i < 5) or (not match and len(sample_rows) < 50)
            if log_this:
                status = "✓" if match else "✗"
                logger.info(
                    "  [%s] %s  chain=%s  predicted=%s  found=%d/%d  mode=%s",
                    self.label, status, exp_names, pred_names,
                    found_count, n_exp, failure_mode,
                )
                sample_rows.append({
                    "prompt": example.prompt[:300],
                    "expected_chain": str(exp_names),
                    "predicted_calls": str(pred_names),
                    "found": f"{found_count}/{n_exp}",
                    "failure_mode": failure_mode,
                    "match": match,
                    "n_predicted": n_pred,
                    "model_output": output[:800],
                })

        # Summary diagnostics
        logger.info(
            "  [%s] Multi-step failure breakdown — zero_calls=%d  one_call=%d  "
            "partial=%d  wrong_order=%d  correct=%d  total=%d",
            self.label, count_zero_calls, count_one_call,
            count_partial, count_wrong_order, correct, total,
        )

        if sample_rows:
            table = wandb.Table(
                columns=["prompt", "expected_chain", "predicted_calls",
                         "found", "failure_mode", "match", "n_predicted", "model_output"],
                data=[[r["prompt"], r["expected_chain"], r["predicted_calls"],
                       r["found"], r["failure_mode"], r["match"],
                       r["n_predicted"], r["model_output"]] for r in sample_rows],
            )
            wandb.log({f"{self.label}/multi_step_samples": table})
            # Also log failure mode breakdown as scalars
            wandb.log({
                f"{self.label}/multi_step_zero_calls": count_zero_calls / total if total else 0,
                f"{self.label}/multi_step_one_call_only": count_one_call / total if total else 0,
                f"{self.label}/multi_step_partial": count_partial / total if total else 0,
                f"{self.label}/multi_step_wrong_order": count_wrong_order / total if total else 0,
            })

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
            output, n_tokens = await self.generate(
                example.prompt,
                max_new_tokens=1024,
                system_prompt=self._build_system_prompt(example),
            )
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

    async def evaluate_all(
        self,
        examples: List[EvalExample],
        benchmarks: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Run selected benchmarks. Pass benchmarks=[...] to run a subset."""
        _map = {
            "tool_selection": lambda: self.evaluate_tool_selection(examples),
            "argument_accuracy": lambda: self.evaluate_argument_accuracy(examples),
            "schema_compliance": lambda: self.evaluate_schema_compliance(examples),
            "multi_step": lambda: self.evaluate_multi_step(examples),
            "latency": lambda: self.evaluate_latency(examples, num_samples=100),
        }
        _all = list(_map.keys())
        to_run = benchmarks or _all
        for name in to_run:
            if name not in _map:
                raise ValueError(f"Unknown benchmark '{name}'. Choose from: {_all}")
            # Skip if already completed in a previous run (checkpoint resume)
            primary_key = self._BENCHMARK_RESULT_KEYS.get(name)
            if primary_key and primary_key in self.results:
                logger.info("[%s] Skipping '%s' — already in checkpoint", self.label, name)
                continue
            await _map[name]()
            # Save after each benchmark so a crash loses at most one benchmark's work
            if self.checkpoint_file:
                self._all_checkpoint_data[self.label] = dict(self.results)
                _save_checkpoint(self.checkpoint_file, self._all_checkpoint_data)
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
        choices=["baseline", "sft", "grpo", "compare", "all", "fetch-compare"],
        default="all",
        help="Which stage(s) to evaluate. Use 'fetch-compare' to build the "
             "comparison table from already-completed W&B runs without any inference.",
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
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        metavar="BENCHMARK",
        help="Subset of benchmarks to run: tool_selection argument_accuracy "
             "schema_compliance multi_step latency. Default: all.",
    )
    parser.add_argument(
        "--checkpoint-file",
        default="outputs/eval_checkpoint.json",
        help="Path to checkpoint file for saving/resuming partial results. "
             "Delete this file to start a fresh run.",
    )
    # fetch-compare: pull results from existing completed W&B runs (no inference)
    parser.add_argument(
        "--baseline-run-path",
        default=None,
        metavar="ENTITY/PROJECT/RUN_ID",
        help="W&B run path for a completed baseline eval run (fetch-compare mode).",
    )
    parser.add_argument(
        "--sft-run-path",
        default=None,
        metavar="ENTITY/PROJECT/RUN_ID",
        help="W&B run path for a completed SFT eval run (fetch-compare mode).",
    )
    parser.add_argument(
        "--grpo-run-path",
        default=None,
        metavar="ENTITY/PROJECT/RUN_ID",
        help="W&B run path for a completed GRPO eval run (fetch-compare mode).",
    )
    args = parser.parse_args()

    if args.mode == "fetch-compare":
        _fetch_compare(args)
    else:
        asyncio.run(_run_eval(args))


def _fetch_compare(args) -> None:
    """Build a comparison table from already-completed W&B eval runs.

    Pulls summary metrics from existing runs via the W&B API — no Tinker
    inference is performed.  Useful when individual stage eval runs completed
    successfully but the combined 'compare' run crashed.

    Usage:
        python src/evaluate.py --mode fetch-compare \\
            --baseline-run-path  ENTITY/PROJECT/RUN_ID \\
            --sft-run-path       ENTITY/PROJECT/RUN_ID \\
            --grpo-run-path      ENTITY/PROJECT/RUN_ID \\
            --output outputs/eval_comparison.json
    """
    api = wandb.Api()

    stage_run_paths: Dict[str, str] = {}
    if args.baseline_run_path:
        stage_run_paths["baseline"] = args.baseline_run_path
    if args.sft_run_path:
        stage_run_paths["sft"] = args.sft_run_path
    if args.grpo_run_path:
        stage_run_paths["grpo"] = args.grpo_run_path

    if not stage_run_paths:
        raise RuntimeError(
            "fetch-compare requires at least one of: "
            "--baseline-run-path, --sft-run-path, --grpo-run-path"
        )

    all_results: Dict[str, Dict[str, float]] = {}
    for stage, run_path in stage_run_paths.items():
        logger.info("Fetching W&B summary for stage '%s' from run: %s", stage, run_path)
        run = api.run(run_path)
        summary = dict(run.summary)
        # Metrics are logged as "{stage}/metric_name" — strip the prefix
        prefix = f"{stage}/"
        metrics = {
            k[len(prefix):]: float(v)
            for k, v in summary.items()
            if k.startswith(prefix) and isinstance(v, (int, float))
        }
        if not metrics:
            raise RuntimeError(
                f"No metrics found for stage '{stage}' in run {run_path}. "
                f"Available summary keys: {[k for k in summary if not k.startswith('_')]}"
            )
        all_results[stage] = metrics
        logger.info("  Got %d metrics: %s", len(metrics), list(metrics.keys()))

    compare_stages(all_results)

    # Save combined results locally
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Combined results saved to %s", args.output)

    # Log a new W&B run with the comparison table so it appears in the project
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "qwen3-8b-tool-use"),
        entity=os.getenv("WANDB_ENTITY") or None,
        name=f"eval-compare-{args.base_model.split('/')[-1]}",
        tags=["evaluation", "compare", "tool-use", "fetch-compare"],
        config={
            "base_model": args.base_model,
            "mode": "fetch-compare",
            "source_runs": stage_run_paths,
        },
        mode="disabled" if not os.getenv("WANDB_API_KEY") else "online",
    )
    metrics_list = list(next(iter(all_results.values())).keys())
    stages = list(all_results.keys())
    comparison_table = wandb.Table(columns=["metric"] + stages)
    for metric in metrics_list:
        row = [metric] + [all_results[stage].get(metric, float("nan")) for stage in stages]
        comparison_table.add_data(*row)
    wandb.log({"eval/comparison_table": comparison_table})
    # Also log flat summary metrics for easy charting
    for stage, results in all_results.items():
        wandb.log({f"{stage}/{k}": v for k, v in results.items()})
    wandb.save(args.output)
    wandb.finish()
    logger.info("Comparison run logged to W&B.")


async def _run_eval(args) -> None:
    """Async evaluation loop — all inference on Tinker."""

    # ---- Checkpoint: load any existing partial results ----
    checkpoint_file: str = args.checkpoint_file
    checkpoint_data: Dict[str, Any] = {}
    ckpt_path = Path(checkpoint_file)
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            logger.info("Loaded checkpoint from %s — stages present: %s",
                        checkpoint_file, list(checkpoint_data.keys()))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load checkpoint %s (%s) — starting fresh", checkpoint_file, exc)

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
        return ToolUseEvaluator(sc, renderer, label=label,
                                checkpoint_file=checkpoint_file,
                                checkpoint_data=checkpoint_data)

    # Determine which result keys are required for the benchmarks being run
    _benchmarks_to_run = args.benchmarks or list(ToolUseEvaluator._BENCHMARK_RESULT_KEYS.keys())
    _required_keys = [
        ToolUseEvaluator._BENCHMARK_RESULT_KEYS[b]
        for b in _benchmarks_to_run
        if b in ToolUseEvaluator._BENCHMARK_RESULT_KEYS
    ]

    def _stage_complete(stage: str) -> bool:
        """True if all requested benchmarks already have results in the checkpoint."""
        return stage in checkpoint_data and all(
            k in checkpoint_data[stage] for k in _required_keys
        )

    all_results: Dict[str, Dict[str, float]] = {}

    # --- baseline ---
    if args.mode in ("baseline", "compare", "all"):
        if _stage_complete("baseline"):
            logger.info(">>> Baseline: all benchmarks complete in checkpoint — skipping")
            all_results["baseline"] = checkpoint_data["baseline"]
        else:
            logger.info(">>> Baseline evaluation")
            ev = await _make_evaluator("baseline")
            all_results["baseline"] = await ev.evaluate_all(examples, benchmarks=args.benchmarks)

    # --- SFT ---
    if args.mode in ("sft", "compare", "all"):
        if _stage_complete("sft"):
            logger.info(">>> SFT: all benchmarks complete in checkpoint — skipping")
            all_results["sft"] = checkpoint_data["sft"]
        else:
            logger.info(">>> SFT evaluation  (sampler: %s)", sft_sampler)
            ev = await _make_evaluator("sft", model_path=sft_sampler)
            all_results["sft"] = await ev.evaluate_all(examples, benchmarks=args.benchmarks)

    # --- GRPO ---
    if args.mode in ("grpo", "compare", "all"):
        if _stage_complete("grpo"):
            logger.info(">>> GRPO: all benchmarks complete in checkpoint — skipping")
            all_results["grpo"] = checkpoint_data["grpo"]
        else:
            logger.info(">>> GRPO evaluation  (sampler: %s)", grpo_sampler)
            ev = await _make_evaluator("grpo", model_path=grpo_sampler)
            all_results["grpo"] = await ev.evaluate_all(examples, benchmarks=args.benchmarks)

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
