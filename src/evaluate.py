"""Three-stage evaluation for tool-use fine-tuning via Tinker."""

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


@dataclass
class EvalExample:
    prompt: str
    expected_tool: Optional[str]
    expected_args: Optional[Dict[str, Any]]
    expected_chain: List[str]
    source: str
    tools_context: Optional[str] = None


@dataclass
class GeneratedExample:
    output: str
    n_tokens: int
    elapsed_s: float
    call: Optional[Dict[str, Any]]
    calls: List[Dict[str, Any]]


def _stable_test_split(rows: List[Dict[str, Any]], ratio: float = 0.1) -> List[Dict[str, Any]]:
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


def _extract_tools_context(row: Dict[str, Any]) -> Optional[str]:
    for key in ("tools", "api_list", "function"):
        value = _json_load_maybe(row.get(key))
        if isinstance(value, list) and value:
            defs = [item for item in value[:20] if isinstance(item, dict) and item.get("name")]
            if defs:
                return json.dumps(defs, indent=2, ensure_ascii=False)
    return None


def _normalize_row(row: Dict[str, Any], source: str) -> Optional[EvalExample]:
    prompt = row.get("instruction") or row.get("user_instruction") or row.get("query") or row.get("question") or row.get("text")
    if not isinstance(prompt, str) or not prompt.strip():
        return None

    expected_tool: Optional[str] = None
    expected_args: Optional[Dict[str, Any]] = None
    expected_chain: List[str] = []

    for key in ("tool_calls", "expected_calls", "api_calls"):
        calls = _json_load_maybe(row.get(key))
        if not isinstance(calls, list) or not calls:
            continue
        first = calls[0] if isinstance(calls[0], dict) else {}
        expected_tool = expected_tool or first.get("name") or first.get("tool")
        raw_args = first.get("arguments", first.get("parameters", {}))
        if expected_args is None and isinstance(raw_args, dict):
            expected_args = raw_args
        if not expected_chain:
            expected_chain = [
                (c.get("name") or c.get("tool") or "")
                for c in calls
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
            if line:
                rows.append(json.loads(line))
    return rows


def load_eval_examples(max_samples: int = 1000) -> List[EvalExample]:
    examples: List[EvalExample] = []
    test_raw = Path("data/processed/test_raw.jsonl")
    if test_raw.exists():
        for row in _load_jsonl_rows(test_raw):
            ex = _normalize_row(row, source=str(row.get("source", "prepared")))
            if ex:
                examples.append(ex)
        if examples:
            logger.info("Loaded %d prepared test examples from %s", len(examples), test_raw)

    if not examples:
        synth_dir = Path("data/raw/synthetic")
        if synth_dir.exists():
            synth_rows: List[Dict[str, Any]] = []
            for p in list(synth_dir.glob("*.jsonl")) + list(synth_dir.glob("*.json")):
                synth_rows.extend(_load_jsonl_rows(p))
            for row in _stable_test_split(synth_rows, ratio=0.1):
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


def _read_last_sampler_path(log_dir: str) -> Optional[str]:
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
    path = Path(checkpoint_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _load_stage_results_from_json(path: str, stage: str) -> Optional[Dict[str, float]]:
    json_path = Path(path)
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    results = payload.get(stage)
    if results is not None and not isinstance(results, dict):
        raise RuntimeError(f"{path} does not contain a '{stage}' results object.")
    return results


class ToolUseEvaluator:
    SYSTEM_PROMPT = TOOL_USE_SYSTEM_PROMPT
    _BENCHMARK_RESULT_KEYS: Dict[str, str] = {
        "tool_selection": "tool_selection_accuracy",
        "argument_accuracy": "argument_accuracy",
        "schema_compliance": "schema_compliance",
        "multi_step": "multi_step_success",
        "latency": "avg_latency_ms",
    }

    def __init__(self, sampling_client, renderer, label: str = "model", checkpoint_file: Optional[str] = None, checkpoint_data: Optional[Dict[str, Any]] = None):
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.label = label
        self.checkpoint_file = checkpoint_file
        self.results: Dict[str, float] = {}
        self._all_checkpoint_data = checkpoint_data if checkpoint_data is not None else {}
        if label in self._all_checkpoint_data:
            self.results = dict(self._all_checkpoint_data[label])

    @staticmethod
    def _build_system_prompt(example: EvalExample) -> str:
        if example.tools_context:
            return f"{ToolUseEvaluator.SYSTEM_PROMPT}\n\nAvailable tools:\n{example.tools_context}"
        return ToolUseEvaluator.SYSTEM_PROMPT

    async def generate(self, prompt: str, max_new_tokens: int = 512, system_prompt: Optional[str] = None) -> tuple[str, int]:
        convo = [
            {"role": "system", "content": system_prompt or self.SYSTEM_PROMPT},
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
        text = self.renderer.tokenizer.decode(seq.tokens, skip_special_tokens=True)
        if not text:
            parsed_msg, _ = self.renderer.parse_response(seq.tokens)
            text = get_text_content(parsed_msg) or ""
        return text, len(seq.tokens)

    @staticmethod
    def _max_tokens_for_benchmarks(benchmarks: List[str]) -> int:
        max_tokens = 1024
        if "multi_step" in benchmarks:
            max_tokens = max(max_tokens, 2048)
        if "tool_selection" in benchmarks:
            max_tokens = max(max_tokens, 4096)
        return max_tokens

    async def _generate_examples(self, examples: List[EvalExample], benchmarks: List[str]) -> List[GeneratedExample]:
        max_new_tokens = self._max_tokens_for_benchmarks(benchmarks)
        generated: List[GeneratedExample] = []
        logger.info("[%s] Generating %d cached response(s) once for %s", self.label, len(examples), benchmarks)
        for i, example in enumerate(examples):
            if i > 0 and i % 50 == 0:
                logger.info("  [%s] generation progress: %d/%d", self.label, i, len(examples))
            start = time.time()
            output, n_tokens = await self.generate(
                example.prompt,
                max_new_tokens=max_new_tokens,
                system_prompt=self._build_system_prompt(example),
            )
            generated.append(
                GeneratedExample(
                    output=output,
                    n_tokens=n_tokens,
                    elapsed_s=time.time() - start,
                    call=extract_tool_call(output),
                    calls=extract_tool_calls(output),
                )
            )
        return generated

    async def evaluate_all(self, examples: List[EvalExample], benchmarks: Optional[List[str]] = None) -> Dict[str, float]:
        to_run = benchmarks or list(self._BENCHMARK_RESULT_KEYS.keys())
        pending = [
            name for name in to_run
            if self._BENCHMARK_RESULT_KEYS.get(name) not in self.results
        ]
        if not pending:
            return self.results

        cache_examples = examples[:100] if set(pending) == {"latency"} else examples
        generated = await self._generate_examples(cache_examples, pending)

        if "tool_selection" in pending:
            total = 0
            correct = 0
            for example, gen in zip(examples, generated):
                if not example.expected_tool:
                    continue
                predicted = gen.call.get("name") if gen.call and isinstance(gen.call.get("name"), str) else None
                correct += int(bool(predicted and predicted.lower() == example.expected_tool.lower()))
                total += 1
            self.results["tool_selection_accuracy"] = correct / total if total else 0.0

        if "argument_accuracy" in pending:
            total = 0
            correct = 0
            for example, gen in zip(examples, generated):
                if example.expected_args is None:
                    continue
                pred_args = gen.call.get("arguments", gen.call.get("parameters", {})) if gen.call else None
                correct += int(pred_args == example.expected_args)
                total += 1
            self.results["argument_accuracy"] = correct / total if total else 0.0

        if "schema_compliance" in pending:
            valid = 0
            total = 0
            for gen in generated[:len(examples)]:
                call = gen.call
                valid += int(bool(call and isinstance(call.get("name"), str) and isinstance(call.get("arguments", call.get("parameters", {})), dict)))
                total += 1
            self.results["schema_compliance"] = valid / total if total else 0.0

        if "multi_step" in pending:
            pairs = [
                (example, gen)
                for example, gen in zip(examples, generated)
                if len(example.expected_chain) > 1
            ]
            total = len(pairs)
            correct = 0
            zero_calls = 0
            one_call = 0
            partial = 0
            wrong_order = 0
            for example, gen in pairs:
                pred_names = [c.get("name", "").lower().strip() for c in gen.calls]
                exp_names = [name.lower().strip() for name in example.expected_chain]
                if pred_names == exp_names:
                    correct += 1
                elif not pred_names:
                    zero_calls += 1
                elif len(pred_names) == 1 and len(exp_names) > 1:
                    one_call += 1
                elif all(name in pred_names for name in exp_names):
                    wrong_order += 1
                elif any(name in pred_names for name in exp_names):
                    partial += 1
            self.results["multi_step_success"] = correct / total if total else 0.0
            if total:
                wandb.log({
                    f"{self.label}/multi_step_zero_calls": zero_calls / total,
                    f"{self.label}/multi_step_one_call_only": one_call / total,
                    f"{self.label}/multi_step_partial": partial / total,
                    f"{self.label}/multi_step_wrong_order": wrong_order / total,
                })

        if "latency" in pending:
            subset = generated[:100]
            times = [g.elapsed_s for g in subset]
            total_tokens = sum(g.n_tokens for g in subset)
            if times and total_tokens:
                avg_time = float(np.mean(times))
                self.results["avg_latency_ms"] = avg_time * 1000
                self.results["ms_per_token"] = 1000 * avg_time / (total_tokens / len(times))
            else:
                self.results["avg_latency_ms"] = 0.0
                self.results["ms_per_token"] = 0.0

        if self.checkpoint_file:
            self._all_checkpoint_data[self.label] = dict(self.results)
            _save_checkpoint(self.checkpoint_file, self._all_checkpoint_data)
        for k, v in self.results.items():
            wandb.summary[f"{self.label}/{k}"] = v
        wandb.log({f"{self.label}/{k}": v for k, v in self.results.items()})
        return self.results


def compare_stages(all_results: Dict[str, Dict[str, float]]) -> str:
    stages = list(all_results.keys())
    keys: List[str] = []
    for results in all_results.values():
        for key in results:
            if key not in keys:
                keys.append(key)
    rows = ["| Metric | " + " | ".join(stages) + " |", "|---|" + "|".join(["---"] * len(stages)) + "|"]
    for key in keys:
        vals = []
        for stage in stages:
            value = all_results[stage].get(key)
            if value is None:
                vals.append("-")
            elif "accuracy" in key or "success" in key or "compliance" in key:
                vals.append(f"{100 * value:.1f}%")
            else:
                vals.append(f"{value:.2f}")
        rows.append(f"| {key} | " + " | ".join(vals) + " |")
    table = "\n".join(rows)
    print("\n" + "=" * 70)
    print("STAGE COMPARISON")
    print("=" * 70)
    print(table)
    print("=" * 70 + "\n")
    return table


def _local_compare(args) -> None:
    all_results: Dict[str, Dict[str, float]] = {}
    for stage, path in {
        "baseline": args.baseline_results,
        "sft": args.sft_results,
        "grpo": args.grpo_results,
    }.items():
        results = _load_stage_results_from_json(path, stage) if path else None
        if results:
            all_results[stage] = results
    if len(all_results) < 2:
        raise RuntimeError("local-compare requires at least two saved stage result files.")
    compare_stages(all_results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)


def _fetch_compare(args) -> None:
    api = wandb.Api()
    stage_run_paths = {
        "baseline": args.baseline_run_path,
        "sft": args.sft_run_path,
        "grpo": args.grpo_run_path,
    }
    all_results: Dict[str, Dict[str, float]] = {}
    for stage, run_path in stage_run_paths.items():
        if not run_path:
            continue
        run = api.run(run_path)
        summary = dict(run.summary)
        prefix = f"{stage}/"
        metrics = {
            k[len(prefix):]: float(v)
            for k, v in summary.items()
            if k.startswith(prefix) and isinstance(v, (int, float))
        }
        if metrics:
            all_results[stage] = metrics
    if len(all_results) < 2:
        raise RuntimeError("fetch-compare requires at least two completed runs.")
    compare_stages(all_results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)


async def _run_eval(args) -> None:
    checkpoint_file = args.checkpoint_file
    checkpoint_data: Dict[str, Any] = {}
    ckpt_path = Path(checkpoint_file)
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            checkpoint_data = {}

    if not os.getenv("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY environment variable is required for Tinker-based evaluation.")
    service_client = ServiceClient()
    sft_sampler = args.sft_sampler_path or _read_last_sampler_path(args.sft_output_dir)
    grpo_sampler = args.grpo_sampler_path or _read_last_sampler_path(args.grpo_output_dir)

    if args.mode in ("sft", "compare", "all") and not sft_sampler:
        raise RuntimeError(f"SFT sampler path not provided and not found in {args.sft_output_dir}/checkpoints.jsonl.")
    if args.mode in ("grpo", "compare", "all") and not grpo_sampler:
        raise RuntimeError(f"GRPO sampler path not provided and not found in {args.grpo_output_dir}/checkpoints.jsonl.")

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
    wandb.config.update({"n_eval_examples": len(examples)}, allow_val_change=True)

    async def _make_evaluator(label: str, model_path: Optional[str] = None) -> ToolUseEvaluator:
        if model_path:
            sc = await service_client.create_sampling_client_async(model_path=model_path)
        else:
            sc = await service_client.create_sampling_client_async(base_model=args.base_model)
        renderer = get_renderer(get_recommended_renderer_name(args.base_model), sc.get_tokenizer())
        return ToolUseEvaluator(sc, renderer, label=label, checkpoint_file=checkpoint_file, checkpoint_data=checkpoint_data)

    benchmarks = args.benchmarks or list(ToolUseEvaluator._BENCHMARK_RESULT_KEYS.keys())
    required_keys = [ToolUseEvaluator._BENCHMARK_RESULT_KEYS[b] for b in benchmarks if b in ToolUseEvaluator._BENCHMARK_RESULT_KEYS]

    def _stage_complete(stage: str) -> bool:
        return stage in checkpoint_data and all(key in checkpoint_data[stage] for key in required_keys)

    all_results: Dict[str, Dict[str, float]] = {}
    if args.mode in ("baseline", "compare", "all"):
        if _stage_complete("baseline"):
            all_results["baseline"] = checkpoint_data["baseline"]
        else:
            all_results["baseline"] = await (await _make_evaluator("baseline")).evaluate_all(examples, benchmarks=args.benchmarks)
    if args.mode in ("sft", "compare", "all"):
        if _stage_complete("sft"):
            all_results["sft"] = checkpoint_data["sft"]
        else:
            all_results["sft"] = await (await _make_evaluator("sft", model_path=sft_sampler)).evaluate_all(examples, benchmarks=args.benchmarks)
    if args.mode in ("grpo", "compare", "all"):
        if _stage_complete("grpo"):
            all_results["grpo"] = checkpoint_data["grpo"]
        else:
            all_results["grpo"] = await (await _make_evaluator("grpo", model_path=grpo_sampler)).evaluate_all(examples, benchmarks=args.benchmarks)

    if len(all_results) > 1:
        compare_stages(all_results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    if len(all_results) > 1:
        metrics = list(next(iter(all_results.values())).keys())
        stages = list(all_results.keys())
        table = wandb.Table(columns=["metric"] + stages)
        for metric in metrics:
            row = [metric] + [all_results[stage].get(metric, float("nan")) for stage in stages]
            table.add_data(*row)
        wandb.log({"eval/comparison_table": table})
    wandb.save(args.output)
    wandb.finish()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Three-stage evaluation (Tinker inference)")
    parser.add_argument("--mode", choices=["baseline", "sft", "grpo", "compare", "all", "fetch-compare", "local-compare"], default="all")
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--sft-sampler-path", default=None)
    parser.add_argument("--grpo-sampler-path", default=None)
    parser.add_argument("--sft-output-dir", default="./outputs/sft")
    parser.add_argument("--grpo-output-dir", default="./outputs/grpo")
    parser.add_argument("--output", default="outputs/evaluation_results.json")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--benchmarks", nargs="+", default=None, metavar="BENCHMARK")
    parser.add_argument("--checkpoint-file", default="outputs/eval_checkpoint.json")
    parser.add_argument("--baseline-run-path", default=None)
    parser.add_argument("--sft-run-path", default=None)
    parser.add_argument("--grpo-run-path", default=None)
    parser.add_argument("--baseline-results", default="outputs/eval_baseline.json")
    parser.add_argument("--sft-results", default="outputs/eval_sft.json")
    parser.add_argument("--grpo-results", default="outputs/eval_grpo.json")
    args = parser.parse_args()

    if args.mode == "fetch-compare":
        _fetch_compare(args)
    elif args.mode == "local-compare":
        _local_compare(args)
    else:
        asyncio.run(_run_eval(args))


if __name__ == "__main__":
    main()
