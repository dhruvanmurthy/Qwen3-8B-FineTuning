"""
Programmatic reward functions for tool-use GRPO training.

Reward functions use the same call signature throughout the Tinker-based
training loop:
    reward_fn(completions: list[str], **kwargs) -> list[float]

The individual signals are fully deterministic. Some are binary and some use
partial credit to provide a smoother training signal.
"""

import json
import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Tool call extraction
# ---------------------------------------------------------------------------

def extract_tool_call(text: str) -> Optional[Dict]:
    """Extract a single tool call from model completion.

    Supports:
      - Fenced JSON blocks (```json ... ```)
      - <tool_call> XML tags
      - Qwen <|function_call|> format
      - Raw JSON with "name" key
    
    Also strips thinking tags to improve extraction from reasoning outputs.
    """
    # Strip thinking blocks (complete and incomplete) if present
    text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text_clean = re.sub(r'<think>.*$', '', text_clean, flags=re.DOTALL)
    
    patterns = [
        r'```(?:json)?\s*(\{.*?\})\s*```',
        r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
        r'<\|function_call\|>\s*(\{.*?\})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text_clean, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if "name" in data:
                    return data
            except json.JSONDecodeError:
                continue

    # Fallback: find any JSON object with "name" key
    for match in re.finditer(
        r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*\}', text_clean, re.DOTALL
    ):
        try:
            data = json.loads(match.group())
            if "name" in data:
                return data
        except json.JSONDecodeError:
            continue
    return None


def extract_tool_calls(text: str) -> List[Dict]:
    """Extract all tool calls from a multi-step completion.

    Tries <tool_call>...</tool_call> blocks first (handles pretty-printed
    nested JSON that the flat brace regex can't match), then falls back to
    scanning for flat JSON objects as a best-effort catch-all.
    """
    # Strip thinking blocks
    text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text_clean = re.sub(r'<think>.*$', '', text_clean, flags=re.DOTALL)

    calls = []

    # Primary: <tool_call> ... </tool_call> blocks (model's native format)
    for block in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text_clean, re.DOTALL):
        try:
            data = json.loads(block.group(1))
            if "name" in data:
                calls.append(data)
        except json.JSONDecodeError:
            continue

    if calls:
        return calls

    # Fallback: scan for flat single-line JSON objects with "name" key
    for match in re.finditer(
        r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*\}', text_clean, re.DOTALL
    ):
        try:
            data = json.loads(match.group())
            if "name" in data:
                calls.append(data)
        except json.JSONDecodeError:
            continue
    return calls


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def tool_name_reward(
    completions: list[str],
    expected_tool: list[str] | None = None,
    **kwargs,
) -> list[float]:
    """1.0 if the predicted tool name matches expected, else 0.0."""
    if expected_tool is None:
        return [0.0] * len(completions)

    rewards = []
    for completion, expected in zip(completions, expected_tool):
        call = extract_tool_call(completion)
        if call and call.get("name", "").lower().strip() == expected.lower().strip():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def schema_validation_reward(
    completions: list[str],
    **kwargs,
) -> list[float]:
    """1.0 if completion contains a structurally valid tool call, else 0.0."""
    rewards = []
    for completion in completions:
        call = extract_tool_call(completion)
        if (
            call is not None
            and isinstance(call.get("name"), str)
            and len(call["name"]) > 0
            and isinstance(
                call.get("arguments", call.get("parameters", {})), dict
            )
        ):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def full_chain_reward(
    completions: list[str],
    expected_chain: list[str] | None = None,
    **kwargs,
) -> list[float]:
    """1.0 if all tool calls in a multi-step chain match expected sequence."""
    if expected_chain is None:
        return [0.0] * len(completions)

    rewards = []
    for completion, chain_str in zip(completions, expected_chain):
        try:
            expected = (
                json.loads(chain_str)
                if isinstance(chain_str, str)
                else chain_str
            )
        except (json.JSONDecodeError, TypeError):
            rewards.append(0.0)
            continue

        predicted = extract_tool_calls(completion)
        pred_names = [c.get("name", "").lower().strip() for c in predicted]
        exp_names = [
            (e.lower().strip() if isinstance(e, str) else e.get("name", "").lower().strip())
            for e in expected
        ]

        rewards.append(1.0 if pred_names == exp_names else 0.0)
    return rewards


def chain_partial_reward(
    completions: list[str],
    expected_chain: list[str] | None = None,
    **kwargs,
) -> list[float]:
    """Partial credit: fraction of expected tools produced in correct order.

    Scores each position independently — correct tool at position i earns
    1/N credit.  Gives gradient signal even when the chain is partially right.
    """
    if expected_chain is None:
        return [0.0] * len(completions)

    rewards = []
    for completion, chain_str in zip(completions, expected_chain):
        try:
            expected = (
                json.loads(chain_str)
                if isinstance(chain_str, str)
                else chain_str
            )
        except (json.JSONDecodeError, TypeError):
            rewards.append(0.0)
            continue

        predicted = extract_tool_calls(completion)
        pred_names = [c.get("name", "").lower().strip() for c in predicted]
        exp_names = [
            (e.lower().strip() if isinstance(e, str) else e.get("name", "").lower().strip())
            for e in expected
        ]
        if not exp_names:
            rewards.append(0.0)
            continue

        # Position-wise credit
        score = sum(
            1 for i, name in enumerate(exp_names)
            if i < len(pred_names) and pred_names[i] == name
        ) / len(exp_names)
        rewards.append(score)
    return rewards


def argument_f1_reward(
    completions: list[str],
    expected_args: list[str] | None = None,
    **kwargs,
) -> list[float]:
    """Partial credit based on argument key-value F1.

    For each completion, computes:
        precision = matching_keys / predicted_keys
        recall    = matching_keys / expected_keys
        F1        = 2 * p * r / (p + r)

    A key-value pair is "matching" when both the key AND string-normalised
    value are equal.  Returns 1.0 for exact match, 0.0 for no overlap.
    """
    if expected_args is None:
        return [0.0] * len(completions)

    rewards = []
    for completion, expected_str in zip(completions, expected_args):
        call = extract_tool_call(completion)
        if call is None:
            rewards.append(0.0)
            continue

        try:
            expected = (
                json.loads(expected_str)
                if isinstance(expected_str, str)
                else expected_str
            )
        except (json.JSONDecodeError, TypeError):
            rewards.append(0.0)
            continue

        pred_args = call.get("arguments", call.get("parameters", {}))
        if isinstance(pred_args, str):
            try:
                pred_args = json.loads(pred_args)
            except json.JSONDecodeError:
                rewards.append(0.0)
                continue

        if not isinstance(pred_args, dict) or not isinstance(expected, dict):
            rewards.append(0.0)
            continue

        # Build flat key=value sets for comparison (stringify values)
        def _flat(d: dict) -> set:
            return {f"{k}={json.dumps(v, sort_keys=True)}" for k, v in d.items()}

        pred_set = _flat(pred_args)
        exp_set = _flat(expected)

        if not pred_set and not exp_set:
            rewards.append(1.0)
            continue

        tp = len(pred_set & exp_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(exp_set) if exp_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rewards.append(f1)
    return rewards


# ---------------------------------------------------------------------------
# Composite reward (used by GRPO training and environments)
# ---------------------------------------------------------------------------

def compute_rewards(
    completions: list[str],
    metadata: dict,
    return_components: bool = False,
) -> "list[float] | tuple[list[float], dict[str, list[float]]]":
    """Compute average of all applicable reward signals.

    Uses exact schema and tool checks plus partial-credit scoring for argument
    and chain quality to provide a smoother gradient signal than strict
    all-or-nothing matching.

    Args:
        completions: List of completion strings to grade.
        metadata: Dict with expected_tool, expected_args, expected_chain keys.
        return_components: If True, also return per-component reward breakdown.

    Returns:
        Mean rewards list, or (mean_rewards, component_rewards) if return_components.
    """
    reward_fns: dict = {"schema": schema_validation_reward}
    if metadata.get("expected_tool"):
        reward_fns["tool_name"] = tool_name_reward
    if metadata.get("expected_args"):
        # Use F1 for partial credit instead of strict binary match
        reward_fns["argument_f1"] = argument_f1_reward
    if metadata.get("expected_chain") and metadata["expected_chain"] != "[]":
        # Full credit for exact chain + partial credit for positional overlap
        reward_fns["chain"] = full_chain_reward
        reward_fns["chain_partial"] = chain_partial_reward

    n_fns = len(reward_fns)
    all_rewards = [0.0] * len(completions)
    component_rewards: dict[str, list[float]] = {}

    for name, fn in reward_fns.items():
        fn_rewards = fn(completions, **metadata)
        component_rewards[name] = list(fn_rewards)
        for i, r in enumerate(fn_rewards):
            all_rewards[i] += r / n_fns

    if return_components:
        return all_rewards, component_rewards
    return all_rewards
