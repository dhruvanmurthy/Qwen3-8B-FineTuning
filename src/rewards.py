"""
Binary verifiable reward functions for tool-use GRPO training.

Each function follows TRL GRPOTrainer's reward function interface:
    reward_fn(completions: list[str], **kwargs) -> list[float]

All rewards are binary: 1.0 (correct) or 0.0 (incorrect).
No learned reward models — only programmatic verification.
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
    """
    patterns = [
        r'```(?:json)?\s*(\{.*?\})\s*```',
        r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
        r'<\|function_call\|>\s*(\{.*?\})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if "name" in data:
                    return data
            except json.JSONDecodeError:
                continue

    # Fallback: find any JSON object with "name" key
    for match in re.finditer(
        r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*\}', text, re.DOTALL
    ):
        try:
            data = json.loads(match.group())
            if "name" in data:
                return data
        except json.JSONDecodeError:
            continue
    return None


def extract_tool_calls(text: str) -> List[Dict]:
    """Extract all tool calls from a multi-step completion."""
    calls = []
    for match in re.finditer(
        r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*\}', text, re.DOTALL
    ):
        try:
            data = json.loads(match.group())
            if "name" in data:
                calls.append(data)
        except json.JSONDecodeError:
            continue
    return calls


# ---------------------------------------------------------------------------
# Binary reward functions (GRPOTrainer-compatible)
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


def argument_match_reward(
    completions: list[str],
    expected_args: list[str] | None = None,
    **kwargs,
) -> list[float]:
    """1.0 if predicted arguments exactly match expected, else 0.0."""
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

        rewards.append(1.0 if pred_args == expected else 0.0)
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


# ---------------------------------------------------------------------------
# Composite reward (used by GRPO training and environments)
# ---------------------------------------------------------------------------

def compute_rewards(
    completions: list[str],
    metadata: dict,
    return_components: bool = False,
) -> "list[float] | tuple[list[float], dict[str, list[float]]]":
    """Compute average of all applicable binary reward signals.

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
        reward_fns["argument_match"] = argument_match_reward
    if metadata.get("expected_chain") and metadata["expected_chain"] != "[]":
        reward_fns["chain"] = full_chain_reward

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
