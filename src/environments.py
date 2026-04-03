"""
Reward environment utilities for tool-use GRPO training.

Provides:
  - ToolUseEnvironment: per-source environment that supplies prompts and
    computes rewards. Used by the GRPO training loop.
  - compute_combined_reward: standalone function for grading completions
    with all applicable reward signals.

Compatible with both Tinker-based and standalone training loops.
"""

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
from datasets import Dataset

from rewards import (
    argument_match_reward,
    full_chain_reward,
    schema_validation_reward,
    tool_name_reward,
)

logger = logging.getLogger(__name__)


class ToolUseEnvironment:
    """Single-source environment that provides prompts and computes rewards."""

    def __init__(
        self,
        dataset: Dataset,
        source_name: str,
        reward_fns: Optional[List[Callable]] = None,
    ):
        self.dataset = dataset
        self.source_name = source_name
        self.reward_fns = reward_fns or [tool_name_reward, schema_validation_reward]
        self._index = 0

    def __len__(self):
        return len(self.dataset)

    def get_batch(self, batch_size: int) -> Dict[str, list]:
        """Return next batch of prompts with metadata."""
        indices = []
        for _ in range(batch_size):
            indices.append(self._index % len(self.dataset))
            self._index += 1
        batch = self.dataset.select(indices)
        return {col: batch[col] for col in batch.column_names}

    def score(self, completions: List[str], metadata: Dict) -> List[float]:
        """Average of all binary reward signals for this environment."""
        all_rewards = []
        for fn in self.reward_fns:
            rewards = fn(completions, **metadata)
            all_rewards.append(rewards)
        return np.mean(all_rewards, axis=0).tolist()


def compute_combined_reward(completions: list[str], **kwargs) -> list[float]:
    """Compute average of all applicable binary reward signals.

    Standalone function usable from any training loop (Tinker or otherwise).
    """
    all_rewards: list[list[float]] = []

    # Schema reward (always active — needs no metadata)
    all_rewards.append(schema_validation_reward(completions, **kwargs))

    if kwargs.get("expected_tool") is not None:
        all_rewards.append(tool_name_reward(completions, **kwargs))

    if kwargs.get("expected_args") is not None:
        all_rewards.append(argument_match_reward(completions, **kwargs))

    if kwargs.get("expected_chain") is not None:
        all_rewards.append(full_chain_reward(completions, **kwargs))

    return np.mean(all_rewards, axis=0).tolist()
