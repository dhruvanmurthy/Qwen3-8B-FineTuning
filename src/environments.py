"""
Atropos-pattern coordinator for bridging reward environments with GRPOTrainer.

Provides in-process environment management for single-machine training.
For multi-machine setups, swap with the full ``atropos`` package.
"""

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
from datasets import Dataset, concatenate_datasets

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


class AtroposCoordinator:
    """In-process coordinator bridging environments with TRL GRPOTrainer.

    Follows the Atropos pattern:
      * Environments provide prompts and compute rewards.
      * Coordinator aggregates and presents a unified interface to the trainer.

    For distributed multi-machine setups, replace with the full
    ``atropos`` package from NousResearch.
    """

    def __init__(
        self,
        environments: List[ToolUseEnvironment],
        reward_weights: Optional[List[float]] = None,
    ):
        assert len(environments) > 0, "Need at least one environment"
        self.environments = environments
        self.reward_weights = reward_weights or [1.0] * len(environments)
        assert len(self.environments) == len(self.reward_weights)

        total = sum(self.reward_weights)
        self.reward_weights = [w / total for w in self.reward_weights]

        logger.info(
            "AtroposCoordinator: %d environments — %s",
            len(environments),
            [e.source_name for e in environments],
        )

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def build_prompt_dataset(self) -> Dataset:
        """Build unified prompt dataset from all environments."""
        datasets = []
        for env, weight in zip(self.environments, self.reward_weights):
            n_samples = min(
                int(len(env.dataset) * weight * len(self.environments)),
                len(env.dataset),
            )
            indices = np.random.choice(len(env.dataset), n_samples, replace=False)
            datasets.append(env.dataset.select(indices))

        combined = concatenate_datasets(datasets).shuffle(seed=42)
        logger.info("Prompt dataset: %d examples", len(combined))
        return combined

    # ------------------------------------------------------------------
    # Reward interface
    # ------------------------------------------------------------------

    def get_reward_funcs(self) -> List[Callable]:
        """Return individual binary reward functions for GRPOTrainer.

        GRPOTrainer logs each reward dimension separately, which is
        useful for tracking reward-signal quality.
        """
        return [
            tool_name_reward,
            argument_match_reward,
            schema_validation_reward,
            full_chain_reward,
        ]

    def get_combined_reward_fn(self) -> Callable:
        """Return a single combined reward function.

        Averages all active reward signals (skips those whose
        required metadata is missing).
        """

        def combined_reward(completions: list[str], **kwargs) -> list[float]:
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

        return combined_reward
