"""
Distributed sampling and per-prompt statistics utilities for RL trainers.

Provides:
- DistributedKRepeatSampler: Ensures each prompt appears exactly K times in global batch.
- PerPromptStatTracker: Per-prompt reward normalization for advantage computation.

Reference: DiffusionNFT (arXiv 2509.16117).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class DistributedKRepeatSampler(Sampler):
    """Distributed sampler ensuring each prompt appears exactly K times globally.

    For per-prompt stat tracking, we need K samples per prompt to compute
    meaningful per-prompt advantages. This sampler:
    1. Selects M = total_samples // K unique prompts
    2. Repeats each prompt K times → M*K = total_samples indices
    3. Shuffles and shards across ranks so each rank gets batch_size indices

    Constraints (validated at init):
    - total_samples = num_replicas * batch_size
    - total_samples % K == 0  (so M is integer)
    - batch_size * num_replicas % K == 0

    This is an infinite sampler: each `__iter__` yields one batch of indices
    per rank. Call `set_epoch(epoch)` before each iteration for determinism.

    Used with DataLoader's `batch_sampler` param and `num_workers=0` to avoid
    prefetching issues with resume.

    Reference: DiffusionNFT scripts/train_nft_sd3.py DistributedKRepeatSampler
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        k: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ):
        """
        Args:
            dataset_size: Total number of prompts in the dataset.
            batch_size: Per-GPU batch size for NFT sampling.
            k: Number of images per prompt (K-repeat count).
            num_replicas: World size (number of GPUs).
            rank: This process's rank.
            seed: Base random seed for reproducibility.
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, (
            f"total_samples ({self.total_samples} = {num_replicas} * {batch_size}) "
            f"must be divisible by k ({k})"
        )
        self.m = self.total_samples // self.k  # number of unique prompts per iteration
        self.epoch = 0

    def __iter__(self):
        """Yield one batch of indices for this rank (infinite generator)."""
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # Select M unique prompts from the dataset
            indices = torch.randperm(self.dataset_size, generator=g)[: self.m].tolist()
            # Repeat each prompt K times
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            # Shuffle the repeated indices
            shuffled_order = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_order]
            # Shard: each rank gets batch_size consecutive elements
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling.

        Callers must call this BEFORE each ``next()`` of the sampler iter
        to get a fresh global K-repeat partition; the sampler reads
        ``self.epoch`` on every yield resume.
        """
        self.epoch = epoch


class PerPromptStatTracker:
    """Per-prompt reward normalization for advantage computation.

    Computes advantages by normalizing rewards relative to per-prompt
    mean and standard deviation. This ensures that the advantage signal
    reflects whether a sample is better/worse *for that specific prompt*,
    rather than across all prompts.

    Adapted from DiffusionNFT's per-prompt statistic tracker.
    """

    def __init__(self, global_std: bool = False):
        """
        Args:
            global_std: If True, use global std across all prompts instead
                of per-prompt std. Can be more stable with few samples per prompt.
        """
        self.global_std = global_std
        self.stats: dict[str, Any] = {}
        self.history_prompts: set[int] = set()

    def update(self, prompts: list[str], rewards: np.ndarray) -> np.ndarray:
        """Compute per-prompt normalized advantages.

        Args:
            prompts: List of prompt strings [N].
            rewards: Reward scores [N] or [N, T] (numpy array).

        Returns:
            Advantages array with same shape as rewards.
        """
        prompts_arr = np.array(prompts)
        rewards_arr = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts_arr)
        advantages = np.zeros_like(rewards_arr)

        # Accumulate per-prompt reward history
        for prompt in unique:
            prompt_rewards = rewards_arr[prompts_arr == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards.tolist() if prompt_rewards.ndim == 1 else prompt_rewards.flatten().tolist())
            self.history_prompts.add(hash(prompt))

        # Compute advantages
        # NOTE: self.stats[prompt] is kept as list (not converted to ndarray)
        # so that .extend() in the accumulation loop above works across
        # multiple update() calls without clear() in between.
        for prompt in unique:
            history = np.array(self.stats[prompt])
            prompt_rewards = rewards_arr[prompts_arr == prompt]
            mean = np.mean(history, axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards_arr, axis=0, keepdims=True) + 1e-4
            else:
                std = np.std(history, axis=0, keepdims=True) + 1e-4
            advantages[prompts_arr == prompt] = (prompt_rewards - mean) / std

        return advantages

    def get_stats(self) -> tuple[float, int]:
        """Get summary statistics.

        Returns:
            Tuple of (average_group_size, number_of_unique_prompts_seen).
        """
        avg_group_size = (
            sum(len(v) if isinstance(v, (list, np.ndarray)) else 1 for v in self.stats.values())
            / max(len(self.stats), 1)
        )
        return avg_group_size, len(self.history_prompts)

    def get_mean_of_top_rewards(self, top_percentage: float) -> float:
        """Get mean of top-K% rewards across all prompts.

        Args:
            top_percentage: Percentage (0-100) of top rewards to average.

        Returns:
            Mean of top rewards, 0.0 if no data.
        """
        if not self.stats:
            return 0.0
        assert 0 <= top_percentage <= 100

        per_prompt_top_means = []
        for prompt_rewards in self.stats.values():
            rewards = np.array(prompt_rewards) if isinstance(prompt_rewards, list) else prompt_rewards
            if rewards.size == 0:
                continue
            if top_percentage == 100:
                per_prompt_top_means.append(np.mean(rewards))
                continue
            lower_bound = 100 - top_percentage
            threshold = np.percentile(rewards, lower_bound)
            top_rewards = rewards[rewards >= threshold]
            if top_rewards.size > 0:
                per_prompt_top_means.append(np.mean(top_rewards))

        return float(np.mean(per_prompt_top_means)) if per_prompt_top_means else 0.0

    def clear(self) -> None:
        """Clear all accumulated statistics."""
        self.stats = {}

    def state_dict(self) -> dict[str, Any]:
        """Serialize tracker state for checkpoint saving."""
        # Convert numpy arrays to lists for safe pickling
        serialized_stats = {}
        for k, v in self.stats.items():
            if isinstance(v, np.ndarray):
                serialized_stats[k] = v.tolist()
            elif isinstance(v, list):
                serialized_stats[k] = v
            else:
                serialized_stats[k] = [v]
        return {
            "stats": serialized_stats,
            "history_prompts": list(self.history_prompts),
            "global_std": self.global_std,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore tracker state from checkpoint."""
        self.stats = {k: v for k, v in state.get("stats", {}).items()}
        self.history_prompts = set(state.get("history_prompts", []))
        logger.info(
            f"  PerPromptStatTracker restored: {len(self.stats)} prompts, "
            f"{len(self.history_prompts)} unique seen"
        )
