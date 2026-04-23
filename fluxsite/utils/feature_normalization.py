#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature normalization utilities.

Provides standardization for pre-extracted ESM features before training to keep feature scales
consistent across proteins and improve training stability.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


class _RunningStats:
    """Utility for accumulating per-dimension mean and variance."""

    def __init__(self) -> None:
        self.count: int = 0
        self.sum: Optional[np.ndarray] = None
        self.sumsq: Optional[np.ndarray] = None

    def update(self, batch: np.ndarray) -> None:
        """Update running statistics with a new batch.

        Args:
            batch: A 2D array of shape (N, D).
        """
        if batch.size == 0:
            return

        if batch.ndim != 2:
            raise ValueError(f", {batch.shape}")

        # Use float32 inputs to reduce temporary memory; keep float64 precision for reductions
        batch = np.asarray(batch, dtype=np.float32)
        batch_sum = np.add.reduce(batch, axis=0, dtype=np.float64)
        batch_sumsq = np.einsum("ij,ij->j", batch, batch, dtype=np.float64)

        if self.sum is None:
            self.sum = batch_sum
            self.sumsq = batch_sumsq
            self.count = batch.shape[0]
        else:
            self.sum += batch_sum
            self.sumsq += batch_sumsq
            self.count += batch.shape[0]

    def finalize(self, epsilon: float) -> Optional[Dict[str, np.ndarray]]:
        if self.count == 0 or self.sum is None or self.sumsq is None:
            return None

        mean = self.sum / max(self.count, 1)

        if self.count > 1:
            variance = (self.sumsq - (np.square(self.sum) / self.count)) / (self.count - 1)
        else:
            variance = np.zeros_like(mean)

        variance = np.clip(variance, epsilon ** 2, None)
        std = np.sqrt(variance)
        std = np.where(std < epsilon, 1.0, std)

        return {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "count": np.array([self.count], dtype=np.int64),
        }


class FeatureNormalizer:
    """Computes and applies feature normalization."""

    TARGET_KEYS: List[str] = [
        "sequence_features",
        "local_features",
        "global_features",
        "structure_features",
        "structure_local",
        "structure_global",
    ]

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon
        self.stats: Dict[str, Dict[str, np.ndarray]] = {}
        self._aggregators: Optional[Dict[str, _RunningStats]] = None

    def fit(self, samples: Iterable[Dict]) -> None:
        """Estimate mean and standard deviation from training samples."""
        aggregators = {key: _RunningStats() for key in self.TARGET_KEYS}
        total_samples = 0

        for sample in samples:
            total_samples += 1
            for key, aggregator in aggregators.items():
                value = sample.get(key)
                if value is None:
                    continue
                array = np.asarray(value)
                if array.ndim == 1:
                    array = array.reshape(1, -1)
                elif array.ndim != 2:
                    raise ValueError(f"{key} 1D 2D, {array.shape}")
                aggregator.update(array)

        if total_samples == 0:
            logger.warning("fit Samples,Skip Parameters ")
            self.stats = {}
            return

        self.stats = {}
        for key, aggregator in aggregators.items():
            result = aggregator.finalize(self.epsilon)
            if result is not None:
                self.stats[key] = result

        if not self.stats:
            logger.warning(" Parameters.")
        else:
            logger.info(" Parameters: %s", ", ".join(self.stats.keys()))

    def transform(self, samples: List[Dict], inplace: bool = True) -> List[Dict]:
        """Normalize features in a list of samples."""
        if not self.stats:
            logger.debug(" Parameters,Skip transform")
            return samples

        target_samples = samples if inplace else [dict(sample) for sample in samples]

        for sample in target_samples:
            for key, stat in self.stats.items():
                if key not in sample:
                    continue
                value = sample[key]
                if value is None:
                    continue

                array = np.asarray(value, dtype=np.float32)
                mean = stat["mean"].astype(np.float32)
                std = stat["std"].astype(np.float32)

                if array.ndim == 1:
                    normalized = (array - mean) / std
                elif array.ndim == 2:
                    normalized = (array - mean) / std
                else:
                    raise ValueError(f"{key} 1D 2D, {array.shape}")

                sample[key] = normalized.astype(np.float32)

        return target_samples

    def transform_stream(self, samples: Iterable[Dict], inplace: bool = True) -> Iterable[Dict]:
        """Normalize an iterable of samples as a generator to avoid loading everything into memory.

        Args:
            samples: Any iterable of samples.
            inplace: Whether to modify input samples in place.

        Yields:
            Normalized samples.
        """
        if not self.stats:
            yield from samples
            return

        for sample in samples:
            target = sample if inplace else dict(sample)
            for key, stat in self.stats.items():
                value = target.get(key)
                if value is None:
                    continue
                array = np.asarray(value, dtype=np.float32)
                mean = stat["mean"].astype(np.float32)
                std = stat["std"].astype(np.float32)
                if array.ndim == 1:
                    normalized = (array - mean) / std
                elif array.ndim == 2:
                    normalized = (array - mean) / std
                else:
                    raise ValueError(f"{key} 1D 2D, {array.shape}")
                target[key] = normalized.astype(np.float32)
            yield target

    def fit_transform(self, samples: List[Dict]) -> List[Dict]:
        """Convenience helper: fit then normalize in place."""
        self.fit(samples)
        return self.transform(samples)

    def fit_incremental(self, samples: Iterable[Dict]) -> None:
        """Update statistics incrementally, suitable for streaming large datasets.

        Call this method multiple times to accumulate statistics, then call finalize_stats()
        to compute the final parameters.
        """
        if self._aggregators is None:
            self._aggregators = {key: _RunningStats() for key in self.TARGET_KEYS}
        
        for sample in samples:
            for key, aggregator in self._aggregators.items():
                value = sample.get(key)
                if value is None:
                    continue
                array = np.asarray(value)
                if array.ndim == 1:
                    array = array.reshape(1, -1)
                elif array.ndim != 2:
                    logger.warning(f"{key} 1D 2D, {array.shape},Skip")
                    continue
                aggregator.update(array)

    def finalize_stats(self) -> None:
        """Finalize incremental statistics and compute normalization parameters.

        This must be called after fit_incremental().
        """
        if self._aggregators is None:
            logger.warning("finalize_stats ")
            self.stats = {}
            return
        
        self.stats = {}
        for key, aggregator in self._aggregators.items():
            result = aggregator.finalize(self.epsilon)
            if result is not None:
                self.stats[key] = result
        
        if not self.stats:
            logger.warning(" Parameters.")
        else:
            logger.info(" Completed, Parameters: %s", ", ".join(self.stats.keys()))
        
        # Release aggregators to free memory
        self._aggregators = None

    def save(self, path: Path) -> None:
        """Save normalization parameters to disk."""
        if not self.stats:
            logger.warning(" Save Parameters")
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {}
        metadata = {"counts": {}, "epsilon": self.epsilon}
        for key, stat in self.stats.items():
            arrays[f"{key}_mean"] = stat["mean"]
            arrays[f"{key}_std"] = stat["std"]
            metadata["counts"][key] = int(stat["count"][0])

        np.savez_compressed(path, **arrays)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(" Parameters Save %s", path)

    def load(self, path: Path) -> None:
        """Load normalization parameters from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f" Parameters: {path}")

        data = np.load(path)
        stats: Dict[str, Dict[str, np.ndarray]] = {}

        for key in self.TARGET_KEYS:
            mean_key = f"{key}_mean"
            std_key = f"{key}_std"
            if mean_key in data and std_key in data:
                stats[key] = {
                    "mean": data[mean_key].astype(np.float32),
                    "std": np.maximum(data[std_key].astype(np.float32), self.epsilon),
                    "count": np.array([0], dtype=np.int64),
                }

        if not stats:
            raise ValueError(f" {path} ")

        self.stats = stats

        logger.info(" %s Load Parameters", path)

    def get_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Return a copy of the current normalization statistics."""
        return {key: {k: v.copy() for k, v in stat.items()} for key, stat in self.stats.items()}
