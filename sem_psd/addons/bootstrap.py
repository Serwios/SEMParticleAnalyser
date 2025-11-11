"""
Bootstrap confidence interval utilities.

Provides percentile- and mean-based bootstrap CIs
for particle-size statistics and other metrics.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


def bootstrap_ci_percentile(
    x: np.ndarray,
    q: float = 50,
    n: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Return a bootstrap CI for a given percentile (e.g., D50)."""
    if x.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    # Resample with replacement (n replicates)
    idx = rng.integers(0, x.size, size=(n, x.size))
    qs = np.percentile(x[idx], q, axis=1)
    lo, hi = np.percentile(qs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci_mean(
    x: np.ndarray,
    n: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Return a bootstrap CI for the sample mean."""
    if x.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    # Resample and compute mean for each bootstrap replicate
    idx = rng.integers(0, x.size, size=(n, x.size))
    means = np.mean(x[idx], axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
