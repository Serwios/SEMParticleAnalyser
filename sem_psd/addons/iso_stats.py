"""
ISO 9276 weighted statistics.

Computes ISO-style particle size metrics:
  D10, D50, D90 (number),
  D32 (Sauter mean = Σd³/Σd²),
  D43 (De Brouckere mean = Σd⁴/Σd³),
  and basic descriptive stats.
"""

from __future__ import annotations
from typing import Dict
import numpy as np


def iso9276_weighted_means(d_um: np.ndarray) -> Dict[str, float]:
    """Return ISO 9276 weighted and number-based particle size statistics."""
    out = {
        "D10": 0.0, "D50": 0.0, "D90": 0.0,
        "D32": 0.0, "D43": 0.0,
        "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
        "particles": int(d_um.size),
    }
    if d_um.size == 0:
        return out

    d = d_um.astype(np.float64)
    # Number-based percentiles and descriptive stats
    out["D10"] = float(np.percentile(d, 10))
    out["D50"] = float(np.percentile(d, 50))
    out["D90"] = float(np.percentile(d, 90))
    out["mean"] = float(np.mean(d))
    out["std"] = float(np.std(d))
    out["min"] = float(np.min(d))
    out["max"] = float(np.max(d))

    # Volume-weighted means (ISO 9276)
    d2, d3, d4 = d**2, d**3, d**4
    s2, s3, s4 = d2.sum(), d3.sum(), d4.sum()
    out["D32"] = float(s3 / (s2 if s2 != 0 else 1.0))
    out["D43"] = float(s4 / (s3 if s3 != 0 else 1.0))
    return out
