"""
Scale and tilt utilities.

Provides functions for:
- Resolving anisotropic or isotropic pixel scales
- Computing effective isotropic scale (geometric mean)
- Applying tilt correction factors
"""

from __future__ import annotations
import math
from typing import Optional, Tuple


def resolve_scale_xy(
    scale_um_per_px: Optional[float],
    scale_um_per_px_x: Optional[float] = None,
    scale_um_per_px_y: Optional[float] = None,
) -> Tuple[float, float]:
    """Return (um_per_px_x, um_per_px_y) with fallback to isotropic scale."""
    if scale_um_per_px_x and scale_um_per_px_y:
        return float(scale_um_per_px_x), float(scale_um_per_px_y)
    if scale_um_per_px and scale_um_per_px > 0:
        s = float(scale_um_per_px)
        return s, s
    raise ValueError("Valid scale (µm/px) is required: set either isotropic or X/Y.")


def effective_um_per_px_for_isotropic_kernels(umx: float, umy: float) -> float:
    """Return geometric mean for isotropic operations (e.g., LoG σ, morphology radii)."""
    return math.sqrt(umx * umy)


def tilt_corr_factor(tilt_deg: float) -> float:
    """Return correction factor for projection shrinkage due to tilt (1 / cos(tilt))."""
    if abs(tilt_deg) < 1e-9:
        return 1.0
    c = math.cos(math.radians(tilt_deg))
    return 1.0 / max(1e-6, c)
