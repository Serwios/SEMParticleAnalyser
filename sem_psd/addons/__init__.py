"""
Add-ons package for SEM PSD analysis.

Provides helper functions for:
- scale and tilt corrections
- contour geometry calculations
- extended particle metrics
- ISO 9276 weighted statistics
- bootstrap confidence intervals
- enriched results and CSV export
"""

# ---- Scale / tilt utilities ----
from .scale_tilt import (
    resolve_scale_xy,
    effective_um_per_px_for_isotropic_kernels,
    tilt_corr_factor,
)

# ---- Geometry helpers ----
from .geometry import (
    contour_px_to_um,
    perimeter_um,
    feret_diameters_um,
    ellipse_axes_um,
)

# ---- Extended metrics ----
from .metrics import extended_metrics

# ---- ISO 9276 weighted stats ----
from .iso_stats import iso9276_weighted_means

# ---- Bootstrap confidence intervals ----
from .bootstrap import bootstrap_ci_percentile, bootstrap_ci_mean

# ---- Result enrichment / CSV export ----
from .enrich import ParticleRow, enrich_results
from .csv_ext import write_csv_extended


__all__ = [
    # scale / tilt
    "resolve_scale_xy", "effective_um_per_px_for_isotropic_kernels", "tilt_corr_factor",
    # geometry
    "contour_px_to_um", "perimeter_um", "feret_diameters_um", "ellipse_axes_um",
    # metrics
    "extended_metrics",
    # ISO 9276
    "iso9276_weighted_means",
    # bootstrap
    "bootstrap_ci_percentile", "bootstrap_ci_mean",
    # enrichment / csv
    "ParticleRow", "enrich_results", "write_csv_extended",
]
