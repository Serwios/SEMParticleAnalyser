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
from sem_psd.addons.scale_tilt import (
    resolve_scale_xy,
    effective_um_per_px_for_isotropic_kernels,
    tilt_corr_factor,
)

# ---- Geometry helpers ----
from sem_psd.addons.geometry import (
    contour_px_to_um,
    perimeter_um,
    feret_diameters_um,
    ellipse_axes_um,
)

# ---- Extended metrics ----
from sem_psd.addons.metrics import extended_metrics

# ---- ISO 9276 weighted stats ----
from sem_psd.addons.iso_stats import iso9276_weighted_means

# ---- Bootstrap confidence intervals ----
from sem_psd.addons.bootstrap import bootstrap_ci_percentile, bootstrap_ci_mean

# ---- Result enrichment / CSV export ----
from sem_psd.addons.enrich import ParticleRow, enrich_results
from sem_psd.addons.csv_ext import write_csv_extended


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
