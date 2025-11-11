"""
Extended particle geometry metrics.

Computes area, perimeter, equivalent diameters,
Feret diameters, ellipse axes, aspect ratio,
roundness, and solidity in µm domain.
"""

from __future__ import annotations
import math
import numpy as np
import cv2

from sem_psd.addons.geometry import (
    contour_px_to_um,
    perimeter_um,
    feret_diameters_um,
    ellipse_axes_um,
)
from sem_psd.addons.scale_tilt import tilt_corr_factor


def extended_metrics(cnt_px, umx: float, umy: float, tilt_deg: float = 0.0):
    """Compute extended shape metrics for a contour in the µm domain."""
    cnt_um = contour_px_to_um(cnt_px, umx, umy)
    area = float(cv2.contourArea(cnt_um.astype(np.float32)))
    if area <= 0:
        # Return zeros for invalid or empty contours
        return {k: 0.0 for k in [
            "area_um2", "perimeter_um", "d_eq_area_um", "d_eq_perim_um",
            "feret_min_um", "feret_max_um", "aspect_ratio",
            "ellipse_minor_um", "ellipse_major_um", "roundness", "solidity",
        ]}

    perim = perimeter_um(cnt_um)
    d_eq_area = 2.0 * math.sqrt(area / math.pi)
    d_eq_perim = perim / math.pi

    # Solidity = area / convex hull area
    hull = cv2.convexHull(cnt_um.astype(np.float32))
    area_convex = float(cv2.contourArea(hull)) if hull is not None else area
    solidity = float(area / max(area_convex, 1e-12))

    fmin, fmax = feret_diameters_um(cnt_um, step_deg=2)
    emin, emax = ellipse_axes_um(cnt_um)
    aspect = (fmin / fmax) if fmax > 0 else 0.0
    roundness = (4.0 * math.pi * area) / (perim * perim + 1e-12)

    corr = tilt_corr_factor(tilt_deg)
    return dict(
        area_um2=area * corr,
        perimeter_um=perim,
        d_eq_area_um=d_eq_area * corr,
        d_eq_perim_um=d_eq_perim * corr,
        feret_min_um=fmin * corr,
        feret_max_um=fmax * corr,
        aspect_ratio=aspect,
        ellipse_minor_um=emin * corr,
        ellipse_major_um=emax * corr,
        roundness=roundness,
        solidity=solidity,
    )
