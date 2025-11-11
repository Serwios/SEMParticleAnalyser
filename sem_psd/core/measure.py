"""
Component measurement and diameter statistics.

- Measure connected components with size/circularity/contrast filters
- Compute basic PSD stats (D10/50/90, mean, std, min, max)
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple
import cv2
import numpy as np


def measure_components(
    bw: np.ndarray,
    min_d_um: float,
    max_d_um: float,
    min_circ: float,
    um_per_px: float,
    lev_img: np.ndarray,
    min_rel_contrast: float,
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Measure foreground components with size, circularity, and local-contrast filters.

    Returns:
        List of (contour_px, diameter_um, circularity).
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    results: List[Tuple[np.ndarray, float, float]] = []

    for i in range(1, num):
        a = stats[i, cv2.CC_STAT_AREA]

        def area_from_d_um(d_um: float | None) -> float | None:
            if d_um is None:
                return None
            r_px = max(0.5, (d_um / 2.0) / um_per_px)
            return math.pi * r_px * r_px

        # Size gates via equivalent-circle area
        area_min = area_from_d_um(min_d_um) if min_d_um else None
        area_max = area_from_d_um(max_d_um) if max_d_um else None
        if area_min is not None and a < area_min:
            continue
        if area_max is not None and a > area_max:
            continue

        # Extract contour
        x, y, w, h = (
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
        )
        roi = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        cnt[:, 0, 0] += x
        cnt[:, 0, 1] += y

        # Geometric filters
        A = float(cv2.contourArea(cnt))
        if A < 1.0:
            continue
        P = float(cv2.arcLength(cnt, True))
        circ = (4.0 * math.pi * A) / (P * P + 1e-9)
        if circ < min_circ:
            continue

        # Local relative-contrast check on the leveled image
        mask_obj = np.zeros_like(bw)
        cv2.drawContours(mask_obj, [cnt], -1, 255, thickness=-1)

        ring = np.zeros_like(bw)
        d_px_est = 2.0 * math.sqrt(A / math.pi)
        ring_thick = max(3, int(0.15 * d_px_est))
        cv2.drawContours(ring, [cnt], -1, 255, thickness=ring_thick)
        ring = cv2.subtract(ring, mask_obj)

        vals_in = lev_img[mask_obj == 255]
        vals_bg = lev_img[ring == 255]
        if vals_bg.size < 30:
            # Broaden background ring if too thin
            cv2.drawContours(ring, [cnt], -1, 255, thickness=ring_thick * 2)
            ring = cv2.subtract(ring, mask_obj)
            vals_bg = lev_img[ring == 255]

        mean_in = float(vals_in.mean()) if vals_in.size else 0.0
        mean_bg = float(vals_bg.mean()) if vals_bg.size else 0.0
        rel_contrast = (mean_in - mean_bg) / 255.0
        if rel_contrast < min_rel_contrast:
            continue

        # Diameter from equivalent-area circle
        d_px = 2.0 * math.sqrt(A / math.pi)
        d_um = d_px * um_per_px
        results.append((cnt, float(d_um), circ))

    return results


def stats_from_diams(d_um: np.ndarray) -> Dict[str, float | int]:
    """Return basic PSD statistics for an array of diameters in µm."""
    return {
        "particles": int(d_um.size),
        "D10": float(np.percentile(d_um, 10)) if d_um.size else 0.0,
        "D50": float(np.percentile(d_um, 50)) if d_um.size else 0.0,
        "D90": float(np.percentile(d_um, 90)) if d_um.size else 0.0,
        "mean": float(np.mean(d_um)) if d_um.size else 0.0,
        "std": float(np.std(d_um)) if d_um.size else 0.0,
        "min": float(np.min(d_um)) if d_um.size else 0.0,
        "max": float(np.max(d_um)) if d_um.size else 0.0,
    }
