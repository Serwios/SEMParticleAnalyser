"""
Contour geometry utilities (µm domain).

- Convert pixel contours to µm
- Perimeter computation
- Feret min/max diameters
- Ellipse-fit axes with robust fallback
"""

from __future__ import annotations
import math
from typing import Tuple
import numpy as np
import cv2


def contour_px_to_um(cnt_px: np.ndarray, umx: float, umy: float) -> np.ndarray:
    """Scale a pixel contour to µm using per-axis pixel sizes (umx, umy)."""
    c = cnt_px.astype(np.float32).copy()
    c[:, 0, 0] *= umx
    c[:, 0, 1] *= umy
    return c


def perimeter_um(cnt_um: np.ndarray) -> float:
    """Compute polygonal perimeter (µm) for a single OpenCV-style contour."""
    p = 0.0
    pts = cnt_um[:, 0, :]
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        p += math.hypot(x1 - x0, y1 - y0)
    return float(p)


def feret_diameters_um(cnt_um: np.ndarray, step_deg: int = 2) -> Tuple[float, float]:
    """Return (min_feret, max_feret) in µm via brute-force angular sweep."""
    pts = cnt_um[:, 0, :].astype(np.float64)
    if len(pts) < 3:
        return 0.0, 0.0
    ctr = pts.mean(axis=0)
    pts0 = pts - ctr
    ferets = []
    for ang in range(0, 180, max(1, step_deg)):
        th = math.radians(ang)
        R = np.array([[math.cos(th), -math.sin(th)],
                      [math.sin(th),  math.cos(th)]], dtype=np.float64)
        pr = pts0 @ R.T
        ferets.append(pr[:, 0].max() - pr[:, 0].min())
    ferets = np.asarray(ferets)
    return float(ferets.min()), float(ferets.max())


def ellipse_axes_um(cnt_um: np.ndarray) -> Tuple[float, float]:
    """
    Return (minor_axis, major_axis) in µm from an ellipse fit.
    Falls back to equivalent-diameter circle if fit fails or points < 5.
    """
    pts = cnt_um.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] >= 5:
        try:
            (_, _), (MA, ma), _ = cv2.fitEllipse(pts)
            a, b = max(MA, ma), min(MA, ma)
            return float(b), float(a)
        except Exception:
            pass
    area = float(cv2.contourArea(cnt_um.astype(np.float32)))
    if area <= 0:
        return 0.0, 0.0
    d_eq = 2.0 * math.sqrt(area / math.pi)
    return float(d_eq), float(d_eq)
