# sem_psd_addons.py
# Add-ons: extended particle metrics, ISO 9276 weighted PSD, bootstrap CIs,
# anisotropic pixel & stage tilt helpers. Works with sem_psd_core.

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2


# ---------- Scale & tilt utilities ----------

def resolve_scale_xy(scale_um_per_px: Optional[float],
                     scale_um_per_px_x: Optional[float] = None,
                     scale_um_per_px_y: Optional[float] = None) -> Tuple[float, float]:
    """Return (um_per_px_x, um_per_px_y) with fallback to isotropic scale."""
    if scale_um_per_px_x and scale_um_per_px_y:
        return float(scale_um_per_px_x), float(scale_um_per_px_y)
    if scale_um_per_px and scale_um_per_px > 0:
        s = float(scale_um_per_px)
        return s, s
    raise ValueError("Valid scale (µm/px) is required: set either isotropic or X/Y.")

def effective_um_per_px_for_isotropic_kernels(umx: float, umy: float) -> float:
    """Geometric mean for isotropic ops (LoG sigma, morphology radii)."""
    return math.sqrt(umx * umy)

def tilt_corr_factor(tilt_deg: float) -> float:
    """Invert projection shrinkage by cos(tilt)."""
    if abs(tilt_deg) < 1e-9:
        return 1.0
    c = math.cos(math.radians(tilt_deg))
    return 1.0 / max(1e-6, c)


# ---------- Geometry helpers (µm domain) ----------

def contour_px_to_um(cnt_px: np.ndarray, umx: float, umy: float) -> np.ndarray:
    c = cnt_px.astype(np.float32).copy()
    c[:, 0, 0] *= umx
    c[:, 0, 1] *= umy
    return c

def perimeter_um(cnt_um: np.ndarray) -> float:
    p = 0.0
    pts = cnt_um[:, 0, :]
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        p += math.hypot(x1 - x0, y1 - y0)
    return float(p)

def feret_diameters_um(cnt_um: np.ndarray, step_deg: int = 2) -> Tuple[float, float]:
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
        width = pr[:, 0].max() - pr[:, 0].min()
        ferets.append(width)
    ferets = np.asarray(ferets)
    return float(ferets.min()), float(ferets.max())

def ellipse_axes_um(cnt_um: np.ndarray) -> Tuple[float, float]:
    pts = cnt_um.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] >= 5:
        try:
            (xc, yc), (MA, ma), _ = cv2.fitEllipse(pts)
            a = max(MA, ma); b = min(MA, ma)
            return float(b), float(a)
        except Exception:
            pass
    area = float(cv2.contourArea(cnt_um.astype(np.float32)))
    if area <= 0:
        return 0.0, 0.0
    d_eq = 2.0 * math.sqrt(area / math.pi)
    return float(d_eq), float(d_eq)


# ---------- Extended metrics ----------

def extended_metrics(cnt_px: np.ndarray,
                     umx: float,
                     umy: float,
                     tilt_deg: float = 0.0) -> Dict[str, float]:
    """
    Compute extended metrics in µm-domain (anisotropic pixel + optional tilt):
      area_um2, perimeter_um, d_eq_area_um, d_eq_perim_um,
      feret_min_um, feret_max_um, aspect_ratio,
      ellipse_minor_um, ellipse_major_um, roundness, solidity.
    """
    cnt_um = contour_px_to_um(cnt_px, umx, umy)
    area = float(cv2.contourArea(cnt_um.astype(np.float32)))
    if area <= 0:
        return {k: 0.0 for k in [
            "area_um2","perimeter_um","d_eq_area_um","d_eq_perim_um",
            "feret_min_um","feret_max_um","aspect_ratio",
            "ellipse_minor_um","ellipse_major_um","roundness","solidity"
        ]}

    perim = perimeter_um(cnt_um)
    d_eq_area = 2.0 * math.sqrt(area / math.pi)
    d_eq_perim = perim / math.pi

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


# ---------- ISO 9276 weighted PSD ----------

def iso9276_weighted_means(d_um: np.ndarray) -> Dict[str, float]:
    """
    Return ISO-style weighted/number stats:
      D10, D50, D90 (number), D32 (Sauter = Σd³/Σd²), D43 (De Brouckere = Σd⁴/Σd³),
      mean, std, min, max, particles.
    """
    out = {
        "D10": 0.0, "D50": 0.0, "D90": 0.0,
        "D32": 0.0, "D43": 0.0,
        "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
        "particles": int(d_um.size),
    }
    if d_um.size == 0:
        return out
    d = d_um.astype(np.float64)
    out["D10"] = float(np.percentile(d, 10))
    out["D50"] = float(np.percentile(d, 50))
    out["D90"] = float(np.percentile(d, 90))
    out["mean"] = float(np.mean(d))
    out["std"] = float(np.std(d))
    out["min"] = float(np.min(d))
    out["max"] = float(np.max(d))
    d2 = d * d
    d3 = d2 * d
    d4 = d3 * d
    s2 = d2.sum(); s3 = d3.sum(); s4 = d4.sum()
    out["D32"] = float(s3 / (s2 if s2 != 0 else 1.0))
    out["D43"] = float(s4 / (s3 if s3 != 0 else 1.0))
    return out


# ---------- Bootstrap confidence intervals ----------

def bootstrap_ci_percentile(x: np.ndarray, q=50, n=10000, alpha=0.05, seed=0) -> Tuple[float, float]:
    if x.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n, x.size))
    qs = np.percentile(x[idx], q, axis=1)
    lo, hi = np.percentile(qs, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)

def bootstrap_ci_mean(x: np.ndarray, n=10000, alpha=0.05, seed=0) -> Tuple[float, float]:
    if x.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n, x.size))
    means = np.mean(x[idx], axis=1)
    lo, hi = np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)


# ---------- Results enrichment ----------

@dataclass
class ParticleRow:
    idx: int
    d_um: float
    circ: float
    area_um2: float
    perimeter_um: float
    d_eq_area_um: float
    d_eq_perim_um: float
    feret_min_um: float
    feret_max_um: float
    aspect_ratio: float
    ellipse_minor_um: float
    ellipse_major_um: float
    roundness: float
    solidity: float

def enrich_results(results: List[Tuple[np.ndarray, float, float]],
                   umx: float, umy: float, tilt_deg: float = 0.0) -> List[ParticleRow]:
    rows: List[ParticleRow] = []
    for i, (cnt, d_um, circ) in enumerate(results):
        m = extended_metrics(cnt, umx, umy, tilt_deg)
        rows.append(ParticleRow(
            idx=i, d_um=float(d_um), circ=float(circ),
            area_um2=m["area_um2"], perimeter_um=m["perimeter_um"],
            d_eq_area_um=m["d_eq_area_um"], d_eq_perim_um=m["d_eq_perim_um"],
            feret_min_um=m["feret_min_um"], feret_max_um=m["feret_max_um"],
            aspect_ratio=m["aspect_ratio"],
            ellipse_minor_um=m["ellipse_minor_um"], ellipse_major_um=m["ellipse_major_um"],
            roundness=m["roundness"], solidity=m["solidity"],
        ))
    return rows


# ---------- Extended CSV ----------

def write_csv_extended(path: str,
                       particles: List[ParticleRow],
                       unit: str = "µm") -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "idx",
            f"d_{unit}",
            "circularity",
            f"area_{unit}^2",
            f"perimeter_{unit}",
            f"d_eq_area_{unit}",
            f"d_eq_perim_{unit}",
            f"feret_min_{unit}",
            f"feret_max_{unit}",
            "aspect_ratio",
            f"ellipse_minor_{unit}",
            f"ellipse_major_{unit}",
            "roundness",
            "solidity",
        ])
        for r in particles:
            w.writerow([
                r.idx,
                r.d_um,
                r.circ,
                r.area_um2,
                r.perimeter_um,
                r.d_eq_area_um,
                r.d_eq_perim_um,
                r.feret_min_um,
                r.feret_max_um,
                r.aspect_ratio,
                r.ellipse_minor_um,
                r.ellipse_major_um,
                r.roundness,
                r.solidity,
            ])
