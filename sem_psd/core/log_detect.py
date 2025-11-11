"""
LoG-based blob detection utilities.

Multi-scale Laplacian-of-Gaussian detection with σ² normalization,
non-maximum suppression, minimum separation, ROI masking, and a
local relative-contrast check against the leveled image.

Supports cooperative cancellation via an optional `cancel_cb`.
"""

from __future__ import annotations
import math
from typing import Callable
import cv2
import numpy as np


def sigma_from_d_um(d_um: float, um_per_px: float) -> float:
    """Approximate LoG σ for a target diameter d (µm)."""
    return max(0.6, (d_um / (2.0 * math.sqrt(2.0))) / um_per_px)


def log_response(img_float: np.ndarray, sigma: float) -> np.ndarray:
    """LoG response with σ² normalization (bright blobs on dark)."""
    blur = cv2.GaussianBlur(img_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    return (sigma * sigma) * (-lap)


def nms2d(resp: np.ndarray, radius_px: int) -> np.ndarray:
    """2D non-maximum suppression via dilation comparison."""
    radius_px = max(1, int(radius_px))
    k = 2 * radius_px + 1
    dil = cv2.dilate(resp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return resp >= dil


def detect_blobs_log(
    gray: np.ndarray,
    lev_img: np.ndarray,
    um_per_px: float,
    dmin_um: float,
    dmax_um: float,
    threshold_rel: float = 0.03,
    minsep_um: float = 0.12,
    roi_mask: np.ndarray | None = None,
    min_rel_contrast: float = 0.0,
    cancel_cb: Callable[[], bool] | None = None,   # ← NEW
) -> list[tuple[np.ndarray, float, float]]:
    """
    Detect approximately circular bright blobs using multi-scale LoG.

    Returns a list of (contour_px, diameter_um, score) where the contour is an ellipse polygon.

    Args:
        gray: Grayscale input image (uint8).
        lev_img: Leveled/contrast-normalized image (uint8) for contrast check.
        um_per_px: Micrometers per pixel.
        dmin_um, dmax_um: Expected diameter range (µm).
        threshold_rel: Relative threshold vs. global max LoG response.
        minsep_um: Minimum center-to-center separation (µm).
        roi_mask: Optional mask (255 inside ROI, 0 outside).
        min_rel_contrast: Minimum (mean_in - mean_bg)/255 contrast.
        cancel_cb: Optional function returning True when computation should stop.
    """
    img = gray.astype(np.float32)
    if roi_mask is None:
        roi_mask = np.ones_like(gray, np.uint8) * 255

    # Scale space
    sig_lo = sigma_from_d_um(max(0.01, dmin_um), um_per_px)
    sig_hi = sigma_from_d_um(max(dmax_um, dmin_um + 1e-6), um_per_px)
    n_scales = 13
    sigmas = np.exp(np.linspace(np.log(sig_lo), np.log(sig_hi), n_scales))

    H, W = gray.shape
    resp_max = np.zeros((H, W), np.float32)
    arg_sigma = np.zeros((H, W), np.float32)

    for s in sigmas:
        if cancel_cb and cancel_cb():
            return []
        R = log_response(img, float(s))
        R = cv2.bitwise_and(R, R, mask=roi_mask)
        better = R > resp_max
        resp_max[better] = R[better]
        arg_sigma[better] = float(s)

    mx = float(resp_max.max()) if resp_max.size else 0.0
    if mx <= 0:
        return []

    # Candidate peaks
    thr = threshold_rel * mx
    candidate = (resp_max >= thr) & (roi_mask.astype(bool))
    peaks_mask = candidate & nms2d(resp_max, radius_px=1)
    peaks = np.argwhere(peaks_mask)
    if cancel_cb and cancel_cb():
        return []
    vals = resp_max[peaks_mask]
    order = np.argsort(-vals)
    peaks = peaks[order]

    # Enforce minimum separation, contrast, and build contours
    minsep_px = max(1.0, minsep_um / um_per_px)
    occupied = np.zeros((H, W), np.uint8)
    rad_occ = int(round(minsep_px))
    kept: list[tuple[np.ndarray, float, float]] = []

    for y, x in peaks:
        if cancel_cb and cancel_cb():
            return kept
        if occupied[y, x]:
            continue
        s = float(arg_sigma[y, x])
        if s <= 0:
            continue

        d_px = 2.0 * math.sqrt(2.0) * s
        d_um = d_px * um_per_px

        # Local relative contrast check
        ring_th = max(3, int(0.15 * d_px))
        mask_obj = np.zeros_like(gray, np.uint8)
        cv2.circle(mask_obj, (int(x), int(y)), max(1, int(round(d_px / 2))), 255, thickness=-1)
        ring = np.zeros_like(gray, np.uint8)
        cv2.circle(ring, (int(x), int(y)), max(1, int(round(d_px / 2)) + ring_th), 255, thickness=ring_th)
        ring = cv2.subtract(ring, mask_obj)
        vals_in = lev_img[mask_obj == 255]
        vals_bg = lev_img[ring == 255]
        mean_in = float(vals_in.mean()) if vals_in.size else 0.0
        mean_bg = float(vals_bg.mean()) if vals_bg.size else 0.0
        rel_contrast = (mean_in - mean_bg) / 255.0
        if rel_contrast < min_rel_contrast:
            continue

        if rad_occ > 0:
            cv2.circle(occupied, (int(x), int(y)), rad_occ, 1, thickness=-1)

        # Elliptic contour approximation
        cnt = cv2.ellipse2Poly(
            (int(x), int(y)),
            (int(round(d_px / 2)), int(round(d_px / 2))),
            0, 0, 360, 6
        )
        cnt = cnt.reshape((-1, 1, 2)).astype(np.int32)
        kept.append((cnt, float(d_um), 1.0))

    return kept
