"""
Morphology utilities: open/close, hole filling, watershed splitting,
and simple component counting with diameter-based gates.
"""

from __future__ import annotations
import math
import cv2
import numpy as np


def morph_open(bw: np.ndarray, open_um: float, um_per_px: float) -> np.ndarray:
    """Binary opening with an elliptical kernel sized by open_um (µm)."""
    if open_um <= 0:
        return bw
    r = max(1, int(open_um / um_per_px))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker, iterations=1)


def morph_close(bw: np.ndarray, closing_um: float, um_per_px: float) -> np.ndarray:
    """Binary closing with an elliptical kernel sized by closing_um (µm)."""
    if closing_um <= 0:
        return bw
    r = max(1, int(closing_um / um_per_px))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker, iterations=1)


def fill_small_holes(mask: np.ndarray, max_frac: float) -> np.ndarray:
    """
    Fill holes whose area is <= max_frac of their parent component area.

    Args:
        mask: binary image (nonzero = foreground).
        max_frac: maximum hole/parent area ratio to fill.
    """
    filled = mask.copy()
    contours, hier = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return filled
    hier = hier[0]
    for cnt, h in zip(contours, hier):
        if h[3] != -1:  # only top-level components
            continue
        area = cv2.contourArea(cnt)
        child = h[2]
        while child != -1:
            hole_cnt = contours[child]
            hole_area = cv2.contourArea(hole_cnt)
            if area > 0 and (hole_area / area) <= max_frac:
                cv2.drawContours(filled, [hole_cnt], -1, 255, thickness=-1)
            child = hier[child][0]
    return filled


def split_touching_watershed(
    bw: np.ndarray,
    um_per_px: float,
    min_neck_um: float = 0.12,
    min_seg_d_um: float = 0.20,
    fg_rel: float = 0.40,
    bg_dilate_iters: int = 2,
) -> np.ndarray:
    """
    Split touching objects using marker-based watershed.

    Steps:
      1) distance transform -> sure foreground (dist > fg_rel*max)
      2) dilate(object) -> sure background
      3) markers from sure-fg; unknown = bg - fg
      4) watershed over a gradient image
      5) keep segments with equivalent diameter >= min_seg_d_um
    """
    obj = (bw > 0).astype(np.uint8)
    if obj.max() == 0:
        return bw

    # Kernel sized from neck scale
    neck_px = max(1, int(round(min_neck_um / um_per_px)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * neck_px + 1, 2 * neck_px + 1))

    # Distance field (slightly smoothed)
    dist = cv2.distanceTransform(obj, cv2.DIST_L2, 5).astype(np.float32)
    if neck_px >= 2:
        dist = cv2.GaussianBlur(dist, (0, 0), sigmaX=0.5 * neck_px)

    mxd = float(dist.max())
    if mxd <= 0:
        return bw

    # Sure foreground from relative distance threshold
    fg = (dist > (fg_rel * mxd)).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # If still merged, relax threshold once
    if cv2.connectedComponents(fg)[0] < 3:
        fg2 = (dist > (0.30 * mxd)).astype(np.uint8) * 255
        fg2 = cv2.morphologyEx(fg2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        if cv2.connectedComponents(fg2)[0] >= 3:
            fg = fg2

    # Sure background from dilated object
    bg = cv2.dilate(obj * 255, ker, iterations=max(1, bg_dilate_iters))

    # Unknown region and markers
    unknown = cv2.subtract(bg, fg)
    _, markers = cv2.connectedComponents((fg > 0).astype(np.uint8))
    markers = markers.astype(np.int32) + 1  # background -> 1
    markers[unknown > 0] = 0

    # Watershed on a small gradient image (helps align boundaries)
    grad = cv2.morphologyEx(obj * 255, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    img3 = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img3, markers)  # -1: watershed lines

    # Collect valid segments inside the original object mask
    seg = np.zeros_like(obj, np.uint8)
    for lab in range(2, markers.max() + 1):
        m = (markers == lab) & (obj > 0)
        if not m.any():
            continue
        A = float(m.sum())
        d_px = 2.0 * math.sqrt(A / math.pi)
        d_um = d_px * um_per_px
        if d_um >= min_seg_d_um:
            seg[m] = 255

    return seg


def count_reasonable_components(bw: np.ndarray, um_per_px: float, dmin_um: float | None, dmax_um: float | None) -> int:
    """
    Count components whose equivalent-circle area falls within relaxed
    diameter bounds (0.6×dmin .. 1.5×dmax).
    """
    num, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)

    def area_from_d_um(d_um: float | None) -> float | None:
        if d_um is None:
            return None
        r_px = max(1.0, (d_um / 2.0) / um_per_px)
        return math.pi * r_px * r_px

    amin = area_from_d_um(dmin_um * 0.6 if dmin_um else None)
    amax = area_from_d_um(dmax_um * 1.5 if dmax_um else None)

    k = 0
    for i in range(1, num):
        a = st[i, cv2.CC_STAT_AREA]
        if amin is not None and a < amin:
            continue
        if amax is not None and a > amax:
            continue
        k += 1
    return k
