# sem_psd_core.py
# Non-GUI core: IO, preprocessing, detection, stats, params.

from __future__ import annotations
import math, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2
from PIL import Image


# ---------------- IO / metadata ----------------

def imread_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        pil = Image.open(path)
        if pil.mode not in ("L", "I;16", "I;16B", "I;16L"):
            pil = pil.convert("L")
        arr = np.array(pil)
        if arr.dtype == np.uint16:
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return arr.astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img

def dump_tiff_metadata_text(image_path: str) -> str:
    try:
        pil = Image.open(image_path)
    except Exception as e:
        return f"[ERROR opening TIFF: {e}]"
    out = []
    try:
        for tag, val in getattr(pil, "tag_v2", {}).items():
            if isinstance(val, bytes): s = val.decode(errors="ignore")
            elif isinstance(val, (list, tuple)):
                s = " ".join([v.decode(errors="ignore") if isinstance(v, bytes) else str(v) for v in val])
            else: s = str(val)
            out.append(f"[{tag}] {s}")
    except Exception:
        pass
    try:
        for k, v in (pil.info or {}).items():
            if isinstance(v, bytes): v = v.decode(errors="ignore")
            out.append(f"[{k}] {v}")
    except Exception:
        pass
    return "\n".join(out)

def parse_um_per_px_from_text(txt: str) -> float | None:
    if not txt: return None
    m = re.search(r"PixelWidth\s*=\s*([0-9eE\.\-\+]+)", txt)
    if m:
        try:
            px_m = float(m.group(1))
            if px_m > 0: return px_m * 1e6
        except Exception: pass
    m_hfw = re.search(r"(HorFieldsize|HFW)\s*=\s*([0-9eE\.\-\+]+)", txt)
    m_rx = re.search(r"(ResolutionX|Resolutionx)\s*=\s*([0-9]+)", txt)
    if m_hfw and m_rx:
        try:
            hfw_m = float(m_hfw.group(2)); resx = int(m_rx.group(2))
            if hfw_m > 0 and resx > 0: return (hfw_m * 1e6) / float(resx)
        except Exception: pass
    return None

def scale_from_metadata(image_path: str) -> float | None:
    return parse_um_per_px_from_text(dump_tiff_metadata_text(image_path))


# ---------------- Basic image ops ----------------

def make_roi_mask(shape, top=0.02, bottom=0.22, left=0.0, right=0.0):
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    t = int(h * top); b = int(h * (1.0 - bottom))
    l = int(w * left); r = int(w * (1.0 - right))
    mask[max(0, t): max(0, b), max(0, l): max(0, r)] = 255
    return mask

def preprocess(gray: np.ndarray, clahe_clip=2.0, tophat_um=0.0, um_per_px=0.05, level_strength=0.3) -> np.ndarray:
    lev = gray.copy()
    if level_strength and level_strength > 0:
        bg = cv2.GaussianBlur(gray, (0, 0), 25)
        lev = cv2.addWeighted(gray, 1.0 + level_strength, bg, -level_strength, 0)
    if tophat_um and tophat_um > 0:
        rpx = max(1, int(tophat_um / um_per_px))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rpx + 1, 2 * rpx + 1))
        lev = cv2.morphologyEx(lev, cv2.MORPH_TOPHAT, ker)
    lev = cv2.medianBlur(lev, 3)
    if clahe_clip and clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
        lev = clahe.apply(lev)
    return lev

def threshold_pair(lev, roi_mask, method="otsu", block_size=31, C=-10):
    if method.lower() == "adaptive":
        if block_size % 2 == 0: block_size += 1
        bwB = cv2.adaptiveThreshold(lev, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
    else:
        _, bwB = cv2.threshold(lev, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bwB = cv2.bitwise_and(bwB, bwB, mask=roi_mask)
    bwD = 255 - bwB
    return bwB, bwD

def morph_open(bw, open_um, um_per_px):
    if open_um <= 0: return bw
    r = max(1, int(open_um / um_per_px))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker, iterations=1)

def morph_close(bw, closing_um, um_per_px):
    if closing_um <= 0: return bw
    rad_px = max(1, int(closing_um / um_per_px))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad_px + 1, 2 * rad_px + 1))
    return cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker, iterations=1)

def fill_small_holes(mask: np.ndarray, max_frac: float) -> np.ndarray:
    filled = mask.copy()
    contours, hier = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None: return filled
    hier = hier[0]
    for i, (cnt, h) in enumerate(zip(contours, hier)):
        if h[3] != -1: continue
        area = cv2.contourArea(cnt); child = h[2]
        while child != -1:
            hole_cnt = contours[child]; hole_area = cv2.contourArea(hole_cnt)
            if area > 0 and (hole_area / area) <= max_frac:
                cv2.drawContours(filled, [hole_cnt], -1, 255, thickness=-1)
            child = hier[child][0]
    return filled

def split_touching_watershed(bw, um_per_px, min_neck_um=0.12, min_seg_d_um=0.20):
    obj = (bw > 0).astype(np.uint8)
    dist = cv2.distanceTransform(obj, cv2.DIST_L2, 3)
    neck_px = max(1, int(min_neck_um / um_per_px))
    dist_smooth = cv2.GaussianBlur(dist, (0, 0), sigmaX=0.5 * neck_px)
    dn = cv2.normalize(dist_smooth, None, 0, 1.0, cv2.NORM_MINMAX)
    k = max(3, 2 * neck_px + 1)
    med = cv2.medianBlur((dn * 255).astype(np.uint8), k)
    seeds = ((dn * 255).astype(np.uint8) > (med + 5)).astype(np.uint8)
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    num, markers = cv2.connectedComponents(seeds)
    if num < 2: return bw
    markers = markers.astype(np.int32)
    mask3 = cv2.cvtColor((obj * 255), cv2.COLOR_GRAY2BGR)
    cv2.watershed(mask3, markers)
    seg = np.zeros_like(obj, np.uint8)
    for lab in range(1, markers.max() + 1):
        m = (markers == lab) & (obj.astype(bool))
        if not m.any(): continue
        A = float(m.sum())
        d_px = 2.0 * math.sqrt(A / math.pi)
        d_um = d_px * um_per_px
        if d_um >= min_seg_d_um:
            seg[m] = 255
    return seg

def count_reasonable_components(bw, um_per_px, dmin_um, dmax_um):
    num, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    def area_from_d_um(d_um):
        if d_um is None: return None
        r_px = max(1.0, (d_um / 2.0) / um_per_px)
        return math.pi * r_px * r_px
    amin = area_from_d_um(dmin_um * 0.6 if dmin_um else None)
    amax = area_from_d_um(dmax_um * 1.5 if dmax_um else None)
    k = 0
    for i in range(1, num):
        a = st[i, cv2.CC_STAT_AREA]
        if amin is not None and a < amin: continue
        if amax is not None and a > amax: continue
        k += 1
    return k

def measure_components(bw, min_d_um, max_d_um, min_circ, um_per_px, lev_img, min_rel_contrast):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    results = []
    for i in range(1, num):
        a = stats[i, cv2.CC_STAT_AREA]
        def area_from_d_um(d_um):
            if d_um is None: return None
            r_px = max(0.5, (d_um / 2.0) / um_per_px)
            return math.pi * r_px * r_px
        area_min = area_from_d_um(min_d_um) if min_d_um else None
        area_max = area_from_d_um(max_d_um) if max_d_um else None
        if area_min is not None and a < area_min: continue
        if area_max is not None and a > area_max: continue

        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        roi = (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = max(cnts, key=cv2.contourArea); cnt[:, 0, 0] += x; cnt[:, 0, 1] += y

        A = float(cv2.contourArea(cnt))
        if A < 1.0: continue
        P = float(cv2.arcLength(cnt, True))
        circ = (4.0 * math.pi * A) / (P * P + 1e-9)
        if circ < min_circ: continue

        mask_obj = np.zeros_like(bw); cv2.drawContours(mask_obj, [cnt], -1, 255, thickness=-1)
        ring = np.zeros_like(bw); d_px_est = 2.0 * math.sqrt(A / math.pi)
        ring_thick = max(3, int(0.15 * d_px_est))
        cv2.drawContours(ring, [cnt], -1, 255, thickness=ring_thick); ring = cv2.subtract(ring, mask_obj)

        vals_in = lev_img[mask_obj == 255]; vals_bg = lev_img[ring == 255]
        if vals_bg.size < 30:
            cv2.drawContours(ring, [cnt], -1, 255, thickness=ring_thick * 2)
            ring = cv2.subtract(ring, mask_obj); vals_bg = lev_img[ring == 255]
        mean_in = float(vals_in.mean()) if vals_in.size else 0.0
        mean_bg = float(vals_bg.mean()) if vals_bg.size else 0.0
        rel_contrast = (mean_in - mean_bg) / 255.0
        if rel_contrast < min_rel_contrast: continue

        d_px = 2.0 * math.sqrt(A / math.pi); d_um = d_px * um_per_px
        results.append((cnt, d_um, circ))
    return results

def stats_from_diams(d_um: np.ndarray) -> Dict[str, float | int]:
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


# ---------------- LoG helpers ----------------

def sigma_from_d_um(d_um: float, um_per_px: float) -> float:
    return max(0.6, (d_um / (2.0 * math.sqrt(2.0))) / um_per_px)

def log_response(img_float: np.ndarray, sigma: float) -> np.ndarray:
    blur = cv2.GaussianBlur(img_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    return (sigma * sigma) * (-lap)

def nms2d(resp: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px < 1: radius_px = 1
    k = 2 * radius_px + 1
    dil = cv2.dilate(resp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return (resp >= dil)

def detect_blobs_log(gray: np.ndarray, lev_img: np.ndarray, um_per_px: float,
                     dmin_um: float, dmax_um: float, threshold_rel: float = 0.03,
                     minsep_um: float = 0.12, roi_mask: np.ndarray | None = None,
                     min_rel_contrast: float = 0.0) -> list[tuple[np.ndarray, float, float]]:
    img = gray.astype(np.float32)
    if roi_mask is None:
        roi_mask = np.ones_like(gray, np.uint8) * 255
    sig_lo = sigma_from_d_um(max(0.01, dmin_um), um_per_px)
    sig_hi = sigma_from_d_um(max(dmax_um, dmin_um + 1e-6), um_per_px)
    n_scales = 13
    sigmas = np.exp(np.linspace(np.log(sig_lo), np.log(sig_hi), n_scales))

    H, W = gray.shape
    resp_max = np.zeros((H, W), np.float32)
    arg_sigma = np.zeros((H, W), np.float32)

    for s in sigmas:
        R = log_response(img, float(s))
        if roi_mask is not None:
            R = cv2.bitwise_and(R, R, mask=roi_mask)
        better = R > resp_max
        resp_max[better] = R[better]
        arg_sigma[better] = float(s)

    mx = float(resp_max.max()) if resp_max.size else 0.0
    if mx <= 0: return []

    thr = threshold_rel * mx
    candidate = (resp_max >= thr) & (roi_mask.astype(bool))
    nms = nms2d(resp_max, radius_px=1)
    peaks = np.argwhere(candidate & nms)

    vals = resp_max[candidate & nms]
    order = np.argsort(-vals)
    peaks = peaks[order]

    minsep_px = max(1.0, minsep_um / um_per_px)
    kept = []
    occupied = np.zeros((H, W), np.uint8)
    rad_occ = int(round(minsep_px))

    for y, x in peaks:
        if occupied[y, x]: continue
        s = float(arg_sigma[y, x])
        if s <= 0: continue
        d_px = 2.0 * math.sqrt(2.0) * s
        d_um = d_px * um_per_px

        ring_th = max(3, int(0.15 * d_px))
        mask_obj = np.zeros_like(gray, np.uint8); cv2.circle(mask_obj, (int(x), int(y)), max(1, int(round(d_px/2))), 255, thickness=-1)
        ring = np.zeros_like(gray, np.uint8); cv2.circle(ring, (int(x), int(y)), max(1, int(round(d_px/2)) + ring_th), 255, thickness=ring_th)
        ring = cv2.subtract(ring, mask_obj)
        vals_in = lev_img[mask_obj == 255]; vals_bg = lev_img[ring == 255]
        mean_in = float(vals_in.mean()) if vals_in.size else 0.0
        mean_bg = float(vals_bg.mean()) if vals_bg.size else 0.0
        rel_contrast = (mean_in - mean_bg) / 255.0
        if rel_contrast < min_rel_contrast: continue

        if rad_occ > 0: cv2.circle(occupied, (int(x), int(y)), rad_occ, 1, thickness=-1)

        cnt = cv2.ellipse2Poly((int(x), int(y)), (int(round(d_px/2)), int(round(d_px/2))), 0, 0, 360, 6)
        cnt = cnt.reshape((-1, 1, 2)).astype(np.int32)
        kept.append((cnt, float(d_um), 1.0))
    return kept


# ---------------- Parameters ----------------

@dataclass
class Params:
    scale_um_per_px: float | None
    exclude_top: float; exclude_bottom: float; exclude_left: float; exclude_right: float
    min_d_um: float; max_d_um: float
    closing_um: float; open_um: float
    min_circ: float
    thr_method: str; block_size: int; block_C: int
    clahe_clip: float; min_rel_contrast: float; tophat_um: float; level_strength: float
    split_touching: bool; min_neck_um: float; min_seg_d_um: float
    analysis_mode: str
    log_threshold_rel: float
    log_minsep_um: float
