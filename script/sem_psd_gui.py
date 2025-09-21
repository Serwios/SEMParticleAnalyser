# sem_psd_gui.py
# SEM particle-size GUI (PySide6)

from __future__ import annotations
import sys, math, re, csv, textwrap
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image

from PySide6.QtCore import Qt, Signal, QObject, QThread, QEvent
from PySide6.QtGui import QPixmap, QAction, QTransform, QPainter, QImage, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QWidget, QFileDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFormLayout, QDoubleSpinBox, QSpinBox, QTabWidget, QComboBox, QCheckBox,
    QMessageBox, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QProgressBar, QGroupBox,
    QProxyStyle, QStyle, QToolButton, QMenu, QStackedWidget
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import QStandardPaths

# ---------------- Utilities ----------------

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

def stats_from_diams(d_um: np.ndarray) -> dict:
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

def np_to_qpix(img: np.ndarray) -> QPixmap:
    if img.ndim == 2:
        qimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        qimg = img
    h, w, _ = qimg.shape
    qimg = cv2.cvtColor(qimg, cv2.COLOR_BGR2RGB)
    qim = QImage(qimg.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qim)

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

# ---------------- Helper widgets ----------------

class ImageView(QGraphicsView):
    sig_clicked = Signal(int, int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._item = QGraphicsPixmapItem()
        self.scene().addItem(self._item)
        self._img_w = 0
        self._img_h = 0
        self._auto_fit = True

        self.setRenderHints(self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFrameShape(QGraphicsView.NoFrame)

    def set_image(self, np_img: np.ndarray | QPixmap):
        pm = np_img if isinstance(np_img, QPixmap) else np_to_qpix(np_img)
        self._item.setPixmap(pm)
        self._img_w = pm.width()
        self._img_h = pm.height()
        self._auto_fit = True
        self.fit_to_view()

    def fit_to_view(self):
        if self._img_w == 0 or self.width() < 5 or self.height() < 5:
            return
        self.setTransform(QTransform())
        r = self._item.boundingRect()
        m = 2
        self.fitInView(r.adjusted(m, m, -m, -m), Qt.KeepAspectRatio)

    def resizeEvent(self, e):
        if self._auto_fit and self._img_w:
            self.fit_to_view()
        super().resizeEvent(e)

    def wheelEvent(self, e):
        if self._img_w == 0:
            return
        self._auto_fit = False
        factor = 1.15 if e.angleDelta().y() > 0 else 1/1.15
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, e):
        self._auto_fit = True
        self.fit_to_view()
        super().mouseDoubleClickEvent(e)

    def keyPressEvent(self, e):
        if e.modifiers() & Qt.ControlModifier:
            if e.key() == Qt.Key_1:
                self._auto_fit = False
                self.setTransform(QTransform())
                e.accept(); return
            if e.key() == Qt.Key_0:
                self._auto_fit = True
                self.fit_to_view()
                e.accept(); return
            if e.key() == Qt.Key_Plus:
                self._auto_fit = False; self.scale(1.15, 1.15); e.accept(); return
            if e.key() == Qt.Key_Minus:
                self._auto_fit = False; self.scale(1/1.15, 1/1.15); e.accept(); return
        super().keyPressEvent(e)

    def mousePressEvent(self, e):
        if self._img_w:
            p = self.mapToScene(e.pos())
            x = int(round(p.x()))
            y = int(round(p.y()))
            x = max(0, min(self._img_w - 1, x))
            y = max(0, min(self._img_h - 1, y))
            self.sig_clicked.emit(x, y)
        super().mousePressEvent(e)

class ToolTipDelayStyle(QProxyStyle):
    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.SH_ToolTip_WakeUpDelay:
            return 5000
        if hint == QStyle.SH_ToolTip_FallAsleepDelay:
            return 30000
        return super().styleHint(hint, option, widget, returnData)

class MplWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.addWidget(self.canvas)

    def plot_hist(self, d_vals: np.ndarray, st: dict, title: str, unit: str):
        self.fig.clear(); ax = self.fig.add_subplot(111)
        if d_vals.size:
            ax.hist(d_vals, bins=40)
            for name in ("D10","D50","D90"):
                ax.axvline(st[name], linestyle="--", label=f"{name}={st[name]:.2f} {unit}")
            ax.set_xlabel(f"Particle diameter ({unit})"); ax.set_ylabel("Count"); ax.set_title(title)
            ax.grid(True); ax.legend()
        self.canvas.draw()

    def plot_cum(self, d_vals: np.ndarray, st: dict, unit: str):
        self.fig.clear(); ax = self.fig.add_subplot(111)
        if d_vals.size:
            s = np.sort(d_vals); cum = np.arange(1, s.size+1)/s.size*100.0
            ax.plot(s, cum)
            for name in ("D10","D50","D90"):
                ax.axvline(st[name], linestyle="--", label=f"{name}={st[name]:.2f} {unit}")
            ax.set_xlabel(f"Particle diameter ({unit})"); ax.set_ylabel("Cumulative %"); ax.set_title("Cumulative PSD")
            ax.grid(True); ax.legend()
        self.canvas.draw()

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

# ---------------- Worker ----------------

class Worker(QObject):
    finished = Signal(object)
    failed = Signal(object)
    def __init__(self, gray: np.ndarray, params: Params, image_path: Path):
        super().__init__(); self.gray = gray; self.P = params; self.image_path = image_path

    def run(self):
        try:
            roi_mask = make_roi_mask(self.gray.shape, self.P.exclude_top, self.P.exclude_bottom, self.P.exclude_left, self.P.exclude_right)
            lev = preprocess(self.gray, self.P.clahe_clip, self.P.tophat_um, self.P.scale_um_per_px, self.P.level_strength)

            if self.P.analysis_mode == "log":
                results = detect_blobs_log(self.gray, lev, self.P.scale_um_per_px,
                                           self.P.min_d_um, self.P.max_d_um,
                                           self.P.log_threshold_rel, self.P.log_minsep_um,
                                           roi_mask, self.P.min_rel_contrast)
                self.finished.emit({"lev": lev, "thr_raw": None, "thr_proc": None, "results": results}); return

            bwB, bwD = threshold_pair(lev, roi_mask, method=self.P.thr_method, block_size=self.P.block_size, C=self.P.block_C)
            kB = count_reasonable_components(bwB, self.P.scale_um_per_px, self.P.min_d_um, self.P.max_d_um)
            kD = count_reasonable_components(bwD, self.P.scale_um_per_px, self.P.min_d_um, self.P.max_d_um)
            bw = bwB if kB >= kD else bwD; thr_raw = bw.copy()
            bw = morph_close(bw, self.P.closing_um, self.P.scale_um_per_px)
            bw = morph_open(bw, self.P.open_um, self.P.scale_um_per_px)
            bw = fill_small_holes(bw, 0.6)
            if self.P.split_touching:
                bw = split_touching_watershed(bw, self.P.scale_um_per_px, self.P.min_neck_um, self.P.min_seg_d_um)
            results = measure_components(bw, self.P.min_d_um, self.P.max_d_um, self.P.min_circ, self.P.scale_um_per_px, lev, self.P.min_rel_contrast)
            self.finished.emit({"lev": lev, "thr_raw": thr_raw, "thr_proc": bw, "results": results})
        except Exception as e:
            self.failed.emit(str(e))

# ---------------- MainWindow ----------------

class MainWindow(QWidget):
    # Auto-switch to nm if scale is finer than this (µm/px)
    auto_nm_threshold_um_per_px = 0.10  # <0.10 µm/px → show nm

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SEM PSD (Particles Analysis)")
        self.resize(1400, 900)

        # State
        self.image_path: Path | None = None
        self.gray: np.ndarray | None = None
        self.overlay_img: np.ndarray | None = None
        self.lev: np.ndarray | None = None
        self.thr_raw: np.ndarray | None = None
        self.thr_proc: np.ndarray | None = None
        self.diams_um: np.ndarray = np.array([], float)
        self.stats: dict = {}
        self.um_per_px_from_meta: float | None = None
        self.results: list[tuple[np.ndarray, float, float]] = []
        self.excluded_idx: set[int] = set()
        self.remove_mode: bool = False
        self._thread: QThread | None = None
        self._worker: Worker | None = None
        self._open_in_progress: bool = False

        # Hover/highlight state for table/overlay
        self.row_to_result: list[int] = []   # row index -> result index
        self.hover_idx: int | None = None    # highlighted result index

        self.build_ui()

    # ---------- UI ----------
    def build_ui(self):
        topbar = QHBoxLayout()

        self.btn_open = QToolButton()
        self.btn_open.setText("Open image…")
        self.btn_open.setToolTip("Open a SEM image file or a built-in sample.")
        self.btn_open.setPopupMode(QToolButton.MenuButtonPopup)
        self.btn_open.clicked.connect(self.open_image)

        menu = QMenu(self)
        act_open = menu.addAction("Open…")
        act_open.triggered.connect(self.open_image)
        smpl = menu.addMenu("Open test sample")
        smpl.addAction("blobs.tif", lambda: self.open_sample("blobs.tif"))
        smpl.addAction("granular.tif", lambda: self.open_sample("granular.tif"))
        self.btn_open.setMenu(menu)

        self.meta_label = QLabel("Scale: —"); self.meta_label.setStyleSheet("color:#666")
        self.progress = QProgressBar(); self.progress.setRange(0,0); self.progress.setVisible(False); self.progress.setFixedHeight(8); self.progress.setTextVisible(False)

        topbar.addWidget(self.btn_open); topbar.addWidget(self.meta_label); topbar.addStretch(1); topbar.addWidget(self.progress)

        # Left panel (controls)
        left = QWidget(); form = QFormLayout(left)

        self.sb_scale = QDoubleSpinBox(); self.sb_scale.setRange(1e-6, 1000.0); self.sb_scale.setDecimals(6); self.sb_scale.setValue(0.0)
        self.sb_scale.setToolTip("Micrometers per pixel. If 0, will try to read from TIFF metadata.")
        self.sb_scale.valueChanged.connect(self.on_scale_changed)

        self.cb_autoscale = QPushButton("Read scale from metadata"); self.cb_autoscale.setToolTip("Attempt to read µm/px from TIFF tags (PixelWidth or HFW/ResolutionX).")
        self.cb_autoscale.clicked.connect(self.read_scale_meta)

        self.cb_mode = QComboBox(); self.cb_mode.addItems(["Contours (threshold)", "Nano (LoG)"])
        self.cb_mode.setToolTip("Choose algorithm: contour thresholding or multi-scale LoG.")
        self.cb_mode.currentIndexChanged.connect(self.update_param_visibility)
        self.btn_help = QPushButton("Help"); self.btn_help.setToolTip("Show quick reference."); self.btn_help.clicked.connect(self.show_help)

        mode_row = QWidget(); hl = QHBoxLayout(mode_row); hl.setContentsMargins(0,0,0,0); hl.addWidget(self.cb_mode); hl.addWidget(self.btn_help)
        form.addRow("Scale (µm/px)", self.sb_scale); form.addRow(" ", self.cb_autoscale); form.addRow("Analysis mode", mode_row)

        # --- Auto/Manual display units controls ---
        self.cb_auto_units = QCheckBox("Auto-detect display units from scale")
        self.cb_auto_units.setChecked(True)
        self.cb_auto_units.setToolTip("When enabled: display units (µm/nm) are chosen automatically from the scale (µm/px). Manual selection is disabled.")
        self.cb_auto_units.toggled.connect(self.on_auto_units_toggled)
        form.addRow(self.cb_auto_units)

        self.cb_units = QComboBox()
        self.cb_units.addItems(["µm", "nm"])
        self.cb_units.setToolTip("Display units for plots/table/stats/CSV (works if auto-detect is disabled).")
        self.cb_units.currentIndexChanged.connect(self.on_units_changed)
        self.cb_units.setEnabled(False)  # start locked because auto=on
        form.addRow("Display units", self.cb_units)

        # ROI
        self.sb_top = QDoubleSpinBox(); self.sb_bottom = QDoubleSpinBox(); self.sb_left = QDoubleSpinBox(); self.sb_right = QDoubleSpinBox()
        for s in (self.sb_top, self.sb_bottom, self.sb_left, self.sb_right):
            s.setRange(0,0.9); s.setSingleStep(0.01); s.setDecimals(3)
        self.sb_top.setValue(0.02); self.sb_bottom.setValue(0.22); self.sb_left.setValue(0.0); self.sb_right.setValue(0.0)
        self.sb_top.setToolTip("Exclude a top strip (fraction of image height).")
        self.sb_bottom.setToolTip("Exclude a bottom strip (fraction of image height).")
        self.sb_left.setToolTip("Exclude a left strip (fraction of image width).")
        self.sb_right.setToolTip("Exclude a right strip (fraction of image width).")
        row_roi = QWidget(); lroi = QFormLayout(row_roi); lroi.addRow("Top", self.sb_top); lroi.addRow("Bottom", self.sb_bottom); lroi.addRow("Left", self.sb_left); lroi.addRow("Right", self.sb_right)
        form.addRow(QLabel("ROI exclude ratios"), row_roi)

        # General
        self.sb_min_d = QDoubleSpinBox(); self.sb_max_d = QDoubleSpinBox(); self.sb_min_rc = QDoubleSpinBox()
        self.sb_min_d.setRange(0.0, 1000.0); self.sb_min_d.setDecimals(3); self.sb_min_d.setValue(0.01); self.sb_min_d.setToolTip("Minimum particle diameter (µm) after noise rejection.")
        self.sb_max_d.setRange(0.0, 10000.0); self.sb_max_d.setDecimals(3); self.sb_max_d.setValue(10.0); self.sb_max_d.setToolTip("Maximum particle diameter (µm).")
        self.sb_min_rc.setRange(0.0,1.0); self.sb_min_rc.setDecimals(2); self.sb_min_rc.setValue(0.15); self.sb_min_rc.setToolTip("Minimum relative contrast: (inside - ring)/255 in 0..1.")
        form.addRow("Min diameter (µm)", self.sb_min_d); form.addRow("Max diameter (µm)", self.sb_max_d); form.addRow("Min rel. contrast", self.sb_min_rc)

        # Preprocess
        self.sb_clahe = QDoubleSpinBox(); self.sb_tophat = QDoubleSpinBox(); self.sb_level = QDoubleSpinBox()
        self.sb_clahe.setRange(0.0,10.0); self.sb_clahe.setDecimals(2); self.sb_clahe.setValue(2.0); self.sb_clahe.setToolTip("CLAHE clip limit (0 disables).")
        self.sb_tophat.setRange(0.0,100.0); self.sb_tophat.setDecimals(3); self.sb_tophat.setValue(0.0); self.sb_tophat.setToolTip("Top-hat radius (µm) to suppress background.")
        self.sb_level.setRange(0.0,1.5); self.sb_level.setDecimals(2); self.sb_level.setValue(0.3); self.sb_level.setToolTip("Local contrast leveling strength (0..1.5).")
        form.addRow("CLAHE clip", self.sb_clahe); form.addRow("Top-hat radius (µm)", self.sb_tophat); form.addRow("Level strength", self.sb_level)

        # Threshold-only
        self.grp_thr = QGroupBox("Threshold-only")
        lay_thr = QFormLayout(self.grp_thr)
        self.sb_min_circ = QDoubleSpinBox(); self.sb_min_circ.setRange(0.0, 1.0); self.sb_min_circ.setDecimals(3); self.sb_min_circ.setValue(0.10); self.sb_min_circ.setToolTip("Minimum circularity: 4πA/P².")
        self.sb_closing = QDoubleSpinBox(); self.sb_closing.setRange(0.0, 100.0); self.sb_closing.setDecimals(3); self.sb_closing.setValue(0.12); self.sb_closing.setToolTip("Closing (µm) to close contour gaps.")
        self.sb_open = QDoubleSpinBox(); self.sb_open.setRange(0.0, 100.0); self.sb_open.setDecimals(3); self.sb_open.setValue(0.08); self.sb_open.setToolTip("Opening (µm) to remove small noise/objects.")
        self.cb_thr = QComboBox(); self.cb_thr.addItems(["otsu","adaptive"]); self.cb_thr.setToolTip("Thresholding strategy: global Otsu or adaptive Gaussian.")
        self.sb_block = QSpinBox(); self.sb_block.setRange(3,999); self.sb_block.setValue(31); self.sb_block.setToolTip("Block size for adaptive threshold (odd).")
        self.sb_C = QSpinBox(); self.sb_C.setRange(-255,255); self.sb_C.setValue(-10); self.sb_C.setToolTip("Bias/offset for adaptive threshold.")
        self.cb_split = QCheckBox("Split touching (watershed)"); self.cb_split.setChecked(False); self.cb_split.setToolTip("Watershed-based splitting of touching particles.")
        self.sb_neck = QDoubleSpinBox(); self.sb_neck.setRange(0.0,10.0); self.sb_neck.setDecimals(3); self.sb_neck.setValue(0.12); self.sb_neck.setToolTip("Minimum neck (µm) to cut during watershed.")
        self.sb_seg = QDoubleSpinBox(); self.sb_seg.setRange(0.0,100.0); self.sb_seg.setDecimals(3); self.sb_seg.setValue(0.20); self.sb_seg.setToolTip("Minimum resulting segment diameter (µm) after watershed.")
        rows_thr = [
            ("Min circularity", self.sb_min_circ),
            ("Closing (µm)",    self.sb_closing),
            ("Opening (µm)",    self.sb_open),
            ("Threshold",       self.cb_thr),
            ("Adaptive block size", self.sb_block),
            ("Adaptive C",      self.sb_C),
            (None,              self.cb_split),
            ("Min neck (µm)",   self.sb_neck),
            ("Min segment d (µm)", self.sb_seg),
        ]
        for label, widget in rows_thr:
            if label is None: lay_thr.addRow(widget)
            else: lay_thr.addRow(label, widget)
        form.addRow(self.grp_thr)

        # LoG-only
        self.grp_log = QGroupBox("LoG-only")
        lay_log = QFormLayout(self.grp_log)
        self.sb_log_thr = QDoubleSpinBox(); self.sb_log_thr.setRange(0.0, 1.0); self.sb_log_thr.setDecimals(3); self.sb_log_thr.setValue(0.030); self.sb_log_thr.setToolTip("Relative LoG response threshold (fraction of max).")
        self.sb_log_sep = QDoubleSpinBox(); self.sb_log_sep.setRange(0.0, 10.0); self.sb_log_sep.setDecimals(3); self.sb_log_sep.setValue(0.120); self.sb_log_sep.setToolTip("Minimum distance between peaks (µm).")
        lay_log.addRow("LoG threshold (rel)", self.sb_log_thr); lay_log.addRow("LoG min separation (µm)", self.sb_log_sep)
        form.addRow(self.grp_log)

        # Actions
        self.btn_autotune = QPushButton("Auto-tune"); self.btn_autotune.setToolTip("Automatically adjust parameters for the current image and mode."); self.btn_autotune.clicked.connect(self.autotune_params)
        self.btn_run = QPushButton("Run analysis"); self.btn_run.setToolTip("Start processing with the current parameters."); self.btn_run.clicked.connect(self.run_analysis)
        self.btn_export_csv = QPushButton("Export CSV…"); self.btn_export_csv.setToolTip("Export active particle diameters to CSV."); self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_save_overlay = QPushButton("Save overlay…"); self.btn_save_overlay.setToolTip("Save the overlay image (green=active, red=excluded)."); self.btn_save_overlay.clicked.connect(self.save_overlay)
        self.btn_toggle_remove = QPushButton("Remove blobs (click)"); self.btn_toggle_remove.setCheckable(True); self.btn_toggle_remove.setToolTip("Click a green contour to exclude / click again to restore."); self.btn_toggle_remove.toggled.connect(self.on_toggle_remove)
        self.btn_clear_excl = QPushButton("Clear exclusions"); self.btn_clear_excl.setToolTip("Remove all manual exclusions."); self.btn_clear_excl.clicked.connect(self.on_clear_exclusions)
        form.addRow(self.btn_autotune); form.addRow(self.btn_run); form.addRow(self.btn_export_csv); form.addRow(self.btn_save_overlay); form.addRow(self.btn_toggle_remove); form.addRow(self.btn_clear_excl)

        # -------- Right workspace --------
        right = QWidget(); right_layout = QVBoxLayout(right); right_layout.setContentsMargins(0,0,0,0)
        self.tabs = QTabWidget(); self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.view_original = ImageView();  self.tabs.addTab(self.view_original, "Original")
        self.view_leveled  = ImageView();  self.tabs.addTab(self.view_leveled,  "Leveled")
        self.view_thresh   = ImageView();  self.tabs.addTab(self.view_thresh,   "Threshold")
        self.view_overlay  = ImageView();  self.tabs.addTab(self.view_overlay,  "Overlay")
        self.view_overlay.sig_clicked.connect(self.on_overlay_clicked)

        self.plot_hist = MplWidget(); self.tabs.addTab(self.plot_hist, "Histogram")
        self.plot_cum  = MplWidget(); self.tabs.addTab(self.plot_cum,  "Cumulative")
        right_layout.addWidget(self.tabs, 1)
        self.tabs.currentChanged.connect(lambda _: self.render_overlay_with_highlight())

        self.stats_label = QLabel("—")
        self.table = QTableWidget(0,1); self.table.setHorizontalHeaderLabels(["diameter (µm)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setMinimumHeight(150)
        # Hover support:
        self.table.setMouseTracking(True)
        self.table.cellEntered.connect(self.on_table_cell_entered)
        self.table.viewport().installEventFilter(self)
        right_layout.addWidget(self.stats_label, 0); right_layout.addWidget(self.table, 0)

        # -------- Welcome screen --------
        welcome = QWidget(); wl = QVBoxLayout(welcome)
        lbl = QLabel(
            "<h2 style='margin:0'>SEM PSD (Particle Analysis)</h2>"
            "<p>Open a SEM image (Ctrl+O) or load a test sample:</p>"
            "<ul><li><b>Ctrl+1</b> — samples/blobs.tif</li>"
            "<li><b>Ctrl+2</b> — samples/granular.tif</li></ul>"
            "<p>Pick the analysis mode (Contours / Nano [LoG]) and press <br> <b>Run analysis</b> - <b>Ctrl+R</b><br></p>"
        )
        lbl.setWordWrap(True); lbl.setAlignment(Qt.AlignCenter)
        wl.addStretch(1); wl.addWidget(lbl, alignment=Qt.AlignCenter); wl.addStretch(1)

        # Stacked right area
        self.right_stack = QStackedWidget(); self.right_stack.addWidget(welcome); self.right_stack.addWidget(right)
        self.right_stack.setCurrentIndex(0)

        # Splitter
        splitter = QSplitter(); splitter.addWidget(left); splitter.addWidget(self.right_stack)
        splitter.setStretchFactor(0,0); splitter.setStretchFactor(1,1); splitter.setChildrenCollapsible(False)

        # Root
        root = QVBoxLayout(self); root.addLayout(topbar); root.addWidget(splitter, 1)

        self.add_actions()
        self.update_param_visibility()

    # ---------- Hotkeys ----------
    def add_actions(self):
        self.act_open_global = QAction(self)
        self.act_open_global.setShortcut(QKeySequence(QKeySequence.Open))
        self.act_open_global.setShortcutContext(Qt.ApplicationShortcut)
        self.act_open_global.triggered.connect(self.open_image)
        self.addAction(self.act_open_global)

        self.act_run = QAction(self, triggered=self.run_analysis)
        self.act_run.setShortcut(QKeySequence("Ctrl+R"))
        self.act_run.setShortcutContext(Qt.ApplicationShortcut)
        self.addAction(self.act_run)

        self.act_sample1 = QAction(self, triggered=lambda: self.open_sample("blobs.tif"))
        self.act_sample1.setShortcut(QKeySequence("Ctrl+1"))
        self.act_sample1.setShortcutContext(Qt.ApplicationShortcut)
        self.addAction(self.act_sample1)

        self.act_sample2 = QAction(self, triggered=lambda: self.open_sample("granular.tif"))
        self.act_sample2.setShortcut(QKeySequence("Ctrl+2"))
        self.act_sample2.setShortcutContext(Qt.ApplicationShortcut)
        self.addAction(self.act_sample2)

    # ---------- Help ----------
    def show_help(self):
        txt = textwrap.dedent("""
        ### Modes
        • **Nano (LoG)** — multi-scale Laplacian of Gaussian with σ² normalization. Good for nanoparticles and agglomerates.
        • **Contours (threshold)** — thresholding + morphology + optional watershed.

        ### Common parameters
        • **Min/Max diameter** — size filter in micrometers.
        • **Min rel. contrast** — inside vs. ring average difference (0..1). Filters weak/flat objects.
        • **CLAHE / Top-hat / Level strength** — preprocessing to enhance local contrast and remove background.

        ### Threshold-only
        • **Threshold** (otsu/adaptive), **Block size / C** — binary mask parameters.
        • **Opening / Closing** — noise removal or gap closing.
        • **Watershed (Split touching)** + **Min neck / Min segment d** — splitting touching particles.

        ### LoG-only
        • **LoG threshold (rel)** — minimum relative response (fraction of maximum).
        • **LoG min separation** — minimum distance between peaks to avoid duplicates.
        """).strip()
        QMessageBox.information(self, "Help", txt)

    # ---------- Visibility ----------
    def update_param_visibility(self):
        is_log = (self.cb_mode.currentIndex() == 1)
        self.grp_log.setVisible(is_log)
        self.grp_thr.setVisible(not is_log)
        self.tabs.setTabEnabled(self.tabs.indexOf(self.view_thresh), not is_log)

    # ---------- Unit helpers ----------
    def unit_label(self) -> str:
        return "nm" if self.cb_units.currentIndex() == 1 else "µm"

    def unit_factor(self) -> float:
        return 1000.0 if self.unit_label() == "nm" else 1.0  # µm -> chosen unit

    def scale_stats_for_display(self, st: dict) -> dict:
        if not st:
            return {}
        out = dict(st)
        factor = self.unit_factor()
        for k in ("D10","D50","D90","mean","std","min","max"):
            if k in out:
                out[k] = float(out[k]) * factor
        return out

    def decide_units_from_scale(self, scale_um_per_px: float | None) -> str:
        if scale_um_per_px is None or scale_um_per_px <= 0:
            return self.unit_label()
        return "nm" if scale_um_per_px < self.auto_nm_threshold_um_per_px else "µm"

    def set_display_unit(self, unit: str):
        target_idx = 1 if unit == "nm" else 0
        if self.cb_units.currentIndex() != target_idx:
            self.cb_units.blockSignals(True)
            self.cb_units.setCurrentIndex(target_idx)
            self.cb_units.blockSignals(False)

    def update_units_lock(self):
        self.cb_units.setEnabled(not self.cb_auto_units.isChecked())

    def maybe_update_auto_units(self):
        if not self.cb_auto_units.isChecked():
            return
        scale = self.sb_scale.value()
        if (not scale or scale <= 0) and self.um_per_px_from_meta:
            scale = self.um_per_px_from_meta
        unit = self.decide_units_from_scale(scale)
        self.set_display_unit(unit)
        self.render_previews()
        self.update_stats_table()

    def on_auto_units_toggled(self, on: bool):
        self.update_units_lock()
        if on:
            self.maybe_update_auto_units()

    def on_scale_changed(self, *_):
        self.maybe_update_auto_units()

    def on_units_changed(self, *_):
        # manual change -> just re-render
        self.render_previews()
        self.update_stats_table()

    # ---------- File open helpers ----------
    def _open_path(self, p: Path):
        self.image_path = p
        try:
            self.gray = imread_gray(str(p))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read image:\n{e}")
            return
        self.um_per_px_from_meta = scale_from_metadata(str(p))
        if self.um_per_px_from_meta:
            self.sb_scale.setValue(round(self.um_per_px_from_meta, 6))
            self.meta_label.setText(f"Scale: from meta {self.um_per_px_from_meta:.6f} µm/px (also set in control)")
        else:
            self.meta_label.setText("Scale: — (set µm/px or read metadata)")
        self.maybe_update_auto_units()

        self.view_original.set_image(cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR))
        self.tabs.setCurrentWidget(self.view_original)
        self._show_tabs(True)

    def open_image(self):
        if self._open_in_progress:
            return
        self._open_in_progress = True
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Open SEM image", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)")
            if not path: return
            self._open_path(Path(path))
        finally:
            self._open_in_progress = False

    def _show_tabs(self, on: bool):
        self.right_stack.setCurrentIndex(1 if on else 0)

    def open_sample(self, filename: str):
        samples_dir = Path(__file__).resolve().parent.parent / "samples"
        p = samples_dir / filename
        if not p.exists():
            QMessageBox.warning(self, "Sample not found", f"File not found:\n{p}")
            return
        self._open_path(p)

    def read_scale_meta(self):
        if not self.image_path:
            QMessageBox.information(self, "Info", "Open an image first."); return
        val = scale_from_metadata(str(self.image_path))
        if val is None:
            QMessageBox.warning(self, "Meta", "No usable TIFF metadata found."); return
        self.um_per_px_from_meta = val; self.sb_scale.setValue(round(val, 6))
        self.meta_label.setText(f"Scale: from meta {val:.6f} µm/px (also set in control)")
        self.maybe_update_auto_units()

    def current_params(self) -> Params:
        scale = self.sb_scale.value()
        if scale <= 0 and self.um_per_px_from_meta: scale = self.um_per_px_from_meta
        analysis_mode = "log" if self.cb_mode.currentIndex() == 1 else "contours"
        return Params(
            scale_um_per_px=scale if scale > 0 else None,
            exclude_top=self.sb_top.value(), exclude_bottom=self.sb_bottom.value(),
            exclude_left=self.sb_left.value(), exclude_right=self.sb_right.value(),
            min_d_um=self.sb_min_d.value(), max_d_um=self.sb_max_d.value(),
            closing_um=self.sb_closing.value(), open_um=self.sb_open.value(),
            min_circ=self.sb_min_circ.value(), thr_method=self.cb_thr.currentText(),
            block_size=self.sb_block.value(), block_C=self.sb_C.value(),
            clahe_clip=self.sb_clahe.value(), min_rel_contrast=self.sb_min_rc.value(),
            tophat_um=self.sb_tophat.value(), level_strength=self.sb_level.value(),
            split_touching=self.cb_split.isChecked(), min_neck_um=self.sb_neck.value(),
            min_seg_d_um=self.sb_seg.value(),
            analysis_mode=analysis_mode,
            log_threshold_rel=self.sb_log_thr.value(),
            log_minsep_um=self.sb_log_sep.value(),
        )

    def set_busy(self, busy: bool):
        for w in [self.btn_open, self.btn_run, self.btn_export_csv, self.btn_save_overlay,
                  self.btn_toggle_remove, self.btn_clear_excl, self.cb_mode, self.btn_autotune]:
            w.setEnabled(not busy)
        self.progress.setVisible(busy)
        QApplication.processEvents()

    # ---------- Auto-tune ----------
    def autotune_params(self):
        if self.gray is None:
            QMessageBox.information(self, "Auto-tune", "Open an image first.")
            return
        P = self.current_params()

        def bad_scale(v: float | None) -> bool:
            return (v is None) or (v <= 0) or (v < 1e-4) or (v > 1e3)
        if bad_scale(P.scale_um_per_px):
            meta_val = scale_from_metadata(str(self.image_path)) if self.image_path else None
            if not bad_scale(meta_val):
                self.sb_scale.setValue(round(meta_val, 6))
                P.scale_um_per_px = meta_val
                self.maybe_update_auto_units()
                QMessageBox.information(self, "Auto-tune", f"Scale was invalid. Using scale from metadata: {meta_val:.6f} µm/px")
            else:
                QMessageBox.warning(self, "Auto-tune", "Scale (µm/px) is invalid and no usable value found in metadata.\nPlease set a proper scale manually.")
                return

        roi_mask = make_roi_mask(self.gray.shape, P.exclude_top, P.exclude_bottom, P.exclude_left, P.exclude_right)
        lev = preprocess(self.gray, clahe_clip=max(1.5, P.clahe_clip or 0), tophat_um=max(0.0, P.tophat_um),
                         um_per_px=P.scale_um_per_px, level_strength=max(0.2, P.level_strength))

        d_lo, d_hi = 0.01, min(2.0, max(1.0, (P.max_d_um or 1.0)))
        sigmas = np.exp(np.linspace(np.log(sigma_from_d_um(d_lo, P.scale_um_per_px)),
                                    np.log(sigma_from_d_um(d_hi, P.scale_um_per_px)), 9))
        img = self.gray.astype(np.float32)
        H, W = self.gray.shape
        resp_max = np.zeros((H, W), np.float32)
        arg_sigma = np.zeros((H, W), np.float32)
        for s in sigmas:
            R = log_response(img, float(s)); R = cv2.bitwise_and(R, R, mask=roi_mask)
            better = R > resp_max; resp_max[better] = R[better]; arg_sigma[better] = float(s)

        mx = float(resp_max.max()) if resp_max.size else 0.0
        if mx <= 0:
            QMessageBox.warning(self, "Auto-tune", "Could not estimate LoG response on this image."); return

        candidates = [0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050]
        chosen_thr, chosen_count = candidates[3], 0
        nms = nms2d(resp_max, 1)
        for t in candidates:
            m = (resp_max >= (t * mx)) & nms
            n = int(np.count_nonzero(m))
            chosen_thr, chosen_count = t, n
            if 300 <= n <= 8000: break

        mask = (resp_max >= (chosen_thr * mx)) & nms
        sig_est = arg_sigma[mask]
        if sig_est.size == 0:
            QMessageBox.information(self, "Auto-tune", "No reliable peaks for auto-tune. Try manual."); return

        p20 = float(np.percentile(sig_est, 20))
        p50 = float(np.percentile(sig_est, 50))
        p80 = float(np.percentile(sig_est, 80))

        def sigma_to_um(s: float) -> float:
            return (2.0 * math.sqrt(2.0) * s) * P.scale_um_per_px
        d20_um, d50_um, d80_um = sigma_to_um(p20), sigma_to_um(p50), sigma_to_um(p80)

        if self.cb_mode.currentIndex() == 1:  # LoG
            self.sb_log_thr.setValue(float(min(0.06, max(0.02, chosen_thr))))
            min_d = float(min(0.20, max(0.01, 0.5 * d20_um)))
            max_d = float(min(2.0, max(min_d + 0.02, 1.6 * d80_um)))
            self.sb_min_d.setValue(min_d); self.sb_max_d.setValue(max_d)
            self.sb_log_sep.setValue(float(min(0.25, max(0.05, 0.6 * d50_um))))
            if self.sb_clahe.value() == 0: self.sb_clahe.setValue(2.0)
            self.sb_tophat.setValue(float(min(0.08, max(0.02, 0.6 * d50_um))))
            self.sb_level.setValue(float(min(0.6, max(0.3, self.sb_level.value()))))
        else:  # Threshold
            min_d = float(min(0.30, max(0.01, 0.6 * d20_um)))
            max_d = float(min(1.5, max(min_d + 0.02, 1.8 * d80_um)))
            self.sb_min_d.setValue(min_d); self.sb_max_d.setValue(max_d)
            self.cb_thr.setCurrentText("otsu")
            self.sb_open.setValue(float(min(0.40, max(0.04, 0.30 * d50_um))))
            self.sb_closing.setValue(float(min(0.50, max(0.06, 0.40 * d50_um))))
            split = d50_um < 0.40; self.cb_split.setChecked(split)
            self.sb_neck.setValue(float(min(0.40, max(0.08, 0.30 * d50_um))))
            self.sb_seg.setValue(float(min(0.60, max(0.16, 0.70 * d50_um))))
            self.sb_min_circ.setValue(0.10)
            if self.sb_clahe.value() < 1.5: self.sb_clahe.setValue(2.0)
            self.sb_level.setValue(float(max(0.25, self.sb_level.value())))

        QMessageBox.information(
            self, "Auto-tune",
            f"Estimated sizes (µm): d20≈{d20_um:.3f}, d50≈{d50_um:.3f}, d80≈{d80_um:.3f}\n"
            f"Peaks: {chosen_count} @ rel threshold {chosen_thr:.3f}\n"
            f"Parameters updated for the current mode."
        )

        # Ensure units reflect current scale if auto-units is enabled
        self.maybe_update_auto_units()

    # ---------- Run ----------
    def run_analysis(self):
        if self.gray is None or self.image_path is None:
            QMessageBox.information(self, "Info", "Open an image first."); return
        P = self.current_params()
        if P.scale_um_per_px is None or P.scale_um_per_px <= 0:
            QMessageBox.warning(self, "Scale", "Set Scale (µm/px) or read from metadata."); return
        if self._thread and self._thread.isRunning(): return

        self.set_busy(True)
        self._thread = QThread(self); self._worker = Worker(self.gray.copy(), P, self.image_path)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self.on_worker_finished); self._worker.failed.connect(self.on_worker_failed)
        self._worker.finished.connect(self._thread.quit); self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished); self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()

    def _on_thread_finished(self):
        self.set_busy(False); self._worker = None; self._thread = None

    def on_worker_finished(self, out: dict):
        self.lev = out.get("lev"); self.thr_raw = out.get("thr_raw"); self.thr_proc = out.get("thr_proc")
        self.results = out.get("results", []); self.excluded_idx.clear()
        self.hover_idx = None
        self._rebuild_overlay_and_stats(); self.render_previews(); self.update_stats_table(); self.tabs.setCurrentWidget(self.view_overlay)

    def on_worker_failed(self, msg: str):
        QMessageBox.critical(self, "Error", f"Processing failed:\n{msg}")

    # ---------- Exclusions ----------
    def on_toggle_remove(self, on: bool):
        self.remove_mode = on
        if on: self.tabs.setCurrentWidget(self.view_overlay)

    def on_clear_exclusions(self):
        self.excluded_idx.clear()
        self.hover_idx = None
        self._rebuild_overlay_and_stats(); self.render_previews(); self.update_stats_table()

    def on_overlay_clicked(self, x: int, y: int):
        if not self.remove_mode or self.overlay_img is None or not self.results: return
        hit = None
        for i, (cnt, _, _) in enumerate(self.results):
            if cv2.pointPolygonTest(cnt, (float(x), float(y)), measureDist=False) >= 0:
                hit = i; break
        if hit is None: return
        if hit in self.excluded_idx: self.excluded_idx.remove(hit)
        else: self.excluded_idx.add(hit)
        self.hover_idx = None
        self._rebuild_overlay_and_stats(); self.render_previews(); self.update_stats_table()

    # ---------- Hover ↔ highlight ----------
    def on_table_cell_entered(self, row: int, col: int):
        if 0 <= row < len(self.row_to_result):
            ridx = self.row_to_result[row]
            self.hover_idx = ridx
        else:
            self.hover_idx = None
        self.render_overlay_with_highlight()

    def eventFilter(self, obj, event):
        # Clear highlight when cursor leaves the table viewport,
        # or update it on mouse move (for blank areas)
        if obj is self.table.viewport():
            if event.type() == QEvent.Leave:
                if self.hover_idx is not None:
                    self.hover_idx = None
                    self.render_overlay_with_highlight()
            elif event.type() == QEvent.MouseMove:
                idx = self.table.indexAt(event.pos())
                if idx.isValid():
                    self.on_table_cell_entered(idx.row(), idx.column())
                else:
                    if self.hover_idx is not None:
                        self.hover_idx = None
                        self.render_overlay_with_highlight()
        return super().eventFilter(obj, event)

    # ---------- Build overlay & stats ----------
    def _rebuild_overlay_and_stats(self):
        if self.gray is None: return
        over = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
        active_diams = []
        for i, (cnt, d_um, _c) in enumerate(self.results):
            color = (0,0,255) if i in self.excluded_idx else (0,255,0)
            cv2.drawContours(over, [cnt], -1, color, 2)
            if i not in self.excluded_idx: active_diams.append(d_um)
        self.overlay_img = over
        self.diams_um = np.array(active_diams, float)
        self.stats = stats_from_diams(self.diams_um)

    # ---------- Rendering / Table / Export ----------
    def render_overlay_with_highlight(self):
        # оновлюємо підсвітку тільки коли активна вкладка Overlay
        if self.tabs.currentWidget() is not self.view_overlay:
            return
        if self.overlay_img is None:
            return

        img = self.overlay_img.copy()
        if self.hover_idx is not None and 0 <= self.hover_idx < len(self.results):
            cnt, d_um, _ = self.results[self.hover_idx]

            # приглушення інших + напівпрозора жовта заливка
            base = img.copy()
            img = (img * 0.35).astype(np.uint8)
            mask = np.zeros(img.shape[:2], np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
            mask_dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 1)
            img[mask_dil > 0] = base[mask_dil > 0]
            fill = img.copy()
            cv2.drawContours(fill, [cnt], -1, (0, 255, 255), thickness=-1)
            cv2.addWeighted(fill, 0.35, img, 0.65, 0, dst=img)

            # вужчі контури + менший центр (як просив)
            cv2.drawContours(img, [cnt], -1, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.drawContours(img, [cnt], -1, (0, 255, 255), 2, lineType=cv2.LINE_AA)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]);
                cy = int(M["m01"] / M["m00"])
                cv2.circle(img, (cx, cy), 3, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

                unit = self.unit_label();
                factor = self.unit_factor()
                txt = f"{d_um * factor:.1f} {unit}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                x0, y0 = cx + 8, max(10, cy - 10)
                cv2.rectangle(img, (x0 - 4, y0 - th - 6), (x0 + tw + 4, y0 + 6), (0, 0, 0), -1)
                cv2.putText(img, txt, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        self.view_overlay.set_image(img)

    def render_previews(self):
        if self.gray is not None:
            self.view_original.set_image(cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR))
        if self.lev is not None:
            self.view_leveled.set_image(cv2.cvtColor(self.lev, cv2.COLOR_GRAY2BGR))
        thr_to_show = self.thr_proc if self.thr_proc is not None else self.thr_raw
        if thr_to_show is not None and self.cb_mode.currentIndex() == 0:
            self.view_thresh.set_image(thr_to_show)

        # overlay (with potential highlight)
        self.render_overlay_with_highlight()

        factor = self.unit_factor()
        unit = self.unit_label()
        d_disp = self.diams_um * factor
        st_disp = self.scale_stats_for_display(self.stats)

        self.plot_hist.plot_hist(d_disp, st_disp, "PSD Histogram", unit)
        self.plot_cum.plot_cum(d_disp, st_disp, unit)

    def update_stats_table(self):
        # Build order: active first, then excluded, but keep a mapping row->result index
        rows: list[tuple[int, float, bool]] = []  # (result_idx, diameter_um, excluded)
        for i, (_cnt, d, _c) in enumerate(self.results):
            if i not in self.excluded_idx: rows.append((i, d, False))
        for i, (_cnt, d, _c) in enumerate(self.results):
            if i in self.excluded_idx: rows.append((i, d, True))

        unit = self.unit_label()
        factor = self.unit_factor()
        self.table.setHorizontalHeaderLabels([f"diameter ({unit})"])

        self.row_to_result = [ri for (ri, _d, _ex) in rows]

        if rows:
            st = self.scale_stats_for_display(self.stats)
            self.stats_label.setText(
                f"Particles: {self.stats.get('particles', 0)} | "
                f"D10={st['D10']:.3f} {unit} | D50={st['D50']:.3f} {unit} | "
                f"D90={st['D90']:.3f} {unit} | mean={st['mean']:.3f} {unit} | "
                f"std={st['std']:.3f} {unit}"
            )
            self.table.setRowCount(len(rows))
            for row, (_ri, d_um, excl) in enumerate(rows):
                v = d_um * factor
                cell = QTableWidgetItem(f"{v:.6f}")
                if excl:
                    cell.setForeground(Qt.gray); f = cell.font(); f.setItalic(True); cell.setFont(f)
                self.table.setItem(row, 0, cell)
        else:
            self.stats_label.setText("No particles accepted. Adjust parameters.")
            self.table.setRowCount(0)

    def export_csv(self):
        if not self.results:
            QMessageBox.information(self, "CSV", "Run analysis first."); return
        active = [d for i, (_c, d, _ci) in enumerate(self.results) if i not in self.excluded_idx]
        if not active:
            QMessageBox.information(self, "CSV", "No active particles to export."); return

        unit = self.unit_label()
        factor = self.unit_factor()

        path, _ = QFileDialog.getSaveFileName(self, "Save diameters CSV", f"diameters_{unit}.csv", "CSV (*.csv)")
        if not path: return
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                col = f"diameter_{'nm' if unit=='nm' else 'um'}"
                w.writerow([col])
                for d_um in active:
                    w.writerow([f"{d_um * factor:.6f}"])
        except Exception as e:
            QMessageBox.critical(self, "CSV", f"Failed to save: {e}")

    def save_overlay(self):
        if self.overlay_img is None:
            QMessageBox.information(self, "Overlay", "Run analysis first.")
            return

        pics = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation) or ""
        default_path = str(Path(pics) / "overlay.png")

        path, selected = QFileDialog.getSaveFileName(
            self, "Save overlay image", default_path,
            "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not path:
            return

        p = Path(path)

        if p.suffix.lower() == "":
            sel = (selected or "").lower()
            if "png" in sel:
                p = p.with_suffix(".png")
            elif "jpg" in sel or "jpeg" in sel:
                p = p.with_suffix(".jpg")
            else:
                p = p.with_suffix(".png")

        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Overlay", f"Cannot create folder:\n{p.parent}\n{e}")
            return

        img = np.ascontiguousarray(self.overlay_img)

        try:
            h, w, _ = img.shape
            qimg = QImage(img.data, w, h, 3 * w, QImage.Format_BGR888)
            if qimg.save(str(p)):
                QMessageBox.information(self, "Overlay", f"Saved:\n{p}")
                return
        except Exception:
            pass

        try:
            ok = cv2.imwrite(str(p), img)
            if ok:
                QMessageBox.information(self, "Overlay", f"Saved:\n{p}")
                return
        except Exception as e:
            QMessageBox.critical(self, "Overlay", f"Failed to save with OpenCV:\n{e}")
            return

        QMessageBox.critical(
            self, "Overlay",
            "Failed to save the overlay image.\n"
            "Ensure the folder is writable and the filename has a valid extension (.png / .jpg)."
        )

def main():
    app = QApplication(sys.argv)
    app.setStyle(ToolTipDelayStyle(app.style()))
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
