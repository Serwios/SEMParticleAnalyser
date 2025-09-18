# sem_psd_gui.py
# A desktop GUI (PySide6) wrapper around your sem_psd_blobs.py pipeline
# - Load SEM image (8/16-bit TIFF supported)
# - Configure parameters (scale, ROI, thresholding, morphology, filters)
# - Run analysis and preview: Original, Leveled, Threshold, Overlay, Hist, Cumulative
# - View PSD stats + table of diameters, export CSV/overlay
#
# Usage: python sem_psd_gui.py
#
# Requirements: PySide6, opencv-python, pillow, numpy, matplotlib
#   pip install PySide6 opencv-python pillow numpy matplotlib

from __future__ import annotations
import sys, math, re, csv, os
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import cv2
from PIL import Image

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication, QWidget, QFileDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFormLayout, QDoubleSpinBox, QSpinBox, QTabWidget, QComboBox, QCheckBox, QGroupBox,
    QMessageBox, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit
)

# Matplotlib for plots inside Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ===================== Utilities from sem_psd_blobs =====================

def safe_stem(st: str) -> str:
    import re as _re
    return _re.sub(r'[^A-Za-z0-9._-]+', '_', st)


def imread_gray(path: str) -> np.ndarray:
    """Reads 8/16-bit images, returns uint8 grayscale."""
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
            if isinstance(val, bytes):
                s = val.decode(errors="ignore")
            elif isinstance(val, (list, tuple)):
                s = " ".join([v.decode(errors="ignore") if isinstance(v, bytes) else str(v) for v in val])
            else:
                s = str(val)
            out.append(f"[{tag}] {s}")
    except Exception:
        pass
    try:
        for k, v in (pil.info or {}).items():
            if isinstance(v, bytes):
                v = v.decode(errors="ignore")
            out.append(f"[{k}] {v}")
    except Exception:
        pass
    return "\n".join(out)


def parse_um_per_px_from_text(txt: str) -> float | None:
    if not txt:
        return None
    m = re.search(r"PixelWidth\s*=\s*([0-9eE\.\-\+]+)", txt)
    if m:
        try:
            px_m = float(m.group(1))
            if px_m > 0:
                return px_m * 1e6
        except Exception:
            pass
    m_hfw = re.search(r"(HorFieldsize|HFW)\s*=\s*([0-9eE\.\-\+]+)", txt)
    m_rx = re.search(r"(ResolutionX|Resolutionx)\s*=\s*([0-9]+)", txt)
    if m_hfw and m_rx:
        try:
            hfw_m = float(m_hfw.group(2))
            resx = int(m_rx.group(2))
            if hfw_m > 0 and resx > 0:
                return (hfw_m * 1e6) / float(resx)
        except Exception:
            pass
    return None


def scale_from_metadata(image_path: str) -> float | None:
    return parse_um_per_px_from_text(dump_tiff_metadata_text(image_path))


def make_roi_mask(shape, top=0.02, bottom=0.22, left=0.0, right=0.0):
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    t = int(h * top)
    b = int(h * (1.0 - bottom))
    l = int(w * left)
    r = int(w * (1.0 - right))
    mask[max(0, t) : max(0, b), max(0, l) : max(0, r)] = 255
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
        if block_size % 2 == 0:
            block_size += 1
        bwB = cv2.adaptiveThreshold(
            lev, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
        )
    else:
        _, bwB = cv2.threshold(lev, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bwB = cv2.bitwise_and(bwB, bwB, mask=roi_mask)
    bwD = 255 - bwB
    return bwB, bwD


def morph_open(bw, open_um, um_per_px):
    if open_um <= 0:
        return bw
    r = max(1, int(open_um / um_per_px))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker, iterations=1)


def morph_close(bw, closing_um, um_per_px):
    if closing_um <= 0:
        return bw
    rad_px = max(1, int(closing_um / um_per_px))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad_px + 1, 2 * rad_px + 1))
    return cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker, iterations=1)


def fill_small_holes(mask: np.ndarray, max_frac: float) -> np.ndarray:
    filled = mask.copy()
    contours, hier = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return filled
    hier = hier[0]
    for i, (cnt, h) in enumerate(zip(contours, hier)):
        if h[3] != -1:
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
    if num < 2:
        return bw
    markers = markers.astype(np.int32)
    mask3 = cv2.cvtColor((obj * 255), cv2.COLOR_GRAY2BGR)
    cv2.watershed(mask3, markers)
    seg = np.zeros_like(obj, np.uint8)
    for lab in range(1, markers.max() + 1):
        m = (markers == lab) & (obj.astype(bool))
        if not m.any():
            continue
        A = float(m.sum())
        d_px = 2.0 * math.sqrt(A / math.pi)
        d_um = d_px * um_per_px
        if d_um >= min_seg_d_um:
            seg[m] = 255
    return seg


def count_reasonable_components(bw, um_per_px, dmin_um, dmax_um):
    num, lbl, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    def area_from_d_um(d_um):
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


def measure_components(bw, min_d_um, max_d_um, min_circ, um_per_px, lev_img, min_rel_contrast):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    H, W = bw.shape
    results = []
    for i in range(1, num):
        a = stats[i, cv2.CC_STAT_AREA]
        def area_from_d_um(d_um):
            if d_um is None:
                return None
            r_px = max(0.5, (d_um / 2.0) / um_per_px)
            return math.pi * r_px * r_px
        area_min = area_from_d_um(min_d_um) if min_d_um else None
        area_max = area_from_d_um(max_d_um) if max_d_um else None
        if area_min is not None and a < area_min:
            continue
        if area_max is not None and a > area_max:
            continue
        x, y, w, h = (
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
        )
        roi = (labels[y : y + h, x : x + w] == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        cnt[:, 0, 0] += x
        cnt[:, 0, 1] += y
        A = float(cv2.contourArea(cnt))
        if A < 1.0:
            continue
        P = float(cv2.arcLength(cnt, True))
        circ = (4.0 * math.pi * A) / (P * P + 1e-9)
        if circ < min_circ:
            continue
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
            cv2.drawContours(ring, [cnt], -1, 255, thickness=ring_thick * 2)
            ring = cv2.subtract(ring, mask_obj)
            vals_bg = lev_img[ring == 255]
        mean_in = float(vals_in.mean()) if vals_in.size else 0.0
        mean_bg = float(vals_bg.mean()) if vals_bg.size else 0.0
        rel_contrast = (mean_in - mean_bg) / 255.0
        if rel_contrast < min_rel_contrast:
            continue
        d_px = 2.0 * math.sqrt(A / math.pi)
        d_um = d_px * um_per_px
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

# ===================== Qt Helpers =====================

class MplWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def plot_hist(self, d_um: np.ndarray, st: dict, title: str):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        if d_um.size:
            ax.hist(d_um, bins=40)
            for name in ("D10", "D50", "D90"):
                ax.axvline(st[name], linestyle="--", label=f"{name}={st[name]:.2f} µm")
            ax.set_xlabel("Particle diameter (µm)")
            ax.set_ylabel("Count")
            ax.set_title(title)
            ax.grid(True)
            ax.legend()
        self.canvas.draw()

    def plot_cum(self, d_um: np.ndarray, st: dict):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        if d_um.size:
            s = np.sort(d_um)
            cum = np.arange(1, s.size + 1) / s.size * 100.0
            ax.plot(s, cum)
            for name in ("D10", "D50", "D90"):
                ax.axvline(st[name], linestyle="--", label=f"{name}={st[name]:.2f} µm")
            ax.set_xlabel("Particle diameter (µm)")
            ax.set_ylabel("Cumulative %")
            ax.set_title("Cumulative PSD")
            ax.grid(True)
            ax.legend()
        self.canvas.draw()


def np_to_qpix(img: np.ndarray) -> QPixmap:
    if img.ndim == 2:
        qimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        qimg = img
    h, w, _ = qimg.shape
    qimg = cv2.cvtColor(qimg, cv2.COLOR_BGR2RGB)
    from PySide6.QtGui import QImage
    qim = QImage(qimg.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qim)


# ===================== Main Window =====================

@dataclass
class Params:
    scale_um_per_px: float | None
    exclude_top: float
    exclude_bottom: float
    exclude_left: float
    exclude_right: float
    min_d_um: float
    max_d_um: float
    closing_um: float
    open_um: float
    min_circ: float
    thr_method: str
    block_size: int
    block_C: int
    clahe_clip: float
    min_rel_contrast: float
    tophat_um: float
    level_strength: float
    split_touching: bool
    min_neck_um: float
    min_seg_d_um: float


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SEM PSD (Blob Analysis) — Desktop GUI")
        self.resize(1400, 900)

        # State
        self.image_path: Path | None = None
        self.gray: np.ndarray | None = None
        self.overlay: np.ndarray | None = None
        self.lev: np.ndarray | None = None
        self.thr_img: np.ndarray | None = None
        self.diams_um: np.ndarray = np.array([], float)
        self.stats: dict = {}
        self.um_per_px_from_meta: float | None = None

        # UI
        self.build_ui()

    def build_ui(self):
        topbar = QHBoxLayout()
        self.btn_open = QPushButton("Open image…")
        self.btn_open.clicked.connect(self.open_image)
        self.meta_label = QLabel("Scale: —")
        self.meta_label.setStyleSheet("color: #666")
        topbar.addWidget(self.btn_open)
        topbar.addWidget(self.meta_label)
        topbar.addStretch(1)

        # --- Left panel (parameters) ---
        left = QWidget()
        form = QFormLayout(left)

        self.sb_scale = QDoubleSpinBox(); self.sb_scale.setRange(0.000001, 1000.0); self.sb_scale.setDecimals(6); self.sb_scale.setValue(0.0)
        self.sb_scale.setToolTip("µm per pixel. 0 = try read from TIFF metadata")
        form.addRow("Scale (µm/px)", self.sb_scale)

        self.cb_autoscale = QPushButton("Read scale from metadata")
        self.cb_autoscale.clicked.connect(self.read_scale_meta)
        form.addRow(" ", self.cb_autoscale)

        # ROI
        self.sb_top = QDoubleSpinBox(); self.sb_top.setRange(0, 0.9); self.sb_top.setSingleStep(0.01); self.sb_top.setValue(0.02)
        self.sb_bottom = QDoubleSpinBox(); self.sb_bottom.setRange(0, 0.9); self.sb_bottom.setSingleStep(0.01); self.sb_bottom.setValue(0.22)
        self.sb_left = QDoubleSpinBox(); self.sb_left.setRange(0, 0.9); self.sb_left.setSingleStep(0.01); self.sb_left.setValue(0.0)
        self.sb_right = QDoubleSpinBox(); self.sb_right.setRange(0, 0.9); self.sb_right.setSingleStep(0.01); self.sb_right.setValue(0.0)
        roi_box = QHBoxLayout();
        for w in (self.sb_top, self.sb_bottom, self.sb_left, self.sb_right):
            w.setDecimals(3)
        row_roi = QWidget(); lroi = QFormLayout(row_roi)
        lroi.addRow("Top", self.sb_top)
        lroi.addRow("Bottom", self.sb_bottom)
        lroi.addRow("Left", self.sb_left)
        lroi.addRow("Right", self.sb_right)
        form.addRow(QLabel("ROI exclude ratios"), row_roi)

        # Filters
        self.sb_min_d = QDoubleSpinBox(); self.sb_min_d.setRange(0.0, 1000.0); self.sb_min_d.setDecimals(3); self.sb_min_d.setValue(0.01)
        self.sb_max_d = QDoubleSpinBox(); self.sb_max_d.setRange(0.0, 10000.0); self.sb_max_d.setDecimals(3); self.sb_max_d.setValue(10.0)
        self.sb_min_circ = QDoubleSpinBox(); self.sb_min_circ.setRange(0.0, 1.0); self.sb_min_circ.setDecimals(3); self.sb_min_circ.setValue(0.10)
        form.addRow("Min diameter (µm)", self.sb_min_d)
        form.addRow("Max diameter (µm)", self.sb_max_d)
        form.addRow("Min circularity", self.sb_min_circ)

        # Morphology
        self.sb_closing = QDoubleSpinBox(); self.sb_closing.setRange(0.0, 100.0); self.sb_closing.setDecimals(3); self.sb_closing.setValue(0.12)
        self.sb_open = QDoubleSpinBox(); self.sb_open.setRange(0.0, 100.0); self.sb_open.setDecimals(3); self.sb_open.setValue(0.08)
        form.addRow("Closing (µm)", self.sb_closing)
        form.addRow("Opening (µm)", self.sb_open)

        # Threshold
        self.cb_thr = QComboBox(); self.cb_thr.addItems(["otsu", "adaptive"])
        self.sb_block = QSpinBox(); self.sb_block.setRange(3, 999); self.sb_block.setValue(31)
        self.sb_C = QSpinBox(); self.sb_C.setRange(-255, 255); self.sb_C.setValue(-10)
        form.addRow("Threshold", self.cb_thr)
        form.addRow("Adaptive block size", self.sb_block)
        form.addRow("Adaptive C", self.sb_C)

        # Preprocess
        self.sb_clahe = QDoubleSpinBox(); self.sb_clahe.setRange(0.0, 10.0); self.sb_clahe.setDecimals(2); self.sb_clahe.setValue(2.0)
        self.sb_tophat = QDoubleSpinBox(); self.sb_tophat.setRange(0.0, 100.0); self.sb_tophat.setDecimals(3); self.sb_tophat.setValue(0.0)
        self.sb_level = QDoubleSpinBox(); self.sb_level.setRange(0.0, 1.5); self.sb_level.setDecimals(2); self.sb_level.setValue(0.3)
        self.sb_min_rc = QDoubleSpinBox(); self.sb_min_rc.setRange(0.0, 1.0); self.sb_min_rc.setDecimals(2); self.sb_min_rc.setValue(0.15)
        form.addRow("CLAHE clip", self.sb_clahe)
        form.addRow("Top-hat radius (µm)", self.sb_tophat)
        form.addRow("Level strength", self.sb_level)
        form.addRow("Min rel. contrast", self.sb_min_rc)

        # Watershed
        self.cb_split = QCheckBox("Split touching (watershed)")
        self.cb_split.setChecked(False)
        self.sb_neck = QDoubleSpinBox(); self.sb_neck.setRange(0.0, 10.0); self.sb_neck.setDecimals(3); self.sb_neck.setValue(0.12)
        self.sb_seg = QDoubleSpinBox(); self.sb_seg.setRange(0.0, 100.0); self.sb_seg.setDecimals(3); self.sb_seg.setValue(0.20)
        form.addRow(self.cb_split)
        form.addRow("Min neck (µm)", self.sb_neck)
        form.addRow("Min segment d (µm)", self.sb_seg)

        self.btn_run = QPushButton("Run analysis")
        self.btn_run.clicked.connect(self.run_analysis)
        form.addRow(self.btn_run)

        self.btn_export_csv = QPushButton("Export CSV…")
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_save_overlay = QPushButton("Save overlay…")
        self.btn_save_overlay.clicked.connect(self.save_overlay)
        form.addRow(self.btn_export_csv)
        form.addRow(self.btn_save_overlay)

        # --- Right panel (previews + stats) ---
        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.tabs = QTabWidget()
        # Original
        self.lbl_original = QLabel("No image"); self.lbl_original.setAlignment(Qt.AlignCenter)
        self.tabs.addTab(self.lbl_original, "Original")
        # Leveled
        self.lbl_leveled = QLabel(); self.lbl_leveled.setAlignment(Qt.AlignCenter)
        self.tabs.addTab(self.lbl_leveled, "Leveled")
        # Threshold
        self.lbl_threshold = QLabel(); self.lbl_threshold.setAlignment(Qt.AlignCenter)
        self.tabs.addTab(self.lbl_threshold, "Threshold")
        # Overlay
        self.lbl_overlay = QLabel(); self.lbl_overlay.setAlignment(Qt.AlignCenter)
        self.tabs.addTab(self.lbl_overlay, "Overlay")
        # Hist & Cumulative
        self.plot_hist = MplWidget(); self.tabs.addTab(self.plot_hist, "Histogram")
        self.plot_cum = MplWidget(); self.tabs.addTab(self.plot_cum, "Cumulative")
        right_layout.addWidget(self.tabs)

        # Stats + table
        self.stats_label = QLabel("—")
        self.table = QTableWidget(0, 1); self.table.setHorizontalHeaderLabels(["diameter (µm)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setMinimumHeight(150)
        right_layout.addWidget(self.stats_label)
        right_layout.addWidget(self.table)

        # Splitter
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Root layout
        root = QVBoxLayout(self)
        root.addLayout(topbar)
        root.addWidget(splitter)

        # Menu-like shortcuts
        self.add_actions()

    def add_actions(self):
        act_open = QAction(self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.open_image)
        self.addAction(act_open)
        act_run = QAction(self)
        act_run.setShortcut("Ctrl+R")
        act_run.triggered.connect(self.run_analysis)
        self.addAction(act_run)

    # ---------------------- Events ----------------------

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open SEM image", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self.image_path = Path(path)
        try:
            self.gray = imread_gray(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read image:\n{e}")
            return
        self.um_per_px_from_meta = scale_from_metadata(path)
        txt = f"Scale: from meta {self.um_per_px_from_meta:.6f} µm/px" if self.um_per_px_from_meta else "Scale: — (set µm/px or read metadata)"
        self.meta_label.setText(txt)
        self.lbl_original.setPixmap(np_to_qpix(cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)).scaled(self.tabs.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.tabs.setCurrentIndex(0)

    def read_scale_meta(self):
        if not self.image_path:
            QMessageBox.information(self, "Info", "Open an image first.")
            return
        val = scale_from_metadata(str(self.image_path))
        if val is None:
            QMessageBox.warning(self, "Meta", "No usable TIFF metadata found.")
            return
        self.um_per_px_from_meta = val
        self.sb_scale.setValue(round(val, 6))
        self.meta_label.setText(f"Scale: from meta {val:.6f} µm/px (also set in control)")

    def current_params(self) -> Params:
        scale = self.sb_scale.value()
        if scale <= 0 and self.um_per_px_from_meta:
            scale = self.um_per_px_from_meta
        return Params(
            scale_um_per_px=scale if scale > 0 else None,
            exclude_top=self.sb_top.value(),
            exclude_bottom=self.sb_bottom.value(),
            exclude_left=self.sb_left.value(),
            exclude_right=self.sb_right.value(),
            min_d_um=self.sb_min_d.value(),
            max_d_um=self.sb_max_d.value(),
            closing_um=self.sb_closing.value(),
            open_um=self.sb_open.value(),
            min_circ=self.sb_min_circ.value(),
            thr_method=self.cb_thr.currentText(),
            block_size=self.sb_block.value(),
            block_C=self.sb_C.value(),
            clahe_clip=self.sb_clahe.value(),
            min_rel_contrast=self.sb_min_rc.value(),
            tophat_um=self.sb_tophat.value(),
            level_strength=self.sb_level.value(),
            split_touching=self.cb_split.isChecked(),
            min_neck_um=self.sb_neck.value(),
            min_seg_d_um=self.sb_seg.value(),
        )

    def run_analysis(self):
        if self.gray is None or self.image_path is None:
            QMessageBox.information(self, "Info", "Open an image first.")
            return
        P = self.current_params()
        if P.scale_um_per_px is None or P.scale_um_per_px <= 0:
            QMessageBox.warning(self, "Scale", "Set Scale (µm/px) or read from metadata.")
            return
        try:
            roi_mask = make_roi_mask(self.gray.shape, P.exclude_top, P.exclude_bottom, P.exclude_left, P.exclude_right)
            self.lev = preprocess(self.gray, P.clahe_clip, P.tophat_um, P.scale_um_per_px, P.level_strength)
            bwB, bwD = threshold_pair(self.lev, roi_mask, method=P.thr_method, block_size=P.block_size, C=P.block_C)
            kB = count_reasonable_components(bwB, P.scale_um_per_px, P.min_d_um, P.max_d_um)
            kD = count_reasonable_components(bwD, P.scale_um_per_px, P.min_d_um, P.max_d_um)
            bw = bwB if kB >= kD else bwD
            bw_m = morph_close(bw, P.closing_um, P.scale_um_per_px)
            bw_m = morph_open(bw_m, P.open_um, P.scale_um_per_px)
            bw_m = fill_small_holes(bw_m, 0.6)
            if P.split_touching:
                bw_m = split_touching_watershed(bw_m, P.scale_um_per_px, P.min_neck_um, P.min_seg_d_um)
            # Measure
            results = measure_components(bw_m, P.min_d_um, P.max_d_um, P.min_circ, P.scale_um_per_px, self.lev, P.min_rel_contrast)

            overlay = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
            diams = []
            for cnt, d_um, circ in results:
                diams.append(d_um)
                cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
            self.overlay = overlay
            self.thr_img = bw
            self.diams_um = np.array(diams, float)
            self.stats = stats_from_diams(self.diams_um)

            # Update previews
            self.render_previews()
            self.update_stats_table()
            self.tabs.setCurrentWidget(self.lbl_overlay)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing failed:\n{e}")

    def render_previews(self):
        if self.gray is not None:
            self.lbl_original.setPixmap(np_to_qpix(cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)).scaled(self.lbl_original.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if self.lev is not None:
            self.lbl_leveled.setPixmap(np_to_qpix(cv2.cvtColor(self.lev, cv2.COLOR_GRAY2BGR)).scaled(self.lbl_leveled.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if self.thr_img is not None:
            self.lbl_threshold.setPixmap(np_to_qpix(cv2.cvtColor(self.thr_img, cv2.COLOR_GRAY2BGR)).scaled(self.lbl_threshold.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if self.overlay is not None:
            self.lbl_overlay.setPixmap(np_to_qpix(self.overlay).scaled(self.lbl_overlay.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # Plots
        self.plot_hist.plot_hist(self.diams_um, self.stats, "PSD Histogram")
        self.plot_cum.plot_cum(self.diams_um, self.stats)

    def update_stats_table(self):
        if self.diams_um.size:
            st = self.stats
            self.stats_label.setText(
                f"Particles: {st['particles']} | D10={st['D10']:.3f} µm | D50={st['D50']:.3f} µm | D90={st['D90']:.3f} µm | mean={st['mean']:.3f} µm | std={st['std']:.3f} µm"
            )
            self.table.setRowCount(len(self.diams_um))
            for i, v in enumerate(self.diams_um):
                self.table.setItem(i, 0, QTableWidgetItem(f"{v:.6f}"))
        else:
            self.stats_label.setText("No particles accepted. Adjust parameters.")
            self.table.setRowCount(0)

    def export_csv(self):
        if not self.diams_um.size:
            QMessageBox.information(self, "CSV", "Run analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save diameters CSV", "diameters.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["diameter_um"])
                for d in self.diams_um:
                    w.writerow([f"{d:.6f}"])
        except Exception as e:
            QMessageBox.critical(self, "CSV", f"Failed to save: {e}")

    def save_overlay(self):
        if self.overlay is None:
            QMessageBox.information(self, "Overlay", "Run analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save overlay image", "overlay.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if not path:
            return
        try:
            bgr = cv2.cvtColor(self.overlay, cv2.COLOR_RGB2BGR) if self.overlay.shape[2] == 3 else self.overlay
            cv2.imwrite(path, bgr)
        except Exception as e:
            QMessageBox.critical(self, "Overlay", f"Failed to save: {e}")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
