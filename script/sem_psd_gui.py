# sem_psd_gui.py
# Desktop GUI (PySide6) for SEM particle-size analysis with click-to-exclude blobs.
# Fullscreen-friendly image viewer (auto-fit), robust Threshold rendering.

from __future__ import annotations
import sys, math, re, csv
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import cv2
from PIL import Image

# PySide6
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QAction, QTransform, QPainter
from PySide6.QtWidgets import (
    QApplication, QWidget, QFileDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFormLayout, QDoubleSpinBox, QSpinBox, QTabWidget, QComboBox, QCheckBox,
    QMessageBox, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)

# Matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


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
        if h[3] != -1: continue  # only outer
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

        A = float(cv2.contourArea(cnt));
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
    from PySide6.QtGui import QImage
    qim = QImage(qimg.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qim)


# ---------------- ImageView (zoom/pan + auto-fit) ----------------

class ImageView(QGraphicsView):
    sig_clicked = Signal(int, int)  # image coords

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
        self.setTransform(QTransform())  # reset zoom
        r = self._item.boundingRect()
        m = 2  # small margin px
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
                self.setTransform(QTransform())  # 100%
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
            # map to image coords and emit
            p = self.mapToScene(e.pos())
            x = int(round(p.x()))
            y = int(round(p.y()))
            # clamp
            x = max(0, min(self._img_w - 1, x))
            y = max(0, min(self._img_h - 1, y))
            self.sig_clicked.emit(x, y)
        super().mousePressEvent(e)


# ---------------- Qt helper widgets ----------------

class MplWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def plot_hist(self, d_um: np.ndarray, st: dict, title: str):
        self.fig.clear(); ax = self.fig.add_subplot(111)
        if d_um.size:
            ax.hist(d_um, bins=40)
            for name in ("D10","D50","D90"):
                ax.axvline(st[name], linestyle="--", label=f"{name}={st[name]:.2f} µm")
            ax.set_xlabel("Particle diameter (µm)"); ax.set_ylabel("Count"); ax.set_title(title)
            ax.grid(True); ax.legend()
        self.canvas.draw()

    def plot_cum(self, d_um: np.ndarray, st: dict):
        self.fig.clear(); ax = self.fig.add_subplot(111)
        if d_um.size:
            s = np.sort(d_um); cum = np.arange(1, s.size+1)/s.size*100.0
            ax.plot(s, cum)
            for name in ("D10","D50","D90"):
                ax.axvline(st[name], linestyle="--", label=f"{name}={st[name]:.2f} µm")
            ax.set_xlabel("Particle diameter (µm)"); ax.set_ylabel("Cumulative %"); ax.set_title("Cumulative PSD")
            ax.grid(True); ax.legend()
        self.canvas.draw()


# ---------------- App ----------------

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

        self.build_ui()

    # ---------- UI ----------

    def build_ui(self):
        topbar = QHBoxLayout()
        self.btn_open = QPushButton("Open image…"); self.btn_open.clicked.connect(self.open_image)
        self.meta_label = QLabel("Scale: —"); self.meta_label.setStyleSheet("color:#666")
        topbar.addWidget(self.btn_open); topbar.addWidget(self.meta_label); topbar.addStretch(1)

        # Left controls
        left = QWidget(); form = QFormLayout(left)

        self.sb_scale = QDoubleSpinBox(); self.sb_scale.setRange(1e-6, 1000.0); self.sb_scale.setDecimals(6); self.sb_scale.setValue(0.0)
        form.addRow("Scale (µm/px)", self.sb_scale)

        self.cb_autoscale = QPushButton("Read scale from metadata"); self.cb_autoscale.clicked.connect(self.read_scale_meta)
        form.addRow(" ", self.cb_autoscale)

        # ROI
        self.sb_top = QDoubleSpinBox(); self.sb_top.setRange(0,0.9); self.sb_top.setSingleStep(0.01); self.sb_top.setValue(0.02)
        self.sb_bottom = QDoubleSpinBox(); self.sb_bottom.setRange(0,0.9); self.sb_bottom.setSingleStep(0.01); self.sb_bottom.setValue(0.22)
        self.sb_left = QDoubleSpinBox(); self.sb_left.setRange(0,0.9); self.sb_left.setSingleStep(0.01); self.sb_left.setValue(0.0)
        self.sb_right = QDoubleSpinBox(); self.sb_right.setRange(0,0.9); self.sb_right.setSingleStep(0.01); self.sb_right.setValue(0.0)
        row_roi = QWidget(); lroi = QFormLayout(row_roi)
        for w in (self.sb_top, self.sb_bottom, self.sb_left, self.sb_right): w.setDecimals(3)
        lroi.addRow("Top", self.sb_top); lroi.addRow("Bottom", self.sb_bottom)
        lroi.addRow("Left", self.sb_left); lroi.addRow("Right", self.sb_right)
        form.addRow(QLabel("ROI exclude ratios"), row_roi)

        # Filters
        self.sb_min_d = QDoubleSpinBox(); self.sb_min_d.setRange(0.0, 1000.0); self.sb_min_d.setDecimals(3); self.sb_min_d.setValue(0.01)
        self.sb_max_d = QDoubleSpinBox(); self.sb_max_d.setRange(0.0, 10000.0); self.sb_max_d.setDecimals(3); self.sb_max_d.setValue(10.0)
        self.sb_min_circ = QDoubleSpinBox(); self.sb_min_circ.setRange(0.0, 1.0); self.sb_min_circ.setDecimals(3); self.sb_min_circ.setValue(0.10)
        form.addRow("Min diameter (µm)", self.sb_min_d); form.addRow("Max diameter (µm)", self.sb_max_d); form.addRow("Min circularity", self.sb_min_circ)

        # Morphology
        self.sb_closing = QDoubleSpinBox(); self.sb_closing.setRange(0.0, 100.0); self.sb_closing.setDecimals(3); self.sb_closing.setValue(0.12)
        self.sb_open = QDoubleSpinBox(); self.sb_open.setRange(0.0, 100.0); self.sb_open.setDecimals(3); self.sb_open.setValue(0.08)
        form.addRow("Closing (µm)", self.sb_closing); form.addRow("Opening (µm)", self.sb_open)

        # Threshold
        self.cb_thr = QComboBox(); self.cb_thr.addItems(["otsu","adaptive"])
        self.sb_block = QSpinBox(); self.sb_block.setRange(3,999); self.sb_block.setValue(31)
        self.sb_C = QSpinBox(); self.sb_C.setRange(-255,255); self.sb_C.setValue(-10)
        form.addRow("Threshold", self.cb_thr); form.addRow("Adaptive block size", self.sb_block); form.addRow("Adaptive C", self.sb_C)

        # Preprocess
        self.sb_clahe = QDoubleSpinBox(); self.sb_clahe.setRange(0.0,10.0); self.sb_clahe.setDecimals(2); self.sb_clahe.setValue(2.0)
        self.sb_tophat = QDoubleSpinBox(); self.sb_tophat.setRange(0.0,100.0); self.sb_tophat.setDecimals(3); self.sb_tophat.setValue(0.0)
        self.sb_level = QDoubleSpinBox(); self.sb_level.setRange(0.0,1.5); self.sb_level.setDecimals(2); self.sb_level.setValue(0.3)
        self.sb_min_rc = QDoubleSpinBox(); self.sb_min_rc.setRange(0.0,1.0); self.sb_min_rc.setDecimals(2); self.sb_min_rc.setValue(0.15)
        form.addRow("CLAHE clip", self.sb_clahe); form.addRow("Top-hat radius (µm)", self.sb_tophat)
        form.addRow("Level strength", self.sb_level); form.addRow("Min rel. contrast", self.sb_min_rc)

        # Watershed
        self.cb_split = QCheckBox("Split touching (watershed)"); self.cb_split.setChecked(False)
        self.sb_neck = QDoubleSpinBox(); self.sb_neck.setRange(0.0,10.0); self.sb_neck.setDecimals(3); self.sb_neck.setValue(0.12)
        self.sb_seg = QDoubleSpinBox(); self.sb_seg.setRange(0.0,100.0); self.sb_seg.setDecimals(3); self.sb_seg.setValue(0.20)
        form.addRow(self.cb_split); form.addRow("Min neck (µm)", self.sb_neck); form.addRow("Min segment d (µm)", self.sb_seg)

        # Run / Export
        self.btn_run = QPushButton("Run analysis"); self.btn_run.clicked.connect(self.run_analysis); form.addRow(self.btn_run)
        self.btn_export_csv = QPushButton("Export CSV…"); self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_save_overlay = QPushButton("Save overlay…"); self.btn_save_overlay.clicked.connect(self.save_overlay)
        form.addRow(self.btn_export_csv); form.addRow(self.btn_save_overlay)

        # Exclusions
        self.btn_toggle_remove = QPushButton("Remove blobs (click)"); self.btn_toggle_remove.setCheckable(True)
        self.btn_toggle_remove.setToolTip("Click a green contour to exclude / click again to restore")
        self.btn_toggle_remove.toggled.connect(self.on_toggle_remove)
        self.btn_clear_excl = QPushButton("Clear exclusions"); self.btn_clear_excl.clicked.connect(self.on_clear_exclusions)
        form.addRow(self.btn_toggle_remove); form.addRow(self.btn_clear_excl)

        # Right panel (views + stats)
        right = QWidget(); right_layout = QVBoxLayout(right); right_layout.setContentsMargins(0,0,0,0)

        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.view_original = ImageView();  self.tabs.addTab(self.view_original, "Original")
        self.view_leveled  = ImageView();  self.tabs.addTab(self.view_leveled,  "Leveled")
        self.view_thresh   = ImageView();  self.tabs.addTab(self.view_thresh,   "Threshold")
        self.view_overlay  = ImageView();  self.tabs.addTab(self.view_overlay,  "Overlay")
        self.view_overlay.sig_clicked.connect(self.on_overlay_clicked)

        self.plot_hist = MplWidget(); self.tabs.addTab(self.plot_hist, "Histogram")
        self.plot_cum  = MplWidget(); self.tabs.addTab(self.plot_cum,  "Cumulative")

        right_layout.addWidget(self.tabs, 1)  # stretch=1 — займе весь простір

        self.stats_label = QLabel("—")
        self.table = QTableWidget(0,1); self.table.setHorizontalHeaderLabels(["diameter (µm)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setMinimumHeight(150)
        right_layout.addWidget(self.stats_label, 0)
        right_layout.addWidget(self.table, 0)

        # Splitter
        splitter = QSplitter(); splitter.addWidget(left); splitter.addWidget(right)
        splitter.setStretchFactor(0,0); splitter.setStretchFactor(1,1)
        splitter.setChildrenCollapsible(False)

        # Root
        root = QVBoxLayout(self)
        root.addLayout(topbar)
        root.addWidget(splitter, 1)  # stretch=1 — без «шапки»

        self.add_actions()

    def add_actions(self):
        act_open = QAction(self); act_open.setShortcut("Ctrl+O"); act_open.triggered.connect(self.open_image); self.addAction(act_open)
        act_run  = QAction(self); act_run.setShortcut("Ctrl+R"); act_run.triggered.connect(self.run_analysis); self.addAction(act_run)

    # ---------- Events / helpers ----------

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open SEM image", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)")
        if not path: return
        self.image_path = Path(path)
        try:
            self.gray = imread_gray(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read image:\n{e}")
            return
        self.um_per_px_from_meta = scale_from_metadata(path)
        txt = f"Scale: from meta {self.um_per_px_from_meta:.6f} µm/px" if self.um_per_px_from_meta else "Scale: — (set µm/px or read metadata)"
        self.meta_label.setText(txt)

        self.view_original.set_image(cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR))
        self.tabs.setCurrentWidget(self.view_original)

    def read_scale_meta(self):
        if not self.image_path:
            QMessageBox.information(self, "Info", "Open an image first."); return
        val = scale_from_metadata(str(self.image_path))
        if val is None:
            QMessageBox.warning(self, "Meta", "No usable TIFF metadata found."); return
        self.um_per_px_from_meta = val; self.sb_scale.setValue(round(val, 6))
        self.meta_label.setText(f"Scale: from meta {val:.6f} µm/px (also set in control)")

    def current_params(self) -> Params:
        scale = self.sb_scale.value()
        if scale <= 0 and self.um_per_px_from_meta: scale = self.um_per_px_from_meta
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
        )

    def run_analysis(self):
        if self.gray is None or self.image_path is None:
            QMessageBox.information(self, "Info", "Open an image first."); return
        P = self.current_params()
        if P.scale_um_per_px is None or P.scale_um_per_px <= 0:
            QMessageBox.warning(self, "Scale", "Set Scale (µm/px) or read from metadata."); return
        try:
            roi_mask = make_roi_mask(self.gray.shape, P.exclude_top, P.exclude_bottom, P.exclude_left, P.exclude_right)
            self.lev = preprocess(self.gray, P.clahe_clip, P.tophat_um, P.scale_um_per_px, P.level_strength)
            bwB, bwD = threshold_pair(self.lev, roi_mask, method=P.thr_method, block_size=P.block_size, C=P.block_C)
            kB = count_reasonable_components(bwB, P.scale_um_per_px, P.min_d_um, P.max_d_um)
            kD = count_reasonable_components(bwD, P.scale_um_per_px, P.min_d_um, P.max_d_um)
            bw = bwB if kB >= kD else bwD
            self.thr_raw = bw.copy()

            bw_m = morph_close(bw, P.closing_um, P.scale_um_per_px)
            bw_m = morph_open(bw_m, P.open_um, P.scale_um_per_px)
            bw_m = fill_small_holes(bw_m, 0.6)
            if P.split_touching:
                bw_m = split_touching_watershed(bw_m, P.scale_um_per_px, P.min_neck_um, P.min_seg_d_um)
            self.thr_proc = bw_m

            self.results = measure_components(bw_m, P.min_d_um, P.max_d_um, P.min_circ, P.scale_um_per_px, self.lev, P.min_rel_contrast)
            self.excluded_idx.clear()
            self._rebuild_overlay_and_stats()
            self.render_previews(); self.update_stats_table()
            self.tabs.setCurrentWidget(self.view_overlay)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing failed:\n{e}")

    # ---------- Exclusions ----------

    def on_toggle_remove(self, on: bool):
        self.remove_mode = on
        if on: self.tabs.setCurrentWidget(self.view_overlay)

    def on_clear_exclusions(self):
        self.excluded_idx.clear()
        self._rebuild_overlay_and_stats(); self.render_previews(); self.update_stats_table()

    def on_overlay_clicked(self, x: int, y: int):
        if not self.remove_mode or self.overlay_img is None or not self.results:
            return
        hit = None
        for i, (cnt, _, _) in enumerate(self.results):
            if cv2.pointPolygonTest(cnt, (float(x), float(y)), measureDist=False) >= 0:
                hit = i; break
        if hit is None: return
        if hit in self.excluded_idx: self.excluded_idx.remove(hit)
        else: self.excluded_idx.add(hit)
        self._rebuild_overlay_and_stats(); self.render_previews(); self.update_stats_table()

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

    def render_previews(self):
        if self.gray is not None:
            self.view_original.set_image(cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR))
        if self.lev is not None:
            self.view_leveled.set_image(cv2.cvtColor(self.lev, cv2.COLOR_GRAY2BGR))
        thr_to_show = self.thr_proc if self.thr_proc is not None else self.thr_raw
        if thr_to_show is not None:
            self.view_thresh.set_image(thr_to_show)
        if self.overlay_img is not None:
            self.view_overlay.set_image(self.overlay_img)
        self.plot_hist.plot_hist(self.diams_um, self.stats, "PSD Histogram")
        self.plot_cum.plot_cum(self.diams_um, self.stats)

    def update_stats_table(self):
        rows: list[tuple[float, bool]] = []
        for i, (_cnt, d, _c) in enumerate(self.results):
            if i not in self.excluded_idx: rows.append((d, False))
        for i, (_cnt, d, _c) in enumerate(self.results):
            if i in self.excluded_idx: rows.append((d, True))
        if rows:
            st = self.stats
            self.stats_label.setText(
                f"Particles: {st['particles']} | D10={st['D10']:.3f} µm | D50={st['D50']:.3f} µm | "
                f"D90={st['D90']:.3f} µm | mean={st['mean']:.3f} µm | std={st['std']:.3f} µm"
            )
            self.table.setRowCount(len(rows))
            for i, (v, excl) in enumerate(rows):
                cell = QTableWidgetItem(f"{v:.6f}")
                if excl:
                    cell.setForeground(Qt.gray); f = cell.font(); f.setItalic(True); cell.setFont(f)
                self.table.setItem(i, 0, cell)
        else:
            self.stats_label.setText("No particles accepted. Adjust parameters.")
            self.table.setRowCount(0)

    def export_csv(self):
        if not self.results:
            QMessageBox.information(self, "CSV", "Run analysis first."); return
        active = [d for i, (_c, d, _ci) in enumerate(self.results) if i not in self.excluded_idx]
        if not active:
            QMessageBox.information(self, "CSV", "No active particles to export."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save diameters CSV", "diameters.csv", "CSV (*.csv)")
        if not path: return
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["diameter_um"])
                for d in active: w.writerow([f"{d:.6f}"])
        except Exception as e:
            QMessageBox.critical(self, "CSV", f"Failed to save: {e}")

    def save_overlay(self):
        if self.overlay_img is None:
            QMessageBox.information(self, "Overlay", "Run analysis first."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save overlay image", "overlay.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if not path: return
        try:
            bgr = cv2.cvtColor(self.overlay_img, cv2.COLOR_RGB2BGR) if self.overlay_img.shape[2] == 3 else self.overlay_img
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
