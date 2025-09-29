# sem_psd_gui.py
# SEM particle-size GUI (PySide6) with:
# - right "Results" bar (user-selectable outputs),
# - extended metrics & ISO/CI (requires sem_psd_addons.py),
# - sortable tables, horizontal scroll & tooltips,
# - Welcome screen (drop target),
# - exclusions ("Remove blobs") affect all results & exports.

from __future__ import annotations
import sys, math, csv, textwrap
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

from PySide6.QtCore import (
    Qt, Signal, QObject, QThread, QEvent, QStandardPaths, QSettings, QRect
)
from PySide6.QtGui import QPixmap, QAction, QTransform, QPainter, QImage, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QWidget, QFileDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFormLayout, QDoubleSpinBox, QSpinBox, QTabWidget, QComboBox, QCheckBox,
    QMessageBox, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QProgressBar, QGroupBox,
    QProxyStyle, QStyle, QToolButton, QMenu, QStackedWidget, QFrame,
    QInputDialog, QAbstractItemView
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---- import core ----
from sem_psd_core import (
    imread_gray, scale_from_metadata,
    make_roi_mask, preprocess, threshold_pair, morph_open, morph_close, fill_small_holes,
    split_touching_watershed, count_reasonable_components, measure_components,
    stats_from_diams, detect_blobs_log, Params
)

# +++ NEW: add-ons
from sem_psd_addons import (
    resolve_scale_xy, enrich_results, iso9276_weighted_means,
    bootstrap_ci_percentile, bootstrap_ci_mean, write_csv_extended
)

# --- Numeric QTableWidgetItem: коректне сортування чисел і тултіп з повним значенням
class NumericItem(QTableWidgetItem):
    def __init__(self, value, fmt="{:.6f}"):
        if isinstance(value, (int, float, np.floating)):
            txt = fmt.format(float(value))
            super().__init__(txt)
            self._num = float(value)
        else:
            super().__init__(str(value))
            self._num = None
        self.setToolTip(self.text())

    def __lt__(self, other):
        if isinstance(other, QTableWidgetItem) and self.column() == other.column():
            a = self._num
            b = getattr(other, "_num", None)
            if a is not None and b is not None:
                return a < b
        return super().__lt__(other)

# ---------------- GUI helpers ----------------

def np_to_qpix(img: np.ndarray) -> QPixmap:
    if img.ndim == 2:
        qimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        qimg = img
    h, w, _ = qimg.shape
    qimg = cv2.cvtColor(qimg, cv2.COLOR_BGR2RGB)
    qim = QImage(qimg.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qim)

# ---------------- LoG/Threshold worker ----------------

class Worker(QObject):
    finished = Signal(object)
    failed = Signal(object)
    def __init__(self, gray: np.ndarray, params: Params, image_path: Path):
        super().__init__(); self.gray = gray; self.P = params; self.image_path = image_path

    def run(self):
        try:
            roi_mask = make_roi_mask(self.gray.shape, self.P.exclude_top, self.P.exclude_bottom, self.P.exclude_left, self.P.exclude_right)
            scale = self.P.scale_um_per_px or 0.0
            if scale <= 0:
                raise ValueError("Scale (µm/px) is required.")
            lev = preprocess(self.gray, self.P.clahe_clip, self.P.tophat_um, scale, self.P.level_strength)

            if self.P.analysis_mode == "log":
                results = detect_blobs_log(self.gray, lev, scale,
                                           self.P.min_d_um, self.P.max_d_um,
                                           self.P.log_threshold_rel, self.P.log_minsep_um,
                                           roi_mask, self.P.min_rel_contrast)
                self.finished.emit({"lev": lev, "thr_raw": None, "thr_proc": None, "results": results}); return

            bwB, bwD = threshold_pair(lev, roi_mask, method=self.P.thr_method, block_size=self.P.block_size, C=self.P.block_C)
            kB = count_reasonable_components(bwB, scale, self.P.min_d_um, self.P.max_d_um)
            kD = count_reasonable_components(bwD, scale, self.P.min_d_um, self.P.max_d_um)
            bw = bwB if kB >= kD else bwD; thr_raw = bw.copy()
            bw = morph_close(bw, self.P.closing_um, scale)
            bw = morph_open(bw, self.P.open_um, scale)
            bw = fill_small_holes(bw, 0.6)
            if self.P.split_touching:
                bw = split_touching_watershed(bw, scale, self.P.min_neck_um, self.P.min_seg_d_um)
            results = measure_components(bw, self.P.min_d_um, self.P.max_d_um, self.P.min_circ, scale, lev, self.P.min_rel_contrast)
            self.finished.emit({"lev": lev, "thr_raw": thr_raw, "thr_proc": bw, "results": results})
        except Exception as e:
            self.failed.emit(str(e))

# ---------------- Tiny plotting widget ----------------

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
                ax.axvline(st.get(name, 0.0), linestyle="--", label=f"{name}={st.get(name,0):.2f} {unit}")
            ax.set_xlabel(f"Particle diameter ({unit})"); ax.set_ylabel("Count"); ax.set_title(title)
            ax.grid(True); ax.legend()
        self.canvas.draw()

    def plot_cum(self, d_vals: np.ndarray, st: dict, unit: str):
        self.fig.clear(); ax = self.fig.add_subplot(111)
        if d_vals.size:
            s = np.sort(d_vals); cum = np.arange(1, s.size+1)/s.size*100.0
            ax.plot(s, cum)
            for name in ("D10","D50","D90"):
                ax.axvline(st.get(name, 0.0), linestyle="--", label=f"{name}={st.get(name,0):.2f} {unit}")
            ax.set_xlabel(f"Particle diameter ({unit})"); ax.set_ylabel("Cumulative %"); ax.set_title("Cumulative PSD")
            ax.grid(True); ax.legend()
        self.canvas.draw()

# ---------------- Image view ----------------

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

# ---------------- Main Window ----------------

class MainWindow(QWidget):
    auto_nm_threshold_um_per_px = 0.10
    label_scale = 1.1

    RECENT_MAX = 10
    SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SEM PSD (Particles Analysis)")
        self.resize(1500, 900)

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

        self.row_to_result: list[int] = []
        self.hover_idx: int | None = None
        self.sort_mode = "natural"

        self.recent_files: list[str] = []

        # Results-bar selections
        self.sel_show_overlay = True
        self.sel_show_threshold = True
        self.sel_show_hist = True
        self.sel_show_cum = True
        self.sel_show_table = True
        self.sel_show_ext_table = False
        self.sel_show_stats_plus = False

        # Extended analytics
        self.ext_rows = []
        self.weighted_stats = {}
        self.boot_ci = {}

        # NEW: чи вже був запуск аналізу (щоб не показувати «плейсхолдер» на старті)
        self._ever_ran = False

        self.build_ui()
        self._init_drop_targets()
        self._create_drop_hint()

        self.recent_files = self._load_recent_files()
        self._rebuild_recent_menu()

    # ---------------- UI ----------------
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
        self.menu_recent = menu.addMenu("Recent")
        self.btn_open.setMenu(menu)

        self.meta_label = QLabel("Scale: —"); self.meta_label.setStyleSheet("color:#666")
        self.progress = QProgressBar(); self.progress.setRange(0,0); self.progress.setVisible(False); self.progress.setFixedHeight(8); self.progress.setTextVisible(False)

        topbar.addWidget(self.btn_open); topbar.addWidget(self.meta_label); topbar.addStretch(1); topbar.addWidget(self.progress)

        # Left control panel
        left = QWidget(); form = QFormLayout(left)

        self.sb_scale = QDoubleSpinBox(); self.sb_scale.setRange(1e-6, 1000.0); self.sb_scale.setDecimals(6); self.sb_scale.setValue(0.0)
        self.sb_scale.setToolTip("Micrometers per pixel. If 0, will try to read from TIFF metadata.")
        self.sb_scale.valueChanged.connect(self.on_scale_changed)

        self.cb_autoscale = QPushButton("Read scale from metadata"); self.cb_autoscale.clicked.connect(self.read_scale_meta)

        self.cb_mode = QComboBox(); self.cb_mode.addItems(["Contours (threshold)", "Nano (LoG)"])
        self.cb_mode.currentIndexChanged.connect(self.update_param_visibility)
        self.btn_help = QPushButton("Help"); self.btn_help.clicked.connect(self.show_help)

        mode_row = QWidget(); hl = QHBoxLayout(mode_row); hl.setContentsMargins(0,0,0,0); hl.addWidget(self.cb_mode); hl.addWidget(self.btn_help)
        form.addRow("Scale (µm/px)", self.sb_scale); form.addRow(" ", self.cb_autoscale); form.addRow("Analysis mode", mode_row)

        self.cb_auto_units = QCheckBox("Auto-detect display units from scale")
        self.cb_auto_units.setChecked(True); self.cb_auto_units.toggled.connect(self.on_auto_units_toggled)
        form.addRow(self.cb_auto_units)

        self.cb_units = QComboBox(); self.cb_units.addItems(["µm", "nm"]); self.cb_units.setEnabled(False)
        self.cb_units.currentIndexChanged.connect(self.on_units_changed)
        form.addRow("Display units", self.cb_units)

        # ROI
        def mkSpin(val): s=QDoubleSpinBox(); s.setRange(0,0.9); s.setSingleStep(0.01); s.setDecimals(3); s.setValue(val); return s
        self.sb_top, self.sb_bottom, self.sb_left, self.sb_right = mkSpin(0.02), mkSpin(0.22), mkSpin(0.0), mkSpin(0.0)
        row_roi = QWidget(); lroi = QFormLayout(row_roi); lroi.addRow("Top", self.sb_top); lroi.addRow("Bottom", self.sb_bottom); lroi.addRow("Left", self.sb_left); lroi.addRow("Right", self.sb_right)
        form.addRow(QLabel("ROI exclude ratios"), row_roi)

        # General
        self.sb_min_d = QDoubleSpinBox(); self.sb_max_d = QDoubleSpinBox(); self.sb_min_rc = QDoubleSpinBox()
        self.sb_min_d.setRange(0.0, 1000.0); self.sb_min_d.setDecimals(3); self.sb_min_d.setValue(0.01)
        self.sb_max_d.setRange(0.0, 10000.0); self.sb_max_d.setDecimals(3); self.sb_max_d.setValue(10.0)
        self.sb_min_rc.setRange(0.0,1.0); self.sb_min_rc.setDecimals(2); self.sb_min_rc.setValue(0.15)
        form.addRow("Min diameter (µm)", self.sb_min_d); form.addRow("Max diameter (µm)", self.sb_max_d); form.addRow("Min rel. contrast", self.sb_min_rc)

        # Preprocess
        self.sb_clahe = QDoubleSpinBox(); self.sb_tophat = QDoubleSpinBox(); self.sb_level = QDoubleSpinBox()
        self.sb_clahe.setRange(0.0,10.0); self.sb_clahe.setDecimals(2); self.sb_clahe.setValue(2.0)
        self.sb_tophat.setRange(0.0,100.0); self.sb_tophat.setDecimals(3); self.sb_tophat.setValue(0.0)
        self.sb_level.setRange(0.0,1.5); self.sb_level.setDecimals(2); self.sb_level.setValue(0.3)
        form.addRow("CLAHE clip", self.sb_clahe); form.addRow("Top-hat radius (µm)", self.sb_tophat); form.addRow("Level strength", self.sb_level)

        # Threshold-only
        self.grp_thr = QGroupBox("Threshold-only"); lay_thr = QFormLayout(self.grp_thr)
        self.sb_min_circ = QDoubleSpinBox(); self.sb_min_circ.setRange(0.0, 1.0); self.sb_min_circ.setDecimals(3); self.sb_min_circ.setValue(0.10)
        self.sb_closing = QDoubleSpinBox(); self.sb_closing.setRange(0.0, 100.0); self.sb_closing.setDecimals(3); self.sb_closing.setValue(0.12)
        self.sb_open = QDoubleSpinBox(); self.sb_open.setRange(0.0, 100.0); self.sb_open.setDecimals(3); self.sb_open.setValue(0.08)
        self.cb_thr = QComboBox(); self.cb_thr.addItems(["otsu","adaptive"])
        self.sb_block = QSpinBox(); self.sb_block.setRange(3,999); self.sb_block.setValue(31)
        self.sb_C = QSpinBox(); self.sb_C.setRange(-255,255); self.sb_C.setValue(-10)
        self.cb_split = QCheckBox("Split touching (watershed)"); self.cb_split.setChecked(False)
        self.sb_neck = QDoubleSpinBox(); self.sb_neck.setRange(0.0,10.0); self.sb_neck.setDecimals(3); self.sb_neck.setValue(0.12)
        self.sb_seg = QDoubleSpinBox(); self.sb_seg.setRange(0.0,100.0); self.sb_seg.setDecimals(3); self.sb_seg.setValue(0.20)
        for label, widget in [
            ("Min circularity", self.sb_min_circ),
            ("Closing (µm)",    self.sb_closing),
            ("Opening (µm)",    self.sb_open),
            ("Threshold",       self.cb_thr),
            ("Adaptive block size", self.sb_block),
            ("Adaptive C",      self.sb_C),
            (None,              self.cb_split),
            ("Min neck (µm)",   self.sb_neck),
            ("Min segment d (µm)", self.sb_seg),
        ]:
            lay_thr.addRow(widget if label is None else label, widget)
        form.addRow(self.grp_thr)

        # LoG-only
        self.grp_log = QGroupBox("LoG-only"); lay_log = QFormLayout(self.grp_log)
        self.sb_log_thr = QDoubleSpinBox(); self.sb_log_thr.setRange(0.0, 1.0); self.sb_log_thr.setDecimals(3); self.sb_log_thr.setValue(0.030)
        self.sb_log_sep = QDoubleSpinBox(); self.sb_log_sep.setRange(0.0, 10.0); self.sb_log_sep.setDecimals(3); self.sb_log_sep.setValue(0.120)
        lay_log.addRow("LoG threshold (rel)", self.sb_log_thr); lay_log.addRow("LoG min separation (µm)", self.sb_log_sep)
        form.addRow(self.grp_log)

        # Actions
        self.btn_autotune = QPushButton("Auto-tune"); self.btn_autotune.clicked.connect(self.autotune_params)
        self.btn_run = QPushButton("Run analysis"); self.btn_run.clicked.connect(self.run_analysis)
        self.btn_export_csv = QPushButton("Export CSV…"); self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_ext = QPushButton("Export Extended CSV…"); self.btn_export_ext.clicked.connect(self.export_extended_csv)
        self.btn_save_overlay = QPushButton("Save overlay…"); self.btn_save_overlay.clicked.connect(self.save_overlay)
        self.btn_toggle_remove = QPushButton("Remove blobs (click)"); self.btn_toggle_remove.setCheckable(True); self.btn_toggle_remove.toggled.connect(self.on_toggle_remove)
        self.btn_clear_excl = QPushButton("Clear exclusions"); self.btn_clear_excl.clicked.connect(self.on_clear_exclusions)
        for b in [self.btn_autotune, self.btn_run, self.btn_export_csv, self.btn_export_ext, self.btn_save_overlay, self.btn_toggle_remove, self.btn_clear_excl]:
            form.addRow(b)

        # -------- Central workspace (tabs) --------
        center = QWidget(); center_layout = QVBoxLayout(center); center_layout.setContentsMargins(0,0,0,0)
        self.tabs = QTabWidget(); self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.view_original = ImageView();  self.tabs.addTab(self.view_original, "Original")
        self.view_leveled  = ImageView();  self.tabs.addTab(self.view_leveled,  "Leveled")
        self.view_thresh   = ImageView();  self.tabs.addTab(self.view_thresh,   "Threshold")
        self.view_overlay  = ImageView();  self.tabs.addTab(self.view_overlay,  "Overlay")
        self.view_overlay.sig_clicked.connect(self.on_overlay_clicked)

        self.plot_hist = MplWidget(); self.tabs.addTab(self.plot_hist, "Histogram")
        self.plot_cum  = MplWidget(); self.tabs.addTab(self.plot_cum,  "Cumulative")

        # NEW: extended particles table
        self.table_ext = QTableWidget(0, 14)
        self.table_ext.setHorizontalHeaderLabels([
            "idx","d (µm)","circ","area (µm²)","perim (µm)","d_eq_area","d_eq_perim",
            "Feret min","Feret max","AR","ellipse min","ellipse max","roundness","solidity"
        ])
        self.table_ext.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table_ext.horizontalHeader().setSectionsMovable(True)
        self.table_ext.horizontalHeader().setMinimumSectionSize(90)
        self.table_ext.horizontalHeader().setDefaultSectionSize(160)
        self.table_ext.setSortingEnabled(True)
        self.table_ext.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table_ext.setTextElideMode(Qt.ElideRight)
        self.table_ext.setEditTriggers(QAbstractItemView.NoEditTriggers)  # <- заборонити редагування
        self.tabs.addTab(self.table_ext, "Particles+")

        center_layout.addWidget(self.tabs, 1)
        self.tabs.currentChanged.connect(self._update_hint_parent)
        self.tabs.currentChanged.connect(lambda _: self.render_overlay_with_highlight())

        ctrl_row = QHBoxLayout()
        self.stats_label = QLabel("—"); ctrl_row.addWidget(self.stats_label); ctrl_row.addStretch(1)
        self.sort_label = QLabel("Order:"); self.cb_sort = QComboBox(); self.cb_sort.addItems(["As scanned", "Asc", "Desc"])
        self.cb_sort.currentIndexChanged.connect(self.on_sort_changed); self.cb_sort.setMaximumWidth(180)
        ctrl_row.addWidget(self.sort_label); ctrl_row.addWidget(self.cb_sort)
        center_layout.addLayout(ctrl_row)

        # ---- Bottom one-column table (diameters) ----
        self.table = QTableWidget(0, 1)
        self.table.setHorizontalHeaderLabels(["diameter (µm)"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        hh.setStretchLastSection(True)
        hh.setSectionsMovable(False)
        hh.setSectionsClickable(False)
        hh.setSortIndicatorShown(False)
        self.table.setSortingEnabled(False)  # керуємося лише Order
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # <- заборонити редагування

        vh = self.table.verticalHeader()
        vh.setVisible(True)
        vh.setDefaultAlignment(Qt.AlignCenter)
        vh.setSectionResizeMode(QHeaderView.Fixed)
        vh.setDefaultSectionSize(22)

        self.table.setMinimumHeight(140)
        self.table.setMaximumHeight(260)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.ElideRight)
        self.table.setMouseTracking(True)
        self.table.cellEntered.connect(self.on_table_cell_entered)
        self.table.viewport().installEventFilter(self)
        center_layout.addWidget(self.table, 0)

        # -------- Right Results bar --------
        right_bar = QWidget(); rb = QVBoxLayout(right_bar); rb.setContentsMargins(8,8,8,8)
        rb.addWidget(QLabel("<b>Results</b>"))
        self.cb_rb_overlay   = QCheckBox("Overlay");   self.cb_rb_overlay.setChecked(self.sel_show_overlay)
        self.cb_rb_threshold = QCheckBox("Threshold"); self.cb_rb_threshold.setChecked(self.sel_show_threshold)
        self.cb_rb_hist      = QCheckBox("Histogram"); self.cb_rb_hist.setChecked(self.sel_show_hist)
        self.cb_rb_cum       = QCheckBox("Cumulative");self.cb_rb_cum.setChecked(self.sel_show_cum)
        self.cb_rb_table     = QCheckBox("Table");     self.cb_rb_table.setChecked(self.sel_show_table)
        self.cb_rb_ext       = QCheckBox("Particles+");self.cb_rb_ext.setChecked(self.sel_show_ext_table)
        self.cb_rb_stats     = QCheckBox("Stats+ (ISO & CI)"); self.cb_rb_stats.setChecked(self.sel_show_stats_plus)
        for cb in [self.cb_rb_overlay, self.cb_rb_threshold, self.cb_rb_hist, self.cb_rb_cum, self.cb_rb_table, self.cb_rb_ext, self.cb_rb_stats]:
            cb.stateChanged.connect(self.on_results_bar_changed); rb.addWidget(cb)

        rb.addStretch(1)
        self.stats_plus = QLabel("Stats+: —")
        self.stats_plus.setWordWrap(True)
        self.stats_plus.setFrameShape(QFrame.StyledPanel)
        rb.addWidget(self.stats_plus)

        self.right_bar = right_bar
        self.right_bar.setVisible(False)

        # -------- Welcome screen --------
        self.welcome = QWidget()
        wl = QVBoxLayout(self.welcome)
        lbl = QLabel(
            "<h2 style='margin:0'>SEM PSD (Particle Analysis)</h2>"
            "<p>Open a SEM image (Ctrl+O) or load a test sample:</p>"
            "<ul><li><b>Ctrl+1</b> — samples/blobs.tif</li>"
            "<li><b>Ctrl+2</b> — samples/granular.tif</li></ul>"
            "<p>Pick the analysis mode (Contours / Nano [LoG]) and press <b>Run analysis</b> (Ctrl+R).</p>"
        )
        lbl.setWordWrap(True); lbl.setAlignment(Qt.AlignCenter)
        wl.addStretch(1); wl.addWidget(lbl, alignment=Qt.AlignCenter); wl.addStretch(1)

        # Центральний стек (Welcome ↔ Робоча зона)
        self.center_stack = QStackedWidget()
        self.center_stack.addWidget(self.welcome)  # 0
        self.center_stack.addWidget(center)        # 1
        self.center_stack.setCurrentIndex(0)

        # -------- Splitter
        splitter = QSplitter(); splitter.addWidget(left); splitter.addWidget(self.center_stack); splitter.addWidget(right_bar)
        splitter.setStretchFactor(0,0); splitter.setStretchFactor(1,1); splitter.setStretchFactor(2,0)
        splitter.setSizes([360, 900, 260])
        splitter.setChildrenCollapsible(False)

        root = QVBoxLayout(self); root.addLayout(topbar); root.addWidget(splitter, 1)

        self.add_actions()
        self.update_param_visibility()
        self._update_results_bar_visibility()

    def _fill_table_safely(self, table, row_count, fill_rows_fn):
        was_sorting = table.isSortingEnabled()
        if was_sorting:
            table.setSortingEnabled(False)
        table.setRowCount(row_count)
        if row_count > 0:
            fill_rows_fn()
        if was_sorting:
            table.setSortingEnabled(True)

    # ---------- Results bar logic ----------
    def _update_results_bar_visibility(self):
        idx_overlay  = self.tabs.indexOf(self.view_overlay)
        idx_thresh   = self.tabs.indexOf(self.view_thresh)
        idx_hist     = self.tabs.indexOf(self.plot_hist)
        idx_cum      = self.tabs.indexOf(self.plot_cum)
        idx_ext      = self.tabs.indexOf(self.table_ext)

        if idx_overlay >= 0: self.tabs.setTabVisible(idx_overlay, self.sel_show_overlay)
        if idx_thresh  >= 0: self.tabs.setTabVisible(idx_thresh,  self.sel_show_threshold and self.cb_mode.currentIndex()==0)
        if idx_hist    >= 0: self.tabs.setTabVisible(idx_hist,    self.sel_show_hist)
        if idx_cum     >= 0: self.tabs.setTabVisible(idx_cum,     self.sel_show_cum)
        if idx_ext     >= 0: self.tabs.setTabVisible(idx_ext,     self.sel_show_ext_table)

        self.table.setVisible(self.sel_show_table)
        self.stats_label.setVisible(self.sel_show_table)
        self.sort_label.setVisible(self.sel_show_table)
        self.cb_sort.setVisible(self.sel_show_table)

        self.stats_plus.setVisible(self.sel_show_stats_plus)

    def on_results_bar_changed(self, *_):
        self.sel_show_overlay = self.cb_rb_overlay.isChecked()
        self.sel_show_threshold = self.cb_rb_threshold.isChecked()
        self.sel_show_hist = self.cb_rb_hist.isChecked()
        self.sel_show_cum = self.cb_rb_cum.isChecked()
        self.sel_show_table = self.cb_rb_table.isChecked()
        self.sel_show_ext_table = self.cb_rb_ext.isChecked()
        self.sel_show_stats_plus = self.cb_rb_stats.isChecked()

        if self.results:
            self._compute_ext_analytics()

        self._update_results_bar_visibility()
        self.render_previews()
        self.update_stats_table()
        self.update_ext_table()
        self.update_stats_plus_panel()

    # ---------- DnD targets ----------
    def _image_views(self):
        return [self.view_original, self.view_leveled, self.view_thresh, self.view_overlay]

    def _init_drop_targets(self):
        for v in self._image_views():
            vp = v.viewport()
            vp.setAcceptDrops(True)
            vp.installEventFilter(self)
        self.welcome.setAcceptDrops(True)
        self.welcome.installEventFilter(self)

    def _current_drop_host(self):
        if self.center_stack.currentIndex() == 0:
            return self.welcome
        cur = self.tabs.currentWidget()
        if isinstance(cur, ImageView):
            return cur.viewport()
        return None

    def _create_drop_hint(self):
        host = self._current_drop_host() or self
        self.drop_hint = QLabel(host)
        self.drop_hint.setText("Drop image to open")
        self.drop_hint.setAlignment(Qt.AlignCenter)
        self.drop_hint.setStyleSheet(
            "QLabel { background-color: rgba(0,0,0,0.55); color: white;"
            " border: 2px dashed #FFD54F; border-radius: 12px; padding: 14px 18px; font-size: 16px; }"
        )
        self.drop_hint.setVisible(False)
        self._position_drop_hint()

    def _update_hint_parent(self):
        host = self._current_drop_host()
        if host is None:
            self.drop_hint.setParent(self)
            self.drop_hint.hide()
            return
        self.drop_hint.setParent(host)
        self.drop_hint.hide()
        self._position_drop_hint()

    def _position_drop_hint(self):
        host = self._current_drop_host()
        if host is None or self.drop_hint is None:
            return
        r = host.rect()
        w, h = 280, 64
        x = int((r.width() - w) / 2)
        y = int((r.height() - h) / 2)
        self.drop_hint.setGeometry(QRect(x, y, w, h))

    def _show_drop_hint(self, on: bool):
        if self.drop_hint:
            self.drop_hint.setVisible(bool(on))

    # ---------- Event filter ----------
    def eventFilter(self, obj, event):
        drop_hosts = [v.viewport() for v in self._image_views()] + [self.welcome]
        if obj in drop_hosts:
            if event.type() in (QEvent.DragEnter, QEvent.DragMove):
                md = event.mimeData()
                if md and md.hasUrls():
                    for u in md.urls():
                        try:
                            p = Path(u.toLocalFile())
                        except Exception:
                            continue
                        if self._is_supported_file(p):
                            if obj is self._current_drop_host():
                                event.acceptProposedAction()
                                self._show_drop_hint(True)
                            else:
                                event.ignore()
                                self._show_drop_hint(False)
                            return True
                self._show_drop_hint(False)
                return False

            if event.type() in (QEvent.DragLeave, QEvent.Leave, QEvent.Hide):
                self._show_drop_hint(False)
                return False

            if event.type() == QEvent.Drop:
                self._show_drop_hint(False)
                if obj is not self._current_drop_host():
                    event.ignore(); return False
                md = event.mimeData()
                if md and md.hasUrls():
                    for u in md.urls():
                        try:
                            p = Path(u.toLocalFile())
                        except Exception:
                            continue
                        if self._is_supported_file(p):
                            self._open_path(p)
                            event.acceptProposedAction()
                            return True
                return False

            if event.type() == QEvent.Resize:
                self._position_drop_hint()

        if obj is getattr(self, "table", None).viewport():
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

    # ---------- Utils ----------
    def _is_supported_file(self, p: Path) -> bool:
        return p.suffix.lower() in self.SUPPORTED_EXTS and p.exists()

    def _settings(self) -> QSettings:
        return QSettings("ACDC", "SEM_PSD")

    def _load_recent_files(self) -> list[str]:
        s = self._settings()
        raw = s.value("recent_files", [])
        if isinstance(raw, str): lst = [raw]
        elif isinstance(raw, (list, tuple)): lst = list(raw)
        else: lst = []
        out, seen = [], set()
        for x in lst:
            try: p = Path(x)
            except Exception: continue
            if not self._is_supported_file(p): continue
            if x not in seen:
                out.append(x); seen.add(x)
        return out[: self.RECENT_MAX]

    def _save_recent_files(self):
        s = self._settings()
        s.setValue("recent_files", self.recent_files)

    def _add_to_recent(self, p: Path):
        if not self._is_supported_file(p): return
        sp = str(p)
        self.recent_files = [x for x in self.recent_files if x != sp]
        self.recent_files.insert(0, sp)
        self.recent_files = self.recent_files[: self.RECENT_MAX]
        self._save_recent_files()
        self._rebuild_recent_menu()

    def _clear_recent(self):
        self.recent_files = []
        self._save_recent_files()
        self._rebuild_recent_menu()

    def _open_recent(self, sp: str):
        p = Path(sp)
        if self._is_supported_file(p):
            self._open_path(p)
        else:
            QMessageBox.warning(self, "Recent", f"File not found or unsupported:\n{sp}")
            self.recent_files = [x for x in self.recent_files if x != sp]
            self._save_recent_files()
            self._rebuild_recent_menu()

    def _rebuild_recent_menu(self):
        if not hasattr(self, "menu_recent") or self.menu_recent is None:
            return
        self.menu_recent.clear()
        if not self.recent_files:
            act = self.menu_recent.addAction("(empty)")
            act.setEnabled(False)
        else:
            for sp in self.recent_files:
                act = self.menu_recent.addAction(sp)
                act.triggered.connect(lambda _, s=sp: self._open_recent(s))
        self.menu_recent.addSeparator()
        clear_act = self.menu_recent.addAction("Clear recent")
        clear_act.triggered.connect(self._clear_recent)

    # ---------- Sort control ----------
    def on_sort_changed(self, idx: int):
        modes = {0: "natural", 1: "asc", 2: "desc"}
        self.sort_mode = modes.get(idx, "natural")
        self.update_stats_table()

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

        # NEW: find by index Ctrl+F
        self.act_find = QAction(self, triggered=self.find_particle)
        self.act_find.setShortcut(QKeySequence.Find)
        self.act_find.setShortcutContext(Qt.ApplicationShortcut)
        self.addAction(self.act_find)

    # ---------- Help ----------
    def show_help(self):
        txt = textwrap.dedent("""
        ### Modes
        • **Nano (LoG)** — multi-scale Laplacian of Gaussian with σ² normalization.
        • **Contours (threshold)** — thresholding + morphology + optional watershed.

        ### Results bar
        • Overlay / Threshold / Histogram / Cumulative / Table / Particles+ / Stats+ (ISO & CI).
        """).strip()
        QMessageBox.information(self, "Help", txt)

    # ---------- Visibility ----------
    def update_param_visibility(self):
        is_log = (self.cb_mode.currentIndex() == 1)
        self.grp_log.setVisible(is_log)
        self.grp_thr.setVisible(not is_log)
        self.tabs.setTabEnabled(self.tabs.indexOf(self.view_thresh), not is_log)
        self._update_results_bar_visibility()

    # ---------- Unit helpers ----------
    def unit_label(self) -> str:
        return "nm" if self.cb_units.currentIndex() == 1 else "µm"

    def unit_factor(self) -> float:
        return 1000.0 if self.unit_label() == "nm" else 1.0

    def scale_stats_for_display(self, st: dict) -> dict:
        if not st: return {}
        out = dict(st); factor = self.unit_factor()
        for k in ("D10","D50","D90","mean","std","min","max"):
            if k in out: out[k] = float(out[k]) * factor
        return out

    def decide_units_from_scale(self, scale_um_per_px: float | None) -> str:
        if scale_um_per_px is None or scale_um_per_px <= 0: return self.unit_label()
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
        if not self.cb_auto_units.isChecked(): return
        scale = self.sb_scale.value()
        if (not scale or scale <= 0) and self.um_per_px_from_meta:
            scale = self.um_per_px_from_meta
        unit = self.decide_units_from_scale(scale)
        self.set_display_unit(unit)
        self.render_previews(); self.update_stats_table()

    def on_auto_units_toggled(self, on: bool):
        self.update_units_lock()
        if on: self.maybe_update_auto_units()

    def on_scale_changed(self, *_):
        self.maybe_update_auto_units()

    def on_units_changed(self, *_):
        self.render_previews(); self.update_stats_table(); self.update_ext_table(); self.update_stats_plus_panel()

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

        if self.right_bar is not None:
            self.right_bar.setVisible(True)

        self.view_original.set_image(cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR))
        self.tabs.setCurrentWidget(self.view_original)
        self.center_stack.setCurrentIndex(1)
        if hasattr(self, "right_bar") and self.right_bar is not None:
            self.right_bar.setVisible(True)
        self._update_results_bar_visibility()
        self._add_to_recent(p)

    def open_image(self):
        if self._open_in_progress: return
        self._open_in_progress = True
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Open SEM image", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)")
            if not path: return
            self._open_path(Path(path))
        finally:
            self._open_in_progress = False

    def read_scale_meta(self):
        """Read µm/px from TIFF metadata and sync the UI."""
        if not self.image_path:
            QMessageBox.information(self, "Info", "Open an image first."); return
        try:
            val = scale_from_metadata(str(self.image_path))
        except Exception as e:
            QMessageBox.warning(self, "Meta", f"Failed to read metadata:\n{e}")
            return
        if val is None or val <= 0:
            QMessageBox.warning(self, "Meta", "No usable TIFF metadata found.")
            return
        self.um_per_px_from_meta = float(val)
        self.sb_scale.setValue(round(self.um_per_px_from_meta, 6))
        self.meta_label.setText(f"Scale: from meta {self.um_per_px_from_meta:.6f} µm/px (also set in control)")
        self.maybe_update_auto_units()
        self.render_previews()
        self.update_stats_table()
        self.update_ext_table()
        self.update_stats_plus_panel()

    def open_sample(self, filename: str):
        samples_dir = Path(__file__).resolve().parent.parent / "samples"
        p = samples_dir / filename
        if not p.exists():
            QMessageBox.warning(self, "Sample not found", f"File not found:\n{p}")
            return
        self._open_path(p)

    # ---------- Active results helper ----------
    def _active_results(self):
        return [t for i, t in enumerate(self.results) if i not in self.excluded_idx]

    # ---------- Results update helpers ----------
    def _compute_ext_analytics(self):
        act = self._active_results()
        if not act:
            self.ext_rows = []; self.weighted_stats = {}; self.boot_ci = {}; return
        scale = self.sb_scale.value() or self.um_per_px_from_meta or 0.0
        umx, umy = scale, scale
        self.ext_rows = enrich_results(act, umx, umy, 0.0)
        diams_um = np.array([d for (_c, d, _ci) in act], float)
        self.weighted_stats = iso9276_weighted_means(diams_um)
        self.boot_ci = {
            "D10": bootstrap_ci_percentile(diams_um, q=10, n=5000, alpha=0.05, seed=0),
            "D50": bootstrap_ci_percentile(diams_um, q=50, n=5000, alpha=0.05, seed=0),
            "D90": bootstrap_ci_percentile(diams_um, q=90, n=5000, alpha=0.05, seed=0),
            "mean": bootstrap_ci_mean(diams_um, n=5000, alpha=0.05, seed=0),
        }

    # ---------- Actions ----------
    def export_extended_csv(self):
        if not self.results:
            QMessageBox.information(self, "CSV+", "Run analysis first."); return
        if not self.ext_rows:
            self._compute_ext_analytics()
        unit = self.unit_label()
        factor = self.unit_factor()
        rows = []
        from dataclasses import replace
        for r in self.ext_rows:
            rows.append(replace(
                r,
                d_um=r.d_um*factor,
                area_um2=r.area_um2*(factor**2),
                perimeter_um=r.perimeter_um*factor,
                d_eq_area_um=r.d_eq_area_um*factor,
                d_eq_perim_um=r.d_eq_perim_um*factor,
                feret_min_um=r.feret_min_um*factor,
                feret_max_um=r.feret_max_um*factor,
                ellipse_minor_um=r.ellipse_minor_um*factor,
                ellipse_major_um=r.ellipse_major_um*factor,
            ))
        path, _ = QFileDialog.getSaveFileName(self, "Save extended CSV", f"particles_extended_{unit}.csv", "CSV (*.csv)")
        if not path: return
        try:
            write_csv_extended(path, rows, unit=unit)
            QMessageBox.information(self, "CSV+", f"Saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "CSV+", f"Failed to save: {e}")

    # ---------- Auto-tune (placeholder) ----------
    def autotune_params(self):
        QMessageBox.information(self, "Auto-tune", "Auto-tune keep it without changes for this version.")

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
        self._ever_ran = True  # вже був запуск
        self.lev = out.get("lev")
        self.thr_raw = out.get("thr_raw")
        self.thr_proc = out.get("thr_proc")
        self.results = out.get("results", [])
        self.excluded_idx.clear()
        self.hover_idx = None

        self._rebuild_overlay_and_stats()
        self._compute_ext_analytics()
        self.center_stack.setCurrentIndex(1)

        self.render_previews()
        self.update_stats_table()
        self.update_ext_table()
        self.update_stats_plus_panel()

        if self.sel_show_overlay:
            self.tabs.setCurrentWidget(self.view_overlay)

    def on_worker_failed(self, msg: str):
        QMessageBox.critical(self, "Error", f"Processing failed:\n{msg}")

    # ---------- Exclusions ----------
    def on_toggle_remove(self, on: bool):
        self.remove_mode = on
        if on: self.tabs.setCurrentWidget(self.view_overlay)

    def on_clear_exclusions(self):
        self.excluded_idx.clear()
        self.hover_idx = None
        self._rebuild_overlay_and_stats(); self._compute_ext_analytics()
        self.render_previews(); self.update_stats_table(); self.update_ext_table(); self.update_stats_plus_panel()

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
        self._rebuild_overlay_and_stats(); self._compute_ext_analytics()
        self.render_previews(); self.update_stats_table(); self.update_ext_table(); self.update_stats_plus_panel()

    # ---------- Hover ↔ highlight ----------
    def on_table_cell_entered(self, row: int, col: int):
        if 0 <= row < len(self.row_to_result):
            ridx = self.row_to_result[row]
            self.hover_idx = ridx
        else:
            self.hover_idx = None
        self.render_overlay_with_highlight()

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
        self.diams_um = np.array(active_diams, float)  # only active!
        self.stats = stats_from_diams(self.diams_um)

    # ---------- Rendering / Table / Export ----------
    def render_overlay_with_highlight(self):
        if self.tabs.currentWidget() is not self.view_overlay:
            if self.sel_show_overlay and self.overlay_img is not None:
                self.view_overlay.set_image(self.overlay_img)
            return
        if self.overlay_img is None:
            return

        img = self.overlay_img.copy()
        if self.hover_idx is not None and 0 <= self.hover_idx < len(self.results):
            cnt, d_um, _ = self.results[self.hover_idx]
            base = img.copy()
            img = (img * 0.35).astype(np.uint8)
            mask = np.zeros(img.shape[:2], np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
            mask_dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 1)
            img[mask_dil > 0] = base[mask_dil > 0]
            fill = img.copy()
            cv2.drawContours(fill, [cnt], -1, (0, 255, 255), thickness=-1)
            cv2.addWeighted(fill, 0.35, img, 0.65, 0, dst=img)

            cv2.drawContours(img, [cnt], -1, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.drawContours(img, [cnt], -1, (0, 255, 255), 2, lineType=cv2.LINE_AA)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
                cv2.circle(img, (cx, cy), 3, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

                unit = self.unit_label(); factor = self.unit_factor(); val = d_um * factor
                txt_num = f"{val:.1f}"; txt_unit = f" {unit}"

                v = max(1e-3, self._overlay_view_scale()); S = getattr(self, "label_scale", 1.35)
                fs_num = max(0.42 * S, min(0.80 * S, (0.62 * S) / v))
                fs_unit = max(0.36 * S, min(0.72 * S, (0.50 * S) / v))
                th_text = max(1, min(3, int(round(1.6 * S / v))))
                pad_x = max(8, min(18, int(round(10 * S / v))))
                pad_y = max(6, min(12, int(round(7 * S / v))))
                th_box = max(1, min(2, int(round(1.2 * S / v))))

                (tn_w, tn_h), _ = cv2.getTextSize(txt_num, cv2.FONT_HERSHEY_SIMPLEX, fs_num, th_text)
                (tu_w, tu_h), _ = cv2.getTextSize(txt_unit, cv2.FONT_HERSHEY_SIMPLEX, fs_unit, th_text)
                tw = tn_w + tu_w; th = max(tn_h, tu_h)

                H, W = img.shape[:2]
                bx0, by0, bx1, by1 = self._best_label_box(cx, cy, tw, th, pad_x, pad_y, W, H, v)

                overlay_bg = img.copy()
                cv2.rectangle(overlay_bg, (bx0, by0), (bx1, by1), (0, 0, 0), -1)
                cv2.addWeighted(overlay_bg, 0.50, img, 0.50, 0, dst=img)
                cv2.rectangle(img, (bx0, by0), (bx1, by1), (0, 255, 255), th_box, lineType=cv2.LINE_AA)

                org = (bx0 + pad_x, by0 + pad_y + th)
                cv2.putText(img, txt_num, org, cv2.FONT_HERSHEY_SIMPLEX, fs_num, (0, 0, 0), th_text + 2, cv2.LINE_AA)
                cv2.putText(img, txt_num, org, cv2.FONT_HERSHEY_SIMPLEX, fs_num, (0, 255, 255), th_text, cv2.LINE_AA)
                org_unit = (org[0] + tn_w, org[1])
                cv2.putText(img, txt_unit, org_unit, cv2.FONT_HERSHEY_SIMPLEX, fs_unit, (0, 0, 0), th_text + 2, cv2.LINE_AA)
                cv2.putText(img, txt_unit, org_unit, cv2.FONT_HERSHEY_SIMPLEX, fs_unit, (0, 255, 255), th_text, cv2.LINE_AA)

        self.view_overlay.set_image(img)

    def render_previews(self):
        if self.gray is not None:
            self.view_original.set_image(cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR))
        if self.lev is not None:
            self.view_leveled.set_image(cv2.cvtColor(self.lev, cv2.COLOR_GRAY2BGR))
        thr_to_show = self.thr_proc if self.thr_proc is not None else self.thr_raw
        if thr_to_show is not None and self.cb_mode.currentIndex() == 0 and self.sel_show_threshold:
            self.view_thresh.set_image(thr_to_show)
        if self.sel_show_overlay and self.overlay_img is not None:
            self.render_overlay_with_highlight()

        factor = self.unit_factor(); unit = self.unit_label()
        d_disp = self.diams_um * factor
        st_disp = self.scale_stats_for_display(self.stats)
        if self.sel_show_hist:
            self.plot_hist.plot_hist(d_disp, st_disp, "PSD Histogram", unit)
        if self.sel_show_cum:
            self.plot_cum.plot_cum(d_disp, st_disp, unit)

    def _sorted_rows(self, rows: list[tuple[int, float, bool]]) -> list[tuple[int, float, bool]]:
        if self.sort_mode == "natural":
            act = [r for r in rows if not r[2]]
            exc = [r for r in rows if r[2]]
            return act + exc
        reverse = (self.sort_mode == "desc")
        return sorted(rows, key=lambda t: (t[1], t[2]), reverse=reverse)

    def update_stats_table(self):
        rows = [(i, d, (i in self.excluded_idx)) for i, (_c, d, _ci) in enumerate(self.results)]
        rows = self._sorted_rows(rows)

        unit = self.unit_label()
        factor = self.unit_factor()
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels([f"diameter ({unit})"])
        self.row_to_result = [ri for (ri, _d, _ex) in rows]

        if rows:
            st = self.scale_stats_for_display(self.stats)
            self.stats_label.setText(
                f"Particles: {self.stats.get('particles', 0)} | "
                f"D10={st.get('D10', 0):.3f} {unit} | D50={st.get('D50', 0):.3f} {unit} | "
                f"D90={st.get('D90', 0):.3f} {unit} | mean={st.get('mean', 0):.3f} {unit} | "
                f"std={st.get('std', 0):.3f} {unit}"
            )
        else:
            # якщо аналіз ще не запускався — чисто
            if not self._ever_ran:
                self.stats_label.setText("—")
                self.table.setRowCount(0)
                self.cb_sort.setEnabled(False)
                return
            self.stats_label.setText("No particles accepted. Adjust parameters.")

        # наповнення
        self.table.setSortingEnabled(False)
        if rows:
            self.table.setRowCount(len(rows))
            for r, (_ri, d_um, excl) in enumerate(rows):
                v = d_um * factor
                cell = QTableWidgetItem(f"{v:.6f}")
                if excl:
                    cell.setForeground(Qt.gray)
                    f = cell.font(); f.setItalic(True); cell.setFont(f)
                cell.setToolTip(f"{v:.6f} {unit}")
                cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(r, 0, cell)
            self.cb_sort.setEnabled(True)
        else:
            self.table.setRowCount(1)
            ph = QTableWidgetItem("—")
            ph.setFlags(ph.flags() & ~Qt.ItemIsEditable)
            ph.setForeground(Qt.gray)
            self.table.setItem(0, 0, ph)
            self.cb_sort.setEnabled(False)
        self.table.setSortingEnabled(False)

    def update_ext_table(self):
        if not self.sel_show_ext_table:
            return
        if not self.ext_rows and self.results:
            self._compute_ext_analytics()

        unit = self.unit_label()
        factor = self.unit_factor()
        headers = [
            "idx", f"d ({unit})", "circ", f"area ({unit}²)", f"perim ({unit})",
            f"d_eq_area ({unit})", f"d_eq_perim ({unit})", f"Feret min ({unit})", f"Feret max ({unit})",
            "AR", f"ellipse min ({unit})", f"ellipse max ({unit})", "roundness", "solidity"
        ]
        self.table_ext.setHorizontalHeaderLabels(headers)
        rows = self.ext_rows or []

        def _fill():
            for r, row in enumerate(rows):
                vals = [
                    int(row.idx),  # індекс як ціле
                    row.d_um * factor,
                    row.circ,
                    row.area_um2 * (factor ** 2),
                    row.perimeter_um * factor,
                    row.d_eq_area_um * factor,
                    row.d_eq_perim_um * factor,
                    row.feret_min_um * factor,
                    row.feret_max_um * factor,
                    row.aspect_ratio,
                    row.ellipse_minor_um * factor,
                    row.ellipse_major_um * factor,
                    row.roundness,
                    row.solidity,
                ]
                for c, v in enumerate(vals):
                    if c == 0:
                        item = QTableWidgetItem(str(v))
                        item.setTextAlignment(Qt.AlignCenter)
                    else:
                        item = NumericItem(v) if isinstance(v, (int, float, np.floating)) else QTableWidgetItem(str(v))
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.table_ext.setItem(r, c, item)

        self._fill_table_safely(self.table_ext, len(rows), _fill)

    def update_stats_plus_panel(self):
        if not self.sel_show_stats_plus:
            return
        unit = self.unit_label(); factor = self.unit_factor()
        w = self.weighted_stats or {}
        st = self.scale_stats_for_display(self.stats)
        d32 = (w.get("D32", 0.0) * factor)
        d43 = (w.get("D43", 0.0) * factor)
        def fmt_ci(k):
            lo, hi = self.boot_ci.get(k, (0.0, 0.0))
            return f"[{lo*factor:.3f}, {hi*factor:.3f}] {unit}"
        txt = (
            f"<b>ISO 9276 weighted means</b><br>"
            f"D<sub>3,2</sub> (Sauter) = <b>{d32:.3f} {unit}</b><br>"
            f"D<sub>4,3</sub> (De&nbsp;Brouckere) = <b>{d43:.3f} {unit}</b><br><br>"
            f"<b>Bootstrap 95% CI</b><br>"
            f"D10 CI: {fmt_ci('D10')}<br>"
            f"D50 CI: {fmt_ci('D50')}<br>"
            f"D90 CI: {fmt_ci('D90')}<br>"
            f"mean CI: {fmt_ci('mean')}"
        )
        self.stats_plus.setText(txt)

    def export_csv(self):
        if not self.results:
            QMessageBox.information(self, "CSV", "Run analysis first."); return
        active = [d for i, (_c, d, _ci) in enumerate(self.results) if i not in self.excluded_idx]
        if not active:
            QMessageBox.information(self, "CSV", "No active particles to export."); return

        unit = self.unit_label(); factor = self.unit_factor()
        path, _ = QFileDialog.getSaveFileName(self, "Save diameters CSV", f"diameters_{unit}.csv", "CSV (*.csv)")
        if not path: return
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                col = f"diameter_{'nm' if unit=='nm' else 'um'}"; w.writerow([col])
                rows = [(i, d, (i in self.excluded_idx)) for i, (_c, d, _ci) in enumerate(self.results) if i not in self.excluded_idx]
                rows = self._sorted_rows(rows)
                for (_ri, d_um, _ex) in rows:
                    w.writerow([f"{d_um * factor:.6f}"])
            QMessageBox.information(self, "CSV", f"Saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "CSV", f"Failed to save: {e}")

    def save_overlay(self):
        if self.overlay_img is None:
            QMessageBox.information(self, "Overlay", "Run analysis first."); return

        pics = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation) or ""
        default_path = str(Path(pics) / "overlay.png")

        path, selected = QFileDialog.getSaveFileName(
            self, "Save overlay image", default_path,
            "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not path: return

        p = Path(path)
        if p.suffix.lower() == "":
            sel = (selected or "").lower()
            if "png" in sel: p = p.with_suffix(".png")
            elif "jpg" in sel or "jpeg" in sel: p = p.with_suffix(".jpg")
            else: p = p.with_suffix(".png")

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
            if cv2.imwrite(str(p), img):
                QMessageBox.information(self, "Overlay", f"Saved:\n{p}")
                return
        except Exception as e:
            QMessageBox.critical(self, "Overlay", f"Failed to save with OpenCV:\n{e}")
            return

        QMessageBox.critical(self, "Overlay",
            "Failed to save the overlay image.\nEnsure the folder is writable and extension is valid (.png/.jpg).")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._position_drop_hint()

    def _overlay_view_scale(self) -> float:
        try:
            tr = self.view_overlay.transform()
            s = (float(tr.m11()) + float(tr.m22())) * 0.5
            return s if s > 1e-3 else 1.0
        except Exception:
            return 1.0

    def _best_label_box(self, cx, cy, tw, th, pad_x, pad_y, W, H, v):
        offs = [
            (+int(12 / v), -int(12 / v)),
            (+int(12 / v), +int(12 / v)),
            (-int(12 / v) - tw - 2 * pad_x, -int(12 / v)),
            (-int(12 / v) - tw - 2 * pad_x, +int(12 / v)),
        ]
        best = None; best_score = -1e9
        for dx, dy in offs:
            bx0 = cx + dx; by0 = cy + dy
            bx1 = bx0 + tw + 2 * pad_x; by1 = by0 + th + 2 * pad_y
            over = max(0, 2 - bx0) + max(0, 2 - by0) + max(0, bx1 - (W - 2)) + max(0, by1 - (H - 2))
            score = -5 * over - (abs(dy) + abs(dx))
            if score > best_score:
                best_score = score; best = (bx0, by0, bx1, by1)
        bx0, by0, bx1, by1 = best
        bx0 = min(max(2, bx0), W - 2); by0 = min(max(2, by0), H - 2)
        bx1 = min(max(2, bx1), W - 2); by1 = min(max(2, by1), H - 2)
        return bx0, by0, bx1, by1

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
        for w in [self.btn_open, self.btn_run, self.btn_export_csv, self.btn_export_ext, self.btn_save_overlay,
                  self.btn_toggle_remove, self.btn_clear_excl, self.cb_mode, self.btn_autotune]:
            w.setEnabled(not busy)
        self.progress.setVisible(busy)
        QApplication.processEvents()

    # ---------- Find by index (Ctrl+F) ----------
    def find_particle(self):
        if not self.results:
            QMessageBox.information(self, "Find", "Run analysis first."); return
        n = len(self.results)
        val, ok = QInputDialog.getInt(self, "Find particle", "Index (1..N):", 1, 1, n, 1)
        if not ok: return
        ridx = val - 1  # 0-based індекс у self.results

        # знайти відповідний рядок у поточному порядку
        try:
            row = self.row_to_result.index(ridx)
        except ValueError:
            QMessageBox.information(self, "Find", "Index not visible with current order/filter."); return

        self.table.setCurrentCell(row, 0)
        self.table.scrollToItem(self.table.item(row, 0))
        self.hover_idx = ridx
        self.tabs.setCurrentWidget(self.view_overlay)
        self.render_overlay_with_highlight()


def main():
    app = QApplication(sys.argv)
    app.setStyle(ToolTipDelayStyle(app.style()))
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
