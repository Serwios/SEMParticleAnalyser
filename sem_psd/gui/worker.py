from __future__ import annotations
from pathlib import Path
import numpy as np
from PySide6.QtCore import QObject, Signal, QThread
from ..core import (
    make_roi_mask, preprocess, threshold_pair, morph_open, morph_close, fill_small_holes,
    split_touching_watershed, count_reasonable_components, measure_components,
    detect_blobs_log, Params,
)

class Worker(QObject):
    finished = Signal(object)
    failed = Signal(object)

    def __init__(self, gray: np.ndarray, params: Params, image_path: Path) -> None:
        super().__init__()
        self.gray = gray
        self.P = params
        self.image_path = image_path
        self._cancelled = False  # ← NEW

    def cancel(self):           # ← NEW
        self._cancelled = True

    def _canceled(self) -> bool:  # ← NEW
        th = QThread.currentThread()
        return self._cancelled or (th and th.isInterruptionRequested())

    def run(self) -> None:
        try:
            roi_mask = make_roi_mask(
                self.gray.shape, self.P.exclude_top, self.P.exclude_bottom,
                self.P.exclude_left, self.P.exclude_right
            )
            scale = float(self.P.scale_um_per_px or 0.0)
            if scale <= 0:
                raise ValueError("Scale (µm/px) is required.")

            lev = preprocess(
                self.gray, clahe_clip=self.P.clahe_clip, tophat_um=self.P.tophat_um,
                um_per_px=scale, level_strength=self.P.level_strength
            )
            if self._canceled():  # ← коротке завершення
                self.finished.emit({"lev": lev, "thr_raw": None, "thr_proc": None, "results": []})
                return

            if self.P.analysis_mode == "log":
                results = detect_blobs_log(
                    gray=self.gray, lev_img=lev, um_per_px=scale,
                    dmin_um=self.P.min_d_um, dmax_um=self.P.max_d_um,
                    threshold_rel=self.P.log_threshold_rel, minsep_um=self.P.log_minsep_um,
                    roi_mask=roi_mask, min_rel_contrast=self.P.min_rel_contrast,
                    cancel_cb=self._canceled,  # ← KEY
                )
                self.finished.emit({"lev": lev, "thr_raw": None, "thr_proc": None, "results": results})
                return

            # ----- threshold branch -----
            bwB, bwD = threshold_pair(lev, roi_mask, method=self.P.thr_method,
                                       block_size=self.P.block_size, C=self.P.block_C)
            if self._canceled():
                self.finished.emit({"lev": lev, "thr_raw": bwB, "thr_proc": None, "results": []})
                return

            kB = count_reasonable_components(bwB, scale, self.P.min_d_um, self.P.max_d_um)
            kD = count_reasonable_components(bwD, scale, self.P.min_d_um, self.P.max_d_um)
            bw = bwB if kB >= kD else bwD
            thr_raw = bw.copy()

            bw = morph_close(bw, self.P.closing_um, scale)
            if self._canceled():
                self.finished.emit({"lev": lev, "thr_raw": thr_raw, "thr_proc": bw, "results": []})
                return
            bw = morph_open(bw, self.P.open_um, scale)
            bw = fill_small_holes(bw, max_frac=0.6)

            if self.P.split_touching and not self._canceled():
                bw = split_touching_watershed(bw, um_per_px=scale,
                                              min_neck_um=self.P.min_neck_um,
                                              min_seg_d_um=self.P.min_seg_d_um)
            if self._canceled():
                self.finished.emit({"lev": lev, "thr_raw": thr_raw, "thr_proc": bw, "results": []})
                return

            results = measure_components(bw, self.P.min_d_um, self.P.max_d_um,
                                         self.P.min_circ, scale, lev, self.P.min_rel_contrast)
            self.finished.emit({"lev": lev, "thr_raw": thr_raw, "thr_proc": bw, "results": results})

        except Exception as e:
            self.failed.emit(str(e))
