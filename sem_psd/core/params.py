"""
Analysis parameter data structure.

Defines the full set of processing and measurement
parameters used by SEM PSD analysis routines.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Params:
    """Configuration parameters for image preprocessing and particle analysis."""

    # Scale / ROI
    scale_um_per_px: float | None
    exclude_top: float
    exclude_bottom: float
    exclude_left: float
    exclude_right: float

    # Size limits
    min_d_um: float
    max_d_um: float

    # Morphology
    closing_um: float
    open_um: float

    # Object quality filter
    min_circ: float

    # Thresholding
    thr_method: str
    block_size: int
    block_C: int

    # Preprocessing & contrast
    clahe_clip: float
    min_rel_contrast: float
    tophat_um: float
    level_strength: float

    # Watershed splitting
    split_touching: bool
    min_neck_um: float
    min_seg_d_um: float

    # Analysis mode ("contours" | "log")
    analysis_mode: str

    # LoG parameters
    log_threshold_rel: float
    log_minsep_um: float
