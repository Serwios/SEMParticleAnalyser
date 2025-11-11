"""
ROI mask creation and image preprocessing utilities.

Includes:
- ROI masking
- Background leveling, top-hat, CLAHE enhancement
- Adaptive / Otsu thresholding pair generation
"""

from __future__ import annotations
import cv2
import numpy as np


def make_roi_mask(shape: tuple[int, int], top: float = 0.02, bottom: float = 0.22,
                  left: float = 0.0, right: float = 0.0) -> np.ndarray:
    """Create a binary mask (255 inside ROI, 0 outside) with fractional exclusions."""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    t = int(h * top)
    b = int(h * (1.0 - bottom))
    l = int(w * left)
    r = int(w * (1.0 - right))
    mask[max(0, t): max(0, b), max(0, l): max(0, r)] = 255
    return mask


def preprocess(
    gray: np.ndarray,
    clahe_clip: float = 2.0,
    tophat_um: float = 0.0,
    um_per_px: float = 0.05,
    level_strength: float = 0.3,
) -> np.ndarray:
    """
    Preprocess a grayscale SEM image:
      1) Leveling by subtracting smoothed background
      2) Optional top-hat filtering
      3) Median blur and CLAHE enhancement
    """
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


def threshold_pair(
    lev: np.ndarray,
    roi_mask: np.ndarray,
    method: str = "otsu",
    block_size: int = 31,
    C: int = -10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate binary (bright/dark) threshold masks using Otsu or adaptive method.

    Returns:
        (bwB, bwD): tuple of binary images (bright, dark).
    """
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
