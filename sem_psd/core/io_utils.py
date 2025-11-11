"""
Image I/O and metadata utilities.

Provides robust grayscale image loading and
extraction of SEM scale (µm/px) from TIFF metadata.
"""

from __future__ import annotations
import re
from typing import Optional
import cv2
import numpy as np
from PIL import Image


def imread_gray(path: str) -> np.ndarray:
    """Read an image and return it as an 8-bit grayscale array."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        # Fallback: use Pillow if OpenCV fails
        pil = Image.open(path)
        if pil.mode not in ("L", "I;16", "I;16B", "I;16L"):
            pil = pil.convert("L")
        arr = np.array(pil)
        if arr.dtype == np.uint16:
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return arr.astype(np.uint8)

    # Normalize 16-bit and convert RGB→gray if needed
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def dump_tiff_metadata_text(image_path: str) -> str:
    """Return TIFF metadata as concatenated text for regex parsing."""
    try:
        pil = Image.open(image_path)
    except Exception as e:
        return f"[ERROR opening TIFF: {e}]"

    out = []
    # Read EXIF/TIFF tags
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

    # Include general info fields
    try:
        for k, v in (pil.info or {}).items():
            if isinstance(v, bytes):
                v = v.decode(errors="ignore")
            out.append(f"[{k}] {v}")
    except Exception:
        pass

    return "\n".join(out)


def parse_um_per_px_from_text(txt: str) -> Optional[float]:
    """Extract µm/px scale from TIFF metadata text."""
    if not txt:
        return None

    # Direct PixelWidth field (in meters)
    m = re.search(r"PixelWidth\s*=\s*([0-9eE\.\-\+]+)", txt)
    if m:
        try:
            px_m = float(m.group(1))
            if px_m > 0:
                return px_m * 1e6  # convert to µm
        except Exception:
            pass

    # Derived from horizontal field width and resolution
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


def scale_from_metadata(image_path: str) -> Optional[float]:
    """Read a TIFF file and return scale (µm/px) parsed from metadata."""
    return parse_um_per_px_from_text(dump_tiff_metadata_text(image_path))
