"""
Utility helpers for the SEM PSD GUI.

Includes:
- NumericItem: sortable numeric table item with tooltip
- np_to_qpix: convert NumPy array to QPixmap
"""

from __future__ import annotations
import numpy as np
import cv2
from PySide6.QtWidgets import QTableWidgetItem
from PySide6.QtGui import QImage, QPixmap


class NumericItem(QTableWidgetItem):
    """Numeric table item with proper sorting and full-value tooltip."""
    def __init__(self, value, fmt: str = "{:.6f}") -> None:
        if isinstance(value, (int, float, np.floating)):
            text = fmt.format(float(value))
            super().__init__(text)
            self._num = float(value)
        else:
            super().__init__(str(value))
            self._num = None
        self.setToolTip(self.text())

    def __lt__(self, other: QTableWidgetItem) -> bool:
        """Ensure numeric sorting if both cells contain numbers."""
        if isinstance(other, QTableWidgetItem) and self.column() == other.column():
            a = self._num
            b = getattr(other, "_num", None)
            if a is not None and b is not None:
                return a < b
        return super().__lt__(other)


def np_to_qpix(img: np.ndarray) -> QPixmap:
    """Convert a NumPy image (BGR or grayscale) to a QPixmap."""
    if img.ndim == 2:
        qimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        qimg = img
    h, w, _ = qimg.shape
    qimg = cv2.cvtColor(qimg, cv2.COLOR_BGR2RGB)
    qimage = QImage(qimg.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)
