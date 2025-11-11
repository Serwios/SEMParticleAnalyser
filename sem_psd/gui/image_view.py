"""
Lightweight image viewer widget for SEM PSD GUI.

Features:
- Smooth panning/zooming (mouse wheel, Ctrl+±, Ctrl+0/1)
- Auto-fit on load and double-click
- Click signal with image-space integer coords
"""

from __future__ import annotations
from typing import Union

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QTransform, QPixmap
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSizePolicy
)

from .utils import np_to_qpix


class ImageView(QGraphicsView):
    """QGraphicsView wrapper to display numpy/QPixmap images with zoom & pan."""
    sig_clicked = Signal(int, int)  # x, y in image coordinates

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._item = QGraphicsPixmapItem()
        self.scene().addItem(self._item)

        self._img_w: int = 0
        self._img_h: int = 0
        self._auto_fit: bool = True

        self.setRenderHints(
            self.renderHints()
            | QPainter.Antialiasing
            | QPainter.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)

    def set_image(self, img: Union[np.ndarray, QPixmap]) -> None:
        """Set image from numpy array (BGR/GRAY) or QPixmap and auto-fit."""
        pm: QPixmap = img if isinstance(img, QPixmap) else np_to_qpix(img)
        self._item.setPixmap(pm)
        self._img_w, self._img_h = pm.width(), pm.height()
        self._auto_fit = True
        self.fit_to_view()

    def fit_to_view(self) -> None:
        """Fit current pixmap to the viewport (keeping aspect)."""
        if self._img_w == 0 or self.width() < 5 or self.height() < 5:
            return
        self.setTransform(QTransform())
        r = self._item.boundingRect()
        m = 2  # small margin to avoid clipping
        self.fitInView(r.adjusted(m, m, -m, -m), Qt.KeepAspectRatio)

    # --- Qt events ---

    def resizeEvent(self, e) -> None:
        if self._auto_fit and self._img_w:
            self.fit_to_view()
        super().resizeEvent(e)

    def wheelEvent(self, e) -> None:
        if self._img_w == 0:
            return
        self._auto_fit = False
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, e) -> None:
        self._auto_fit = True
        self.fit_to_view()
        super().mouseDoubleClickEvent(e)

    def keyPressEvent(self, e) -> None:
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
                self._auto_fit = False
                self.scale(1.15, 1.15)
                e.accept(); return
            if e.key() == Qt.Key_Minus:
                self._auto_fit = False
                self.scale(1/1.15, 1/1.15)
                e.accept(); return
        super().keyPressEvent(e)

    def mousePressEvent(self, e) -> None:
        if self._img_w:
            p = self.mapToScene(e.pos())
            x = max(0, min(self._img_w - 1, int(round(p.x()))))
            y = max(0, min(self._img_h - 1, int(round(p.y()))))
            self.sig_clicked.emit(x, y)
        super().mousePressEvent(e)
