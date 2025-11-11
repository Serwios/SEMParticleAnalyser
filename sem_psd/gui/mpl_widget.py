from __future__ import annotations
from typing import Dict
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplWidget(QWidget):
    """
    Minimal Matplotlib widget:
      - `plot_hist`: PSD histogram with D10/D50/D90 markers.
      - `plot_cum`: cumulative % curve with the same markers.
    Only draws when data is non-empty.
    """

    def __init__(self, parent=None, *, dpi: int = 100):
        super().__init__(parent)
        self.fig = Figure(figsize=(5, 4), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    # ---- helpers ----
    @staticmethod
    def _fd_bins(x: np.ndarray, max_bins: int = 80) -> int:
        """Freedman–Diaconis bins; falls back sensibly for tiny samples."""
        x = np.asarray(x, float)
        n = x.size
        if n < 2:
            return 1
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        if iqr <= 0:
            return min(max_bins, max(1, int(np.sqrt(n))))
        h = 2 * iqr * (n ** (-1 / 3))
        if h <= 0:
            return min(max_bins, max(1, int(np.sqrt(n))))
        k = int(np.ceil((x.max() - x.min()) / h))
        return max(1, min(max_bins, k))

    @staticmethod
    def _add_markers(ax, stats: Dict[str, float], unit: str):
        labels_added = False
        for name in ("D10", "D50", "D90"):
            v = float(stats.get(name, 0.0))
            if v > 0:
                ax.axvline(v, linestyle="--", linewidth=1.2, label=f"{name}={v:.2f} {unit}")
                labels_added = True
        return labels_added

    # ---- plots ----
    def plot_hist(self, d_vals: np.ndarray, st: Dict[str, float], title: str, unit: str):
        """Histogram of particle diameters with percentile markers."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        d_vals = np.asarray(d_vals, float)
        if d_vals.size:
            bins = self._fd_bins(d_vals)
            ax.hist(d_vals, bins=bins)
            has_legend = self._add_markers(ax, st or {}, unit)
            ax.set_xlabel(f"Particle diameter ({unit})")
            ax.set_ylabel("Count")
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            if has_legend:
                ax.legend()
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

        self.fig.tight_layout()
        self.canvas.draw()

    def plot_cum(self, d_vals: np.ndarray, st: Dict[str, float], unit: str):
        """Cumulative % curve (ECDF) with percentile markers."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        d_vals = np.asarray(d_vals, float)
        if d_vals.size:
            s = np.sort(d_vals)
            cum = np.arange(1, s.size + 1, dtype=float) / s.size * 100.0
            ax.plot(s, cum, linewidth=1.5)
            has_legend = self._add_markers(ax, st or {}, unit)
            ax.set_xlabel(f"Particle diameter ({unit})")
            ax.set_ylabel("Cumulative %")
            ax.set_title("Cumulative PSD")
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.25)
            if has_legend:
                ax.legend()
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

        self.fig.tight_layout()
        self.canvas.draw()
