"""
Application entry point for SEM PSD GUI.

Initializes the Qt application, applies tooltip style,
and launches the main analysis window.
"""

from __future__ import annotations
import sys
from PySide6.QtWidgets import QApplication
from sem_psd.gui.main_window import MainWindow
from sem_psd.gui.styles import ToolTipDelayStyle

def main() -> None:
    """Start the SEM PSD GUI application."""
    app = QApplication(sys.argv)
    app.setStyle(ToolTipDelayStyle(app.style()))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
