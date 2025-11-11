"""
Custom Qt style tweaks for the SEM PSD GUI.

Extends QProxyStyle to increase tooltip visibility time
(useful for displaying detailed numeric tooltips in tables).
"""

from PySide6.QtWidgets import QProxyStyle, QStyle


class ToolTipDelayStyle(QProxyStyle):
    """Style override to delay tooltip appearance and hide timeout."""
    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.SH_ToolTip_WakeUpDelay:
            return 5000  # 5 s before showing tooltip
        if hint == QStyle.SH_ToolTip_FallAsleepDelay:
            return 30000  # 30 s visible duration
        return super().styleHint(hint, option, widget, returnData)
