"""Configurable dashboard with tabbed visualization panels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from .interactive_hr import InteractiveHRWidget
from .sky_map import SkyMapWidget
from .light_curve_viewer import LightCurveWidget

if TYPE_CHECKING:
    import pandas as pd

    from ..core.models import AnomalyResult

# Defer 3D import — needs OpenGL which may not be available
_GL_AVAILABLE = False
try:
    from .galactic_3d import Galactic3DWidget
    _GL_AVAILABLE = True
except ImportError:
    pass


class DashboardWidget(QWidget):
    """Tabbed visualization dashboard containing all interactive charts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()

        # HR diagram
        self.hr_widget = InteractiveHRWidget()
        self._tabs.addTab(self.hr_widget, "HR Diagram")

        # Sky map
        self.sky_widget = SkyMapWidget()
        self._tabs.addTab(self.sky_widget, "Sky Map")

        # 3D galactic map
        if _GL_AVAILABLE:
            self.galactic_widget = Galactic3DWidget()
            self._tabs.addTab(self.galactic_widget, "3D Galactic Map")
        else:
            self.galactic_widget = None

        # Light curve viewer
        self.lc_widget = LightCurveWidget()
        self._tabs.addTab(self.lc_widget, "Light Curves")

        layout.addWidget(self._tabs)

    def set_data(self, df: pd.DataFrame, results: list[AnomalyResult] | None = None):
        """Update all visualization panels with new data."""
        self.hr_widget.set_data(df, results)
        self.sky_widget.set_data(df, results)
        if self.galactic_widget is not None:
            self.galactic_widget.set_data(df, results)
