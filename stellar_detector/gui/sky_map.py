"""Interactive sky map with Mollweide projection and anomaly overlay.

Uses matplotlib embedded in a Qt widget for the projection, with
interactive features (click-to-select, zoom).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import numpy as np

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

if TYPE_CHECKING:
    import pandas as pd

    from ..core.models import AnomalyResult


class SkyMapWidget(QWidget):
    """Interactive sky map with projection selector and anomaly overlay."""

    star_clicked = Signal(str)  # source_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df = None
        self._results: list[AnomalyResult] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Controls
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Projection:"))
        self._proj_combo = QComboBox()
        self._proj_combo.addItems(["mollweide", "aitoff", "hammer", "lambert"])
        self._proj_combo.currentTextChanged.connect(self._refresh)
        controls.addWidget(self._proj_combo)
        controls.addStretch()
        self._info_label = QLabel("")
        controls.addWidget(self._info_label)
        layout.addLayout(controls)

        # Matplotlib canvas
        self._figure = Figure(figsize=(12, 6), facecolor="#1e1e2e")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self._canvas.mpl_connect("button_press_event", self._on_click)

    def set_data(self, df: pd.DataFrame, results: list[AnomalyResult] | None = None):
        self._df = df
        self._results = results or []
        self._refresh()

    def _refresh(self, _=None):
        self._figure.clear()

        if self._df is None or "ra" not in self._df.columns or "dec" not in self._df.columns:
            self._canvas.draw()
            return

        projection = self._proj_combo.currentText()
        ax = self._figure.add_subplot(111, projection=projection)
        ax.set_facecolor("#1e1e2e")

        df = self._df.dropna(subset=["ra", "dec"])
        ra_rad = np.deg2rad(df["ra"].values - 180)
        dec_rad = np.deg2rad(df["dec"].values)

        ax.scatter(ra_rad, dec_rad, s=0.5, c="#6c7086", alpha=0.4, rasterized=True)

        # Overlay anomalies
        anomaly_ids = {r.star_id for r in self._results}
        if anomaly_ids and "source_id" in df.columns:
            mask = df["source_id"].astype(str).isin(anomaly_ids)
            if mask.any():
                anom = df[mask]
                ra_a = np.deg2rad(anom["ra"].values - 180)
                dec_a = np.deg2rad(anom["dec"].values)
                ax.scatter(ra_a, dec_a, s=12, c="#f38ba8", alpha=0.9,
                           edgecolors="#eba0ac", linewidths=0.3, zorder=5)

        ax.grid(True, alpha=0.15, color="#585b70")
        ax.set_title("Sky Distribution", color="#cdd6f4", fontsize=12)
        ax.tick_params(colors="#6c7086")

        n_anom = len(anomaly_ids)
        self._info_label.setText(f"{len(df)} stars | {n_anom} anomalies")

        self._figure.tight_layout()
        self._canvas.draw()

    def _on_click(self, event):
        if event.inaxes is None or self._df is None:
            return
        if "ra" not in self._df.columns or "source_id" not in self._df.columns:
            return

        click_ra = np.rad2deg(event.xdata) + 180
        click_dec = np.rad2deg(event.ydata)

        df = self._df.dropna(subset=["ra", "dec"])
        dra = (df["ra"].values - click_ra) * np.cos(np.deg2rad(click_dec))
        ddec = df["dec"].values - click_dec
        dist = dra**2 + ddec**2
        nearest = dist.argmin()

        if dist[nearest] < 4.0:  # within ~2 degrees
            star_id = str(df.iloc[nearest]["source_id"])
            self.star_clicked.emit(star_id)
