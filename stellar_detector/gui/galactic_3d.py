"""3D galactic map using pyqtgraph GLViewWidget."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

if TYPE_CHECKING:
    import pandas as pd

    from ..core.models import AnomalyResult


class Galactic3DWidget(QWidget):
    """Rotatable 3D scatter plot of star positions in galactic coordinates."""

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
        self._info_label = QLabel("Load data to view 3D map")
        controls.addWidget(self._info_label)
        controls.addStretch()
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self._reset_view)
        controls.addWidget(reset_btn)
        layout.addLayout(controls)

        # GL view
        self._view = gl.GLViewWidget()
        self._view.setBackgroundColor("#1e1e2e")
        self._view.setCameraPosition(distance=500)

        # Grid
        grid = gl.GLGridItem()
        grid.setSize(1000, 1000)
        grid.setSpacing(100, 100)
        grid.setColor((88, 91, 112, 40))
        self._view.addItem(grid)

        self._bg_scatter = None
        self._anom_scatter = None

        layout.addWidget(self._view)

    def set_data(self, df: pd.DataFrame, results: list[AnomalyResult] | None = None):
        self._df = df
        self._results = results or []
        self._refresh()

    def _refresh(self):
        # Remove old scatter items
        if self._bg_scatter is not None:
            self._view.removeItem(self._bg_scatter)
            self._bg_scatter = None
        if self._anom_scatter is not None:
            self._view.removeItem(self._anom_scatter)
            self._anom_scatter = None

        if self._df is None:
            return

        required = {"ra", "dec", "distance_pc"}
        if not required.issubset(self._df.columns):
            self._info_label.setText("Need ra, dec, distance_pc for 3D map")
            return

        clean = self._df.dropna(subset=list(required))
        if len(clean) == 0:
            return

        # Convert to Cartesian (simplified galactic)
        ra_rad = np.deg2rad(clean["ra"].values)
        dec_rad = np.deg2rad(clean["dec"].values)
        dist = clean["distance_pc"].values

        # Clip extreme distances for visualization
        dist = np.clip(dist, 0, np.percentile(dist, 98))

        x = dist * np.cos(dec_rad) * np.cos(ra_rad)
        y = dist * np.cos(dec_rad) * np.sin(ra_rad)
        z = dist * np.sin(dec_rad)

        pos = np.column_stack([x, y, z])

        # Background stars
        anomaly_ids = {r.star_id for r in self._results}
        id_col = "source_id" if "source_id" in clean.columns else None
        if id_col:
            is_anom = clean[id_col].astype(str).isin(anomaly_ids).values
        else:
            is_anom = np.zeros(len(clean), dtype=bool)

        bg_pos = pos[~is_anom]
        if len(bg_pos) > 0:
            bg_colors = np.full((len(bg_pos), 4), [0.5, 0.5, 0.65, 0.3])
            self._bg_scatter = gl.GLScatterPlotItem(
                pos=bg_pos, color=bg_colors, size=2, pxMode=True
            )
            self._view.addItem(self._bg_scatter)

        # Anomaly stars
        anom_pos = pos[is_anom]
        if len(anom_pos) > 0:
            anom_colors = np.full((len(anom_pos), 4), [1.0, 0.3, 0.3, 0.9])
            self._anom_scatter = gl.GLScatterPlotItem(
                pos=anom_pos, color=anom_colors, size=6, pxMode=True
            )
            self._view.addItem(self._anom_scatter)

        self._info_label.setText(
            f"3D: {len(clean)} stars | {is_anom.sum()} anomalies"
        )

    def _reset_view(self):
        self._view.setCameraPosition(distance=500, elevation=30, azimuth=45)
