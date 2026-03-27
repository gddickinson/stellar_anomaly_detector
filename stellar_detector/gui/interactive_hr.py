"""Interactive HR diagram using pyqtgraph — zoom, hover tooltips, click-to-select."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

if TYPE_CHECKING:
    import pandas as pd

    from ..core.models import AnomalyResult

# Color map for anomaly types
ANOMALY_COLORS = {
    "hr_diagram_outlier": (255, 80, 80),
    "lifetime_anomaly": (255, 165, 0),
    "unusual_kinematics": (0, 200, 255),
    "unusual_metallicity": (180, 80, 255),
    "chemical_anomaly": (255, 0, 255),
    "variability_anomaly": (0, 255, 128),
    "dyson_sphere_candidate": (255, 255, 0),
    "stellar_engine_candidate": (255, 200, 0),
    "megastructure_candidate": (255, 220, 50),
    "infrared_excess": (255, 100, 50),
    "astrometric_anomaly": (100, 200, 255),
    "cross_catalog_anomaly": (255, 255, 255),
}
DEFAULT_COLOR = (200, 200, 200)


class InteractiveHRWidget(QWidget):
    """Interactive Hertzsprung-Russell diagram with pyqtgraph."""

    star_clicked = Signal(str)  # source_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df: pd.DataFrame | None = None
        self._results: list[AnomalyResult] = []
        self._anomaly_ids: set[str] = set()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Controls row
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Color by:"))
        self._color_combo = QComboBox()
        self._color_combo.addItems([
            "Anomaly type", "Temperature", "Metallicity", "Distance", "Quality",
        ])
        self._color_combo.currentTextChanged.connect(self._recolor)
        controls.addWidget(self._color_combo)
        controls.addStretch()
        self._info_label = QLabel("")
        controls.addWidget(self._info_label)
        layout.addLayout(controls)

        # Plot widget
        self._plot = pg.PlotWidget(title="HR Diagram (Color-Magnitude)")
        self._plot.setLabel("bottom", "BP - RP", units="mag")
        self._plot.setLabel("left", "Absolute G Magnitude")
        self._plot.invertY(True)
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setBackground("#1e1e2e")

        self._bg_scatter = pg.ScatterPlotItem(size=3, pen=None)
        self._anom_scatter = pg.ScatterPlotItem(size=8, pen=pg.mkPen("w", width=0.5))
        self._plot.addItem(self._bg_scatter)
        self._plot.addItem(self._anom_scatter)

        self._anom_scatter.sigClicked.connect(self._on_point_clicked)
        self._bg_scatter.sigClicked.connect(self._on_point_clicked)

        # Tooltip proxy
        self._proxy = pg.SignalProxy(
            self._plot.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse_moved
        )
        self._tooltip_item = pg.TextItem("", anchor=(0, 1), color="w")
        self._tooltip_item.setVisible(False)
        self._plot.addItem(self._tooltip_item)

        layout.addWidget(self._plot)

    def set_data(self, df: pd.DataFrame, results: list[AnomalyResult] | None = None):
        """Load catalog data and optional anomaly results."""
        self._df = df
        self._results = results or []
        self._anomaly_ids = {r.star_id for r in self._results}
        self._refresh()

    def _refresh(self):
        if self._df is None or "bp_rp" not in self._df.columns or "abs_mag" not in self._df.columns:
            return

        clean = self._df.dropna(subset=["bp_rp", "abs_mag"])
        x = clean["bp_rp"].values
        y = clean["abs_mag"].values

        # Background stars
        id_col = "source_id" if "source_id" in clean.columns else None
        if id_col:
            is_anom = clean[id_col].astype(str).isin(self._anomaly_ids)
        else:
            is_anom = np.zeros(len(clean), dtype=bool)

        bg_mask = ~is_anom
        bg_colors = self._compute_colors(clean[bg_mask], alpha=60)
        self._bg_scatter.setData(
            x=x[bg_mask], y=y[bg_mask],
            brush=[pg.mkBrush(*c) for c in bg_colors],
            data=clean.index[bg_mask].tolist(),
        )

        # Anomaly stars
        if is_anom.any():
            anom_colors = self._compute_anomaly_colors(clean[is_anom])
            self._anom_scatter.setData(
                x=x[is_anom], y=y[is_anom],
                brush=[pg.mkBrush(*c) for c in anom_colors],
                data=clean.index[is_anom].tolist(),
            )
        else:
            self._anom_scatter.setData(x=[], y=[])

        self._info_label.setText(
            f"{len(clean)} stars | {is_anom.sum()} anomalies"
        )

    def _recolor(self, _=None):
        self._refresh()

    def _compute_colors(self, df, alpha=60):
        """Compute colors for background stars based on selected mode."""
        mode = self._color_combo.currentText()
        n = len(df)

        if mode == "Temperature" and "teff_gspphot" in df.columns:
            vals = df["teff_gspphot"].fillna(5000).values
            normed = np.clip((vals - 3000) / 27000, 0, 1)
            colors = []
            for v in normed:
                r = int(255 * min(1, v * 2))
                b = int(255 * min(1, (1 - v) * 2))
                g = int(100 * (1 - abs(v - 0.5) * 2))
                colors.append((r, g, b, alpha))
            return colors

        if mode == "Metallicity" and "mh_gspphot" in df.columns:
            vals = df["mh_gspphot"].fillna(0).values
            normed = np.clip((vals + 2) / 3, 0, 1)
            return [(int(255 * v), 100, int(255 * (1 - v)), alpha) for v in normed]

        if mode == "Distance" and "distance_pc" in df.columns:
            vals = df["distance_pc"].fillna(500).values
            normed = np.clip(vals / np.percentile(vals[np.isfinite(vals)], 95), 0, 1)
            return [(100, int(200 * (1 - v)), int(255 * v), alpha) for v in normed]

        if mode == "Quality" and "quality_score" in df.columns:
            vals = df["quality_score"].fillna(0.5).values
            return [(int(255 * (1 - v)), int(255 * v), 100, alpha) for v in vals]

        return [(150, 150, 180, alpha)] * n

    def _compute_anomaly_colors(self, df):
        """Color anomaly points by their anomaly type."""
        colors = []
        id_col = "source_id" if "source_id" in df.columns else None
        type_map = {}
        for r in self._results:
            type_map[r.star_id] = r.anomaly_type.anomaly_name

        for idx, row in df.iterrows():
            star_id = str(row[id_col]) if id_col else str(idx)
            anom_type = type_map.get(star_id, "")
            color = ANOMALY_COLORS.get(anom_type, DEFAULT_COLOR)
            colors.append((*color, 220))
        return colors

    def _on_point_clicked(self, plot, points, ev):
        if not points or self._df is None:
            return
        point = points[0]
        idx = point.data()
        if idx is not None and "source_id" in self._df.columns:
            try:
                star_id = str(self._df.loc[idx, "source_id"])
                self.star_clicked.emit(star_id)
            except (KeyError, IndexError):
                pass

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        if self._plot.sceneBoundingRect().contains(pos):
            mouse_point = self._plot.plotItem.vb.mapSceneToView(pos)
            self._tooltip_item.setPos(mouse_point.x(), mouse_point.y())

            # Find nearest point
            if (self._df is not None
                    and "bp_rp" in self._df.columns
                    and "abs_mag" in self._df.columns):
                clean = self._df.dropna(subset=["bp_rp", "abs_mag"])
                if len(clean) > 0:
                    dx = clean["bp_rp"].values - mouse_point.x()
                    dy = clean["abs_mag"].values - mouse_point.y()
                    dist = dx**2 + dy**2
                    nearest = dist.argmin()

                    # Only show if close enough
                    view_range = self._plot.viewRange()
                    x_range = view_range[0][1] - view_range[0][0]
                    threshold = (x_range * 0.02) ** 2
                    if dist[nearest] < threshold:
                        row = clean.iloc[nearest]
                        text_parts = [f"BP-RP: {row['bp_rp']:.2f}, M_G: {row['abs_mag']:.2f}"]
                        if "source_id" in row.index:
                            text_parts.insert(0, str(row["source_id"]))
                        if "teff_gspphot" in row.index and not np.isnan(row.get("teff_gspphot", np.nan)):
                            text_parts.append(f"Teff: {row['teff_gspphot']:.0f}K")
                        self._tooltip_item.setText("\n".join(text_parts))
                        self._tooltip_item.setVisible(True)
                        return
            self._tooltip_item.setVisible(False)
