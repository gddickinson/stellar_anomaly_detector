"""Light curve viewer with Lomb-Scargle periodogram overlay."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSplitter, QVBoxLayout, QWidget
from PySide6.QtCore import Qt


class LightCurveWidget(QWidget):
    """Interactive light curve and periodogram viewer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Info bar
        info_row = QHBoxLayout()
        self._info_label = QLabel("No light curve loaded")
        info_row.addWidget(self._info_label)
        info_row.addStretch()
        self._fold_btn = QPushButton("Phase Fold")
        self._fold_btn.setEnabled(False)
        self._fold_btn.clicked.connect(self._toggle_fold)
        info_row.addWidget(self._fold_btn)
        layout.addLayout(info_row)

        # Splitter: light curve + periodogram
        splitter = QSplitter(Qt.Vertical)

        # Light curve plot
        self._lc_plot = pg.PlotWidget(title="Light Curve")
        self._lc_plot.setLabel("bottom", "Time", units="days")
        self._lc_plot.setLabel("left", "Magnitude")
        self._lc_plot.invertY(True)
        self._lc_plot.showGrid(x=True, y=True, alpha=0.15)
        self._lc_plot.setBackground("#1e1e2e")
        splitter.addWidget(self._lc_plot)

        # Periodogram plot
        self._pg_plot = pg.PlotWidget(title="Lomb-Scargle Periodogram")
        self._pg_plot.setLabel("bottom", "Period", units="days")
        self._pg_plot.setLabel("left", "Power")
        self._pg_plot.showGrid(x=True, y=True, alpha=0.15)
        self._pg_plot.setBackground("#1e1e2e")
        splitter.addWidget(self._pg_plot)

        splitter.setSizes([300, 200])
        layout.addWidget(splitter)

        self._times = None
        self._mags = None
        self._errs = None
        self._best_period = None
        self._folded = False

    def set_light_curve(
        self,
        times: np.ndarray,
        mags: np.ndarray,
        errs: np.ndarray | None = None,
        star_id: str = "",
    ):
        """Load and display a light curve with automatic periodogram."""
        self._times = times
        self._mags = mags
        self._errs = errs if errs is not None else np.ones_like(mags) * np.std(mags) * 0.1
        self._folded = False

        self._plot_light_curve()
        self._plot_periodogram()
        self._fold_btn.setEnabled(self._best_period is not None and self._best_period > 0)

        n_pts = len(times)
        span = times[-1] - times[0] if n_pts > 1 else 0
        self._info_label.setText(
            f"Star: {star_id} | {n_pts} points | Span: {span:.1f} days"
            + (f" | Best period: {self._best_period:.3f} d" if self._best_period else "")
        )

    def _plot_light_curve(self):
        self._lc_plot.clear()
        if self._times is None:
            return

        err_item = pg.ErrorBarItem(
            x=self._times, y=self._mags,
            height=self._errs * 2,
            pen=pg.mkPen("#585b70", width=1),
        )
        self._lc_plot.addItem(err_item)

        scatter = pg.ScatterPlotItem(
            x=self._times, y=self._mags,
            size=5, pen=None, brush=pg.mkBrush("#89b4fa"),
        )
        self._lc_plot.addItem(scatter)
        self._lc_plot.setTitle("Light Curve")

    def _plot_periodogram(self):
        self._pg_plot.clear()
        if self._times is None or len(self._times) < 10:
            return

        try:
            from astropy.timeseries import LombScargle

            ls = LombScargle(self._times, self._mags, self._errs, fit_mean=True)
            freq, power = ls.autopower(
                minimum_frequency=1.0 / 1000,
                maximum_frequency=1.0 / 0.1,
            )
            periods = 1.0 / freq

            self._pg_plot.plot(periods, power, pen=pg.mkPen("#a6e3a1", width=1.5))

            best_idx = np.argmax(power)
            self._best_period = float(periods[best_idx])

            # Mark best period
            self._pg_plot.plot(
                [self._best_period], [power[best_idx]],
                pen=None, symbol="o", symbolBrush="#f38ba8", symbolSize=8,
            )

            # FAP line
            fap_level = ls.false_alarm_level(0.01)
            self._pg_plot.addLine(y=fap_level, pen=pg.mkPen("#f9e2af", style=Qt.DashLine, width=1))

        except Exception:
            self._best_period = None

    def _toggle_fold(self):
        if self._times is None or self._best_period is None:
            return

        self._folded = not self._folded
        self._lc_plot.clear()

        if self._folded:
            phase = (self._times % self._best_period) / self._best_period
            scatter = pg.ScatterPlotItem(
                x=phase, y=self._mags,
                size=5, pen=None, brush=pg.mkBrush("#89b4fa"),
            )
            self._lc_plot.addItem(scatter)
            self._lc_plot.setLabel("bottom", "Phase")
            self._lc_plot.setTitle(f"Phase-Folded (P={self._best_period:.3f} d)")
            self._fold_btn.setText("Unfold")
        else:
            self._plot_light_curve()
            self._lc_plot.setLabel("bottom", "Time", units="days")
            self._fold_btn.setText("Phase Fold")
