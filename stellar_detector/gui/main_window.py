"""Main application window — dockable panels, menus, toolbar, status bar."""

from __future__ import annotations

import sys
import uuid

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QToolBar,
)

from ..core.models import AnomalyResult, CatalogSource, DetectionConfig
from .analysis_config import AnalysisConfigWidget
from .catalog_browser import CatalogBrowserWidget
from .data_table import DataTableWidget
from .job_manager import JobManagerWidget
from .property_inspector import PropertyInspectorWidget
from .dashboard import DashboardWidget
from .theme import THEMES
from .workers import AnalysisWorker, FetchWorker


class MainWindow(QMainWindow):
    """Central application window with dockable panels."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stellar Anomaly Detector v6.0")
        self.resize(1700, 1000)

        self._df: pd.DataFrame = pd.DataFrame()
        self._results: list[AnomalyResult] = []
        self._config = DetectionConfig()
        self._current_theme = "dark"
        self._workers: list = []

        self._build_central()
        self._build_docks()
        self._build_menus()
        self._build_toolbar()
        self._build_statusbar()
        self._apply_theme("dark")

    # ── Central widget ───────────────────────────────────────────────
    def _build_central(self):
        from PySide6.QtWidgets import QSplitter

        splitter = QSplitter(Qt.Vertical)

        self._dashboard = DashboardWidget()
        splitter.addWidget(self._dashboard)

        self._data_table = DataTableWidget()
        self._data_table.row_selected.connect(self._on_star_selected)
        splitter.addWidget(self._data_table)

        splitter.setSizes([500, 300])
        self.setCentralWidget(splitter)

        # Wire dashboard star clicks to property inspector
        self._dashboard.hr_widget.star_clicked.connect(self._on_star_clicked_by_id)
        self._dashboard.sky_widget.star_clicked.connect(self._on_star_clicked_by_id)

    # ── Dock widgets ─────────────────────────────────────────────────
    def _build_docks(self):
        # Left: catalog browser
        self._catalog_browser = CatalogBrowserWidget()
        self._catalog_browser.fetch_requested.connect(self._on_fetch_requested)
        self._catalog_dock = self._add_dock(
            "Catalogs", self._catalog_browser, Qt.LeftDockWidgetArea
        )

        # Right: property inspector
        self._inspector = PropertyInspectorWidget()
        self._inspector_dock = self._add_dock(
            "Properties", self._inspector, Qt.RightDockWidgetArea
        )

        # Right (tabbed): analysis config
        self._config_panel = AnalysisConfigWidget()
        self._config_panel.config_changed.connect(self._on_config_changed)
        self._config_dock = self._add_dock(
            "Configuration", self._config_panel, Qt.RightDockWidgetArea
        )
        self.tabifyDockWidget(self._inspector_dock, self._config_dock)
        self._inspector_dock.raise_()

        # Bottom: job manager
        self._job_manager = JobManagerWidget()
        self._job_dock = self._add_dock(
            "Jobs & Logs", self._job_manager, Qt.BottomDockWidgetArea
        )

    def _add_dock(self, title: str, widget, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea
        )
        self.addDockWidget(area, dock)
        return dock

    # ── Menu bar ─────────────────────────────────────────────────────
    def _build_menus(self):
        menu = self.menuBar()

        # File
        file_menu = menu.addMenu("&File")
        self._add_action(file_menu, "&Open CSV...", self._open_csv, QKeySequence.Open)
        self._add_action(file_menu, "&Save Results...", self._save_results, QKeySequence.Save)
        file_menu.addSeparator()
        self._add_action(file_menu, "Export &Report...", self._export_report)
        file_menu.addSeparator()
        self._add_action(file_menu, "&Quit", self.close, QKeySequence.Quit)

        # Data
        data_menu = menu.addMenu("&Data")
        for source, name, _ in [
            (CatalogSource.GAIA_DR3, "Fetch Gaia DR3", None),
            (CatalogSource.HIPPARCOS, "Fetch Hipparcos", None),
            (CatalogSource.TYCHO2, "Fetch Tycho-2", None),
            (CatalogSource.SYNTHETIC, "Load Synthetic Data", None),
        ]:
            self._add_action(
                data_menu, name,
                lambda checked=False, s=source: self._quick_fetch(s),
            )

        # Analysis
        analysis_menu = menu.addMenu("&Analysis")
        self._add_action(analysis_menu, "&Run Full Analysis", self._run_analysis, "Ctrl+R")
        analysis_menu.addSeparator()
        self._add_action(analysis_menu, "Configure...", lambda: self._config_dock.raise_())

        # View
        view_menu = menu.addMenu("&View")
        for dock in [self._catalog_dock, self._inspector_dock, self._config_dock, self._job_dock]:
            view_menu.addAction(dock.toggleViewAction())
        view_menu.addSeparator()
        self._add_action(view_menu, "&HR Diagram", lambda: self._dashboard._tabs.setCurrentIndex(0))
        self._add_action(view_menu, "&Sky Map", lambda: self._dashboard._tabs.setCurrentIndex(1))
        self._add_action(view_menu, "&3D Map", lambda: self._dashboard._tabs.setCurrentIndex(2))
        self._add_action(view_menu, "&Light Curves", lambda: self._dashboard._tabs.setCurrentIndex(3))
        view_menu.addSeparator()
        self._add_action(view_menu, "Toggle &Theme", self._toggle_theme, "Ctrl+T")

        # Help
        help_menu = menu.addMenu("&Help")
        self._add_action(help_menu, "&About", self._show_about)

    def _add_action(self, menu, text, slot, shortcut=None):
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    # ── Toolbar ──────────────────────────────────────────────────────
    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        fetch_action = QAction("Fetch Data", self)
        fetch_action.triggered.connect(lambda: self._catalog_dock.raise_())
        tb.addAction(fetch_action)

        run_action = QAction("Run Analysis", self)
        run_action.setShortcut("Ctrl+R")
        run_action.triggered.connect(self._run_analysis)
        tb.addAction(run_action)

        tb.addSeparator()

        theme_action = QAction("Toggle Theme", self)
        theme_action.triggered.connect(self._toggle_theme)
        tb.addAction(theme_action)

    # ── Status bar ───────────────────────────────────────────────────
    def _build_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._status_stars = QAction("Stars: 0", self)
        self._status_anomalies = QAction("Anomalies: 0", self)
        self._statusbar.showMessage("Ready")

    def _update_status(self):
        n_stars = len(self._df)
        n_anom = len(self._results)
        self._statusbar.showMessage(
            f"Stars: {n_stars} | Anomalies: {n_anom} | Theme: {self._current_theme}"
        )

    # ── Slots ────────────────────────────────────────────────────────
    def _on_fetch_requested(self, source, n_stars, ra, dec, radius):
        self._catalog_browser.set_fetching(True)
        job_id = str(uuid.uuid4())[:8]
        self._job_manager.add_job(job_id, f"Fetch {source.value}")

        worker = FetchWorker(source, n_stars, ra, dec, radius)
        worker.signals.progress.connect(
            lambda pct, msg: self._job_manager.update_progress(job_id, pct, msg)
        )
        worker.signals.log.connect(self._job_manager.log)
        worker.signals.finished.connect(lambda df: self._on_fetch_complete(job_id, source, df))
        worker.signals.error.connect(lambda err: self._on_fetch_error(job_id, err))
        self._workers.append(worker)
        worker.start()

    def _on_star_clicked_by_id(self, star_id: str):
        """Handle a star click from a visualization panel."""
        if self._df.empty or "source_id" not in self._df.columns:
            return
        mask = self._df["source_id"].astype(str) == star_id
        if mask.any():
            row_data = self._df[mask].iloc[0].to_dict()
            star_anomalies = [r for r in self._results if r.star_id == star_id]
            self._inspector.show_star(row_data, star_anomalies)
            self._inspector_dock.raise_()

    def _on_fetch_complete(self, job_id, source, df):
        self._df = df
        self._data_table.set_dataframe(df)
        self._dashboard.set_data(df, self._results)
        self._catalog_browser.set_fetching(False)
        self._catalog_browser.update_catalog_status(source, "Loaded", len(df))
        self._job_manager.complete_job(job_id, f"{len(df)} stars loaded")
        self._update_status()

    def _on_fetch_error(self, job_id, error):
        self._catalog_browser.set_fetching(False)
        self._job_manager.fail_job(job_id, error)
        QMessageBox.warning(self, "Fetch Error", error)

    def _on_star_selected(self, row_idx: int, row_data: dict):
        star_id = str(row_data.get("source_id", row_idx))
        star_anomalies = [r for r in self._results if r.star_id == star_id]
        self._inspector.show_star(row_data, star_anomalies)
        self._inspector_dock.raise_()

    def _on_config_changed(self, config: DetectionConfig):
        self._config = config
        self._job_manager.log("Configuration updated")

    def _quick_fetch(self, source: CatalogSource):
        self._on_fetch_requested(source, 2000, 180.0, 0.0, 5.0)

    def _run_analysis(self):
        if self._df.empty:
            QMessageBox.information(self, "No Data", "Load or fetch data first.")
            return

        job_id = str(uuid.uuid4())[:8]
        self._job_manager.add_job(job_id, "Full Analysis")

        worker = AnalysisWorker(self._df, self._config)
        worker.signals.progress.connect(
            lambda pct, msg: self._job_manager.update_progress(job_id, pct, msg)
        )
        worker.signals.log.connect(self._job_manager.log)
        worker.signals.finished.connect(
            lambda result: self._on_analysis_complete(job_id, result)
        )
        worker.signals.error.connect(lambda err: self._job_manager.fail_job(job_id, err))
        self._workers.append(worker)
        worker.start()

    def _on_analysis_complete(self, job_id, result):
        self._results = result["results"]
        stats = result["stats"]
        summary = (
            f"{stats['total']} anomalies, "
            f"{stats.get('unique_stars', 0)} unique stars, "
            f"{stats.get('high_priority_count', 0)} high priority"
        )
        self._job_manager.complete_job(job_id, summary)
        self._dashboard.set_data(self._df, self._results)
        self._update_status()

    # ── File actions ─────────────────────────────────────────────────
    def _open_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Catalog CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        from ..utils.io import load_catalog_csv
        from ..data.preprocessing import preprocess_catalog

        df = load_catalog_csv(path)
        df = preprocess_catalog(df)
        self._df = df
        self._data_table.set_dataframe(df)
        self._update_status()
        self._job_manager.log(f"Loaded {len(df)} rows from {path}")

    def _save_results(self):
        if not self._results:
            QMessageBox.information(self, "No Results", "Run analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "anomalies.csv", "CSV (*.csv);;JSON (*.json)"
        )
        if not path:
            return
        from ..utils.io import save_results
        fmt = "json" if path.endswith(".json") else "csv"
        save_results(self._results, str(path.rsplit("/", 1)[0]),
                     filename=path.rsplit("/", 1)[-1].rsplit(".", 1)[0], fmt=fmt)
        self._job_manager.log(f"Results saved to {path}")

    def _export_report(self):
        if not self._results:
            QMessageBox.information(self, "No Results", "Run analysis first.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not directory:
            return
        from ..utils.io import export_report
        export_report(self._results, self._df, directory)
        self._job_manager.log(f"Report exported to {directory}")

    # ── Theme ────────────────────────────────────────────────────────
    def _toggle_theme(self):
        self._current_theme = "light" if self._current_theme == "dark" else "dark"
        self._apply_theme(self._current_theme)
        self._update_status()

    def _apply_theme(self, name: str):
        app = QApplication.instance()
        if app:
            app.setStyleSheet(THEMES.get(name, ""))

    # ── About ────────────────────────────────────────────────────────
    def _show_about(self):
        QMessageBox.about(
            self,
            "About Stellar Anomaly Detector",
            "Stellar Anomaly Detector v6.0\n\n"
            "Advanced technosignature search engine\n"
            "with multi-catalog analysis and ML pipeline.\n\n"
            "Built with PySide6, scikit-learn, astropy.",
        )


def launch_app():
    """Create and show the main application window."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
