"""QThread workers for non-blocking data fetching and analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd
from PySide6.QtCore import QObject, QThread, Signal

from ..core.models import AnomalyResult, CatalogSource, DetectionConfig


class WorkerSignals(QObject):
    """Signals emitted by background workers."""

    progress = Signal(int, str)       # (percent, message)
    finished = Signal(object)         # result object
    error = Signal(str)               # error message
    log = Signal(str)                 # log message


class FetchWorker(QThread):
    """Background worker for fetching catalog data."""

    signals = WorkerSignals()

    def __init__(
        self,
        source: CatalogSource,
        n_stars: int = 2000,
        ra: float = 180.0,
        dec: float = 0.0,
        radius: float = 5.0,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.source = source
        self.n_stars = n_stars
        self.ra = ra
        self.dec = dec
        self.radius = radius
        self._cancelled = False

    def run(self):
        try:
            from ..data.fetcher import DataFetcher
            from ..data.preprocessing import preprocess_catalog

            self.signals.progress.emit(10, f"Fetching {self.source.value}...")
            self.signals.log.emit(f"Fetching {self.n_stars} stars from {self.source.value}")

            fetcher = DataFetcher()
            df = fetcher.fetch(
                self.source, n_stars=self.n_stars,
                ra_center=self.ra, dec_center=self.dec,
                radius_deg=self.radius,
            )
            if self._cancelled:
                return

            self.signals.progress.emit(60, "Preprocessing...")
            self.signals.log.emit(f"Preprocessing {len(df)} rows")
            df = preprocess_catalog(df)

            self.signals.progress.emit(100, "Done")
            self.signals.log.emit(f"Ready: {len(df)} stars after preprocessing")
            self.signals.finished.emit(df)

        except Exception as e:
            self.signals.error.emit(str(e))

    def cancel(self):
        self._cancelled = True


class AnalysisWorker(QThread):
    """Background worker for running the full analysis pipeline."""

    signals = WorkerSignals()

    def __init__(
        self,
        df: pd.DataFrame,
        config: DetectionConfig,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.df = df
        self.config = config
        self._cancelled = False

    def run(self):
        try:
            from ..analysis.hr_diagram import HRDiagramAnalyzer
            from ..analysis.stellar_lifetime import StellarLifetimeAnalyzer
            from ..analysis.kinematics import KinematicsAnalyzer
            from ..analysis.spectral import SpectralAnalyzer
            from ..analysis.technosignature import TechnosignatureAnalyzer
            from ..analysis.ml_pipeline import MLPipeline
            from ..analysis.ensemble import EnsembleScorer

            analyzers = [
                ("HR Diagram", HRDiagramAnalyzer(self.config)),
                ("Stellar Lifetime", StellarLifetimeAnalyzer(self.config)),
                ("Kinematics", KinematicsAnalyzer(self.config)),
                ("Spectral", SpectralAnalyzer(self.config)),
                ("Technosignature", TechnosignatureAnalyzer(self.config)),
                ("ML Pipeline", MLPipeline(self.config)),
            ]

            all_results: list[AnomalyResult] = []
            n = len(analyzers)

            for i, (name, analyzer) in enumerate(analyzers):
                if self._cancelled:
                    return
                pct = int((i / n) * 90)
                self.signals.progress.emit(pct, f"Running {name}...")
                self.signals.log.emit(f"Running {name} analysis...")

                try:
                    results = analyzer.analyze(self.df)
                    all_results.extend(results)
                    self.signals.log.emit(f"  {name}: {len(results)} anomalies")
                except Exception as e:
                    self.signals.log.emit(f"  {name} failed: {e}")

            self.signals.progress.emit(95, "Aggregating results...")
            scorer = EnsembleScorer()
            merged = scorer.aggregate(all_results)
            stats = scorer.summary_stats(merged)

            self.signals.progress.emit(100, "Analysis complete")
            self.signals.log.emit(
                f"Complete: {stats['total']} anomalies from "
                f"{stats.get('unique_stars', 0)} unique stars"
            )
            self.signals.finished.emit({"results": merged, "stats": stats})

        except Exception as e:
            self.signals.error.emit(str(e))

    def cancel(self):
        self._cancelled = True
