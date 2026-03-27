"""Ensemble machine learning pipeline for anomaly detection.

Combines Isolation Forest, Local Outlier Factor, One-Class SVM, and optional
adaptive methods (coniferest) into a unified scoring pipeline.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ..core.models import AnomalyResult, AnomalyType, DetectionConfig

logger = logging.getLogger(__name__)


class MLPipeline:
    """Multi-algorithm ML anomaly detection pipeline."""

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()
        self.scaler = StandardScaler()
        self._fitted = False

    def analyze(
        self,
        df: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> list[AnomalyResult]:
        """Run the full ML pipeline on a catalog dataframe.

        Args:
            df: Catalog data with numeric feature columns.
            feature_columns: Columns to use as features. If None, auto-selects
                numeric columns excluding IDs and coordinates.
        """
        if feature_columns is None:
            feature_columns = self._auto_select_features(df)

        if len(feature_columns) < 2:
            logger.warning("Too few feature columns (%d) for ML pipeline", len(feature_columns))
            return []

        clean = df.dropna(subset=feature_columns).copy()
        if len(clean) < max(self.config.min_data_points, 30):
            logger.warning("Too few clean rows (%d) for ML pipeline", len(clean))
            return []

        X = self.scaler.fit_transform(clean[feature_columns].values)
        self._fitted = True

        # Run each algorithm
        scores = {}
        scores["isolation_forest"] = self._run_isolation_forest(X)
        scores["lof"] = self._run_lof(X)
        scores["ocsvm"] = self._run_ocsvm(X)
        scores["adaptive_if"] = self._run_adaptive_isolation_forest(X)

        # Combine into ensemble score
        ensemble_scores = self._combine_scores(scores)
        anomaly_mask = ensemble_scores > self.config.ensemble_threshold

        results = []
        for i, (idx, row) in enumerate(clean.iterrows()):
            if not anomaly_mask[i]:
                continue

            # Identify which algorithms flagged this star
            methods = [name for name, s in scores.items() if s is not None and s[i] > 0.5]

            results.append(AnomalyResult(
                star_id=str(row.get("source_id", idx)),
                anomaly_type=AnomalyType.PHOTOMETRY,
                confidence=float(ensemble_scores[i]),
                significance_score=float(ensemble_scores[i]) * 10,
                parameters={
                    "ensemble_score": float(ensemble_scores[i]),
                    "methods_flagged": methods,
                    **{f"score_{k}": float(v[i]) for k, v in scores.items() if v is not None},
                },
                description=(
                    f"ML ensemble anomaly (score={ensemble_scores[i]:.2f}, "
                    f"flagged by {len(methods)}/{len(scores)} methods)"
                ),
                follow_up_priority=min(10, max(1, int(ensemble_scores[i] * 10))),
                detection_method="ml_ensemble",
                statistical_tests={
                    k: float(v[i]) for k, v in scores.items() if v is not None
                },
                catalog_source=str(row.get("catalog_source", "")),
            ))

        logger.info("ML pipeline found %d anomalies from %d stars", len(results), len(clean))
        return results

    def _run_isolation_forest(self, X: np.ndarray) -> np.ndarray:
        """Isolation Forest anomaly scores normalized to [0, 1]."""
        iso = IsolationForest(
            n_estimators=self.config.isolation_n_estimators,
            contamination=self.config.isolation_contamination,
            max_samples=min(self.config.isolation_max_samples, len(X)),
            random_state=42,
            n_jobs=self.config.n_jobs,
        )
        iso.fit(X)
        raw_scores = iso.score_samples(X)
        # Invert and normalize: more negative = more anomalous -> higher score
        return _normalize_scores(-raw_scores)

    def _run_lof(self, X: np.ndarray) -> np.ndarray:
        """Local Outlier Factor scores normalized to [0, 1]."""
        n_neighbors = min(self.config.lof_n_neighbors, len(X) - 1)
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.config.lof_contamination,
            n_jobs=self.config.n_jobs,
        )
        lof.fit_predict(X)
        raw_scores = lof.negative_outlier_factor_
        return _normalize_scores(-raw_scores)

    def _run_ocsvm(self, X: np.ndarray) -> np.ndarray | None:
        """One-Class SVM scores. Skips if dataset is too large (>5000 rows)."""
        if len(X) > 5000:
            logger.info("Skipping One-Class SVM (n=%d > 5000)", len(X))
            return None

        ocsvm = OneClassSVM(nu=self.config.ocsvm_nu, kernel="rbf", gamma="scale")
        ocsvm.fit(X)
        raw_scores = ocsvm.score_samples(X)
        return _normalize_scores(-raw_scores)

    def _run_adaptive_isolation_forest(self, X: np.ndarray) -> np.ndarray | None:
        """Run adaptive Isolation Forest from coniferest if available."""
        try:
            from coniferest.isoforest import IsolationForest as AdaptiveIF

            aif = AdaptiveIF(n_trees=200, random_seed=42)
            aif.fit(X)
            raw_scores = aif.score_samples(X)
            return _normalize_scores(-raw_scores)
        except ImportError:
            logger.debug("coniferest not available — skipping adaptive IF")
            return None

    def _combine_scores(self, scores: dict[str, np.ndarray | None]) -> np.ndarray:
        """Combine algorithm scores into an ensemble score (weighted mean)."""
        valid = {k: v for k, v in scores.items() if v is not None}
        if not valid:
            return np.zeros(0)

        # Weight: IF and LOF get higher weight as they're more reliable for stellar data
        weights = {
            "isolation_forest": 2.0,
            "lof": 1.5,
            "ocsvm": 1.0,
            "adaptive_if": 2.0,
        }

        weighted_sum = np.zeros_like(next(iter(valid.values())))
        total_weight = 0.0
        for name, s in valid.items():
            w = weights.get(name, 1.0)
            weighted_sum += w * s
            total_weight += w

        return weighted_sum / total_weight

    @staticmethod
    def _auto_select_features(df: pd.DataFrame) -> list[str]:
        """Auto-select numeric feature columns, excluding IDs and coordinates."""
        exclude_patterns = [
            "source_id", "catalog_source", "is_injected", "quality_score",
        ]
        candidates = df.select_dtypes(include=[np.number]).columns.tolist()
        return [
            c for c in candidates
            if not any(pat in c.lower() for pat in exclude_patterns)
            and df[c].notna().sum() > len(df) * 0.5  # at least 50% non-null
        ]


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range using min-max scaling."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-10:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)
