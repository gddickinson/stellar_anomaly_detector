"""HR diagram anomaly detection using multiple statistical and ML methods.

Methods implemented:
- DBSCAN noise-point detection on CMD space
- Kernel Density Estimation (low-density outliers)
- Gaussian Mixture Model (low-likelihood outliers)
- Isolation Forest on multi-color photometric features
- Robust z-score deviation from main-sequence model
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from ..core.models import AnomalyResult, AnomalyType, DetectionConfig

logger = logging.getLogger(__name__)


class HRDiagramAnalyzer:
    """Detect anomalous stars on the Hertzsprung-Russell diagram."""

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()

    def analyze(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Run all HR diagram methods and require internal consensus.

        Runs 4 correlated methods (z-score, DBSCAN, KDE, GMM) on the CMD, then
        only emits a result if a star is flagged by at least 2 of the 4 methods.
        This prevents each method independently flooding the ensemble with
        detections of normal population diversity (giants, subdwarfs, etc.).
        """
        if not self._has_required_columns(df):
            logger.warning("Missing required columns for HR analysis (bp_rp, abs_mag)")
            return []

        clean = df.dropna(subset=["bp_rp", "abs_mag"]).copy()
        if len(clean) < self.config.min_data_points:
            logger.warning("Too few data points (%d) for HR analysis", len(clean))
            return []

        # Collect raw detections from each method
        all_raw = []
        all_raw.extend(self._main_sequence_deviation(clean))
        all_raw.extend(self._dbscan_outliers(clean))
        all_raw.extend(self._kde_outliers(clean))
        all_raw.extend(self._gmm_outliers(clean))

        # Internal consensus: only keep stars flagged by 2+ methods
        from collections import defaultdict
        by_star: dict[str, list[AnomalyResult]] = defaultdict(list)
        for r in all_raw:
            by_star[r.star_id].append(r)

        results = []
        for star_id, detections in by_star.items():
            methods = set(d.detection_method for d in detections)
            if len(methods) >= 2:
                # Emit one combined result per star
                best = max(detections, key=lambda d: d.significance_score)
                best.description = (
                    f"{best.description} [HR consensus: {len(methods)}/4 methods]"
                )
                best.detection_method = f"hr_consensus({','.join(sorted(methods))})"
                results.append(best)

        logger.info(
            "HR diagram analysis: %d raw -> %d after consensus", len(all_raw), len(results)
        )
        return results

    def _main_sequence_deviation(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Flag stars with large deviations from the expected main-sequence locus."""
        results = []
        from ..models.stellar_evolution import StellarEvolutionModels

        models = StellarEvolutionModels()
        expected_mag = df["bp_rp"].apply(models.main_sequence_mag_from_color)
        deviation = df["abs_mag"].values - expected_mag.values

        if self.config.use_robust_statistics:
            center = np.nanmedian(deviation)
            scale = median_abs_deviation(deviation, nan_policy="omit")
            scale = scale if scale > 0 else np.nanstd(deviation)
        else:
            center = np.nanmean(deviation)
            scale = np.nanstd(deviation)

        z_scores = (deviation - center) / (scale + 1e-10)

        threshold = self.config.mad_threshold
        outlier_mask = np.abs(z_scores) > threshold

        for idx in df.index[outlier_mask]:
            z = z_scores[df.index.get_loc(idx)]
            dev = deviation[df.index.get_loc(idx)]
            direction = "above" if dev > 0 else "below"
            results.append(AnomalyResult(
                star_id=str(df.loc[idx, "source_id"]) if "source_id" in df.columns else str(idx),
                anomaly_type=AnomalyType.HR_OUTLIER,
                confidence=min(1.0, abs(z) / (2 * threshold)),
                significance_score=abs(z),
                parameters={"z_score": float(z), "deviation_mag": float(dev)},
                description=f"Star {direction} main sequence by {abs(dev):.2f} mag (z={z:.1f})",
                follow_up_priority=min(10, max(1, int(abs(z)))),
                detection_method="main_sequence_robust_zscore",
                statistical_tests={"z_score": float(z), "deviation": float(dev)},
                catalog_source=str(df.loc[idx, "catalog_source"])
                if "catalog_source" in df.columns else "",
            ))
        return results

    def _dbscan_outliers(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Use DBSCAN to find noise points in CMD space."""
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        features = StandardScaler().fit_transform(df[["bp_rp", "abs_mag"]].values)

        # Adaptive eps from k-distance heuristic
        eps = self._k_distance_eps(features, k=self.config.dbscan_min_samples)
        db = DBSCAN(eps=eps, min_samples=self.config.dbscan_min_samples)
        labels = db.fit_predict(features)

        noise_mask = labels == -1
        results = []
        for idx in df.index[noise_mask]:
            i = df.index.get_loc(idx)
            results.append(AnomalyResult(
                star_id=str(df.loc[idx, "source_id"]) if "source_id" in df.columns else str(idx),
                anomaly_type=AnomalyType.HR_OUTLIER,
                confidence=0.7,
                significance_score=6.0,
                parameters={"dbscan_eps": eps, "bp_rp": float(df.loc[idx, "bp_rp"]),
                             "abs_mag": float(df.loc[idx, "abs_mag"])},
                description="DBSCAN noise point in CMD space",
                follow_up_priority=5,
                detection_method="dbscan_cmd",
                statistical_tests={"cluster_label": -1},
                catalog_source=str(df.loc[idx, "catalog_source"])
                if "catalog_source" in df.columns else "",
            ))
        return results

    def _kde_outliers(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Flag stars in low-density regions of the CMD using Gaussian KDE."""
        from scipy.stats import gaussian_kde

        data = df[["bp_rp", "abs_mag"]].values.T
        try:
            kde = gaussian_kde(data, bw_method="scott")
        except np.linalg.LinAlgError:
            logger.warning("KDE failed due to singular matrix")
            return []

        log_density = kde.logpdf(data)
        threshold = np.percentile(log_density, 1)  # bottom 1%

        results = []
        low_mask = log_density < threshold
        for idx, is_low in zip(df.index, low_mask):
            if not is_low:
                continue
            i = df.index.get_loc(idx)
            results.append(AnomalyResult(
                star_id=str(df.loc[idx, "source_id"]) if "source_id" in df.columns else str(idx),
                anomaly_type=AnomalyType.HR_OUTLIER,
                confidence=0.65,
                significance_score=5.5,
                parameters={"log_density": float(log_density[i])},
                description=f"Low CMD density (log_density={log_density[i]:.2f})",
                follow_up_priority=4,
                detection_method="kde_density",
                statistical_tests={"log_density": float(log_density[i]),
                                   "threshold": float(threshold)},
                catalog_source=str(df.loc[idx, "catalog_source"])
                if "catalog_source" in df.columns else "",
            ))
        return results

    def _gmm_outliers(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Fit GMM to CMD and flag low-likelihood stars."""
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        features = StandardScaler().fit_transform(df[["bp_rp", "abs_mag"]].values)
        n_comp = min(self.config.gmm_n_components, len(df) // 10)
        if n_comp < 2:
            return []

        gmm = GaussianMixture(n_components=n_comp, covariance_type="full", random_state=42)
        gmm.fit(features)
        log_likelihood = gmm.score_samples(features)
        threshold = np.percentile(log_likelihood, 1)

        results = []
        for idx, ll in zip(df.index, log_likelihood):
            if ll >= threshold:
                continue
            results.append(AnomalyResult(
                star_id=str(df.loc[idx, "source_id"]) if "source_id" in df.columns else str(idx),
                anomaly_type=AnomalyType.HR_OUTLIER,
                confidence=0.6,
                significance_score=5.0,
                parameters={"gmm_log_likelihood": float(ll)},
                description=f"Low GMM likelihood in CMD (ll={ll:.2f})",
                follow_up_priority=4,
                detection_method="gmm_cmd",
                statistical_tests={"log_likelihood": float(ll), "threshold": float(threshold)},
                catalog_source=str(df.loc[idx, "catalog_source"])
                if "catalog_source" in df.columns else "",
            ))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _has_required_columns(self, df: pd.DataFrame) -> bool:
        return "bp_rp" in df.columns and "abs_mag" in df.columns

    @staticmethod
    def _k_distance_eps(X: np.ndarray, k: int = 5) -> float:
        """Estimate DBSCAN eps using the k-distance elbow heuristic."""
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        k_distances = np.sort(distances[:, -1])

        # Simple elbow: use the distance at the 95th percentile
        return float(np.percentile(k_distances, 95))
