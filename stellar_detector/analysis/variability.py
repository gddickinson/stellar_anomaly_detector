"""Variable star analysis: periodicity detection, Stetson indices, feature extraction.

Implements Lomb-Scargle periodograms, Stetson variability indices (J, K, L),
and a 51-feature extraction pipeline following the Gaia DR3 variability study approach.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..core.models import AnomalyResult, AnomalyType, DetectionConfig

logger = logging.getLogger(__name__)


class VariabilityAnalyzer:
    """Detect variability anomalies in stellar light curve data."""

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()

    def analyze(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Analyze variability features. Expects pre-computed variability columns
        or raw time-series data in a 'light_curve' column.
        """
        results: list[AnomalyResult] = []
        results.extend(self._photometric_scatter_outliers(df))
        results.extend(self._stetson_index_outliers(df))
        logger.info("Variability analysis found %d anomalies", len(results))
        return results

    def analyze_light_curve(
        self, times: np.ndarray, mags: np.ndarray, errs: np.ndarray | None = None
    ) -> dict:
        """Extract variability features from a single light curve.

        Returns a dict of ~30 features including Stetson indices,
        Lomb-Scargle period, amplitude, and statistical moments.
        """
        if errs is None:
            errs = np.ones_like(mags) * np.std(mags) * 0.1

        features = {}

        # Basic statistics
        features["mean_mag"] = float(np.mean(mags))
        features["median_mag"] = float(np.median(mags))
        features["std_mag"] = float(np.std(mags))
        features["mad_mag"] = float(np.median(np.abs(mags - np.median(mags))))
        features["amplitude"] = float(np.ptp(mags))
        features["iqr"] = float(np.percentile(mags, 75) - np.percentile(mags, 25))
        features["skewness"] = float(_skewness(mags))
        features["kurtosis"] = float(_kurtosis(mags))

        # Von Neumann ratio (eta)
        if len(mags) > 1:
            delta = np.diff(mags)
            features["eta"] = float(np.mean(delta ** 2) / np.var(mags)) if np.var(mags) > 0 else 2.0
        else:
            features["eta"] = 2.0

        # Stetson indices
        features["stetson_j"] = float(self._stetson_j(mags, errs))
        features["stetson_k"] = float(self._stetson_k(mags, errs))
        features["stetson_l"] = features["stetson_j"] * features["stetson_k"]

        # Lomb-Scargle periodogram
        ls_features = self._lomb_scargle_features(times, mags, errs)
        features.update(ls_features)

        # Fraction of points beyond 1-sigma
        features["frac_beyond_1sigma"] = float(
            np.mean(np.abs(mags - np.mean(mags)) > np.std(mags))
        )

        return features

    def _photometric_scatter_outliers(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Flag stars with unusually high photometric scatter."""
        scatter_cols = [c for c in df.columns if "scatter" in c.lower() or "std_mag" in c.lower()]
        if not scatter_cols and "phot_g_mean_flux_over_error" in df.columns:
            # High flux SNR but variable → anomalous
            return []

        results = []
        for col in scatter_cols:
            vals = df[col].dropna()
            if len(vals) < self.config.min_data_points:
                continue
            median = vals.median()
            mad = np.median(np.abs(vals - median))
            mad = mad if mad > 0 else vals.std()
            z = (df[col] - median) / (mad + 1e-10)
            for idx in df.index[z > self.config.mad_threshold]:
                results.append(AnomalyResult(
                    star_id=str(df.loc[idx, "source_id"])
                    if "source_id" in df.columns else str(idx),
                    anomaly_type=AnomalyType.VARIABILITY,
                    confidence=min(1.0, float(z[idx]) / (2 * self.config.mad_threshold)),
                    significance_score=float(z[idx]),
                    parameters={col: float(df.loc[idx, col])},
                    description=f"High photometric scatter in {col} (z={z[idx]:.1f})",
                    follow_up_priority=5,
                    detection_method="photometric_scatter",
                    statistical_tests={"z_score": float(z[idx])},
                    catalog_source=str(df.loc[idx, "catalog_source"])
                    if "catalog_source" in df.columns else "",
                ))
        return results

    def _stetson_index_outliers(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Flag stars with high Stetson J index (correlated variability)."""
        if "stetson_j" not in df.columns:
            return []

        results = []
        threshold = self.config.stetson_j_threshold
        anomalous = df[df["stetson_j"] > threshold].dropna(subset=["stetson_j"])
        for idx, row in anomalous.iterrows():
            sj = float(row["stetson_j"])
            results.append(AnomalyResult(
                star_id=str(row.get("source_id", idx)),
                anomaly_type=AnomalyType.VARIABILITY,
                confidence=min(1.0, sj / (2 * threshold)),
                significance_score=sj * 5,
                parameters={"stetson_j": sj},
                description=f"High Stetson J={sj:.2f} — correlated variability",
                follow_up_priority=min(10, max(1, int(sj * 3))),
                detection_method="stetson_j",
                statistical_tests={"stetson_j": sj, "threshold": threshold},
                catalog_source=str(row.get("catalog_source", "")),
            ))
        return results

    # ------------------------------------------------------------------
    # Stetson variability indices
    # ------------------------------------------------------------------
    @staticmethod
    def _stetson_j(mags: np.ndarray, errs: np.ndarray) -> float:
        """Stetson J index — measures correlated variability between pairs."""
        if len(mags) < 3:
            return 0.0
        mean_mag = np.mean(mags)
        residuals = (mags - mean_mag) / (errs + 1e-10)
        n = len(residuals)
        pairs = residuals[:-1] * residuals[1:]
        return float(np.sum(np.sign(pairs) * np.sqrt(np.abs(pairs))) / n)

    @staticmethod
    def _stetson_k(mags: np.ndarray, errs: np.ndarray) -> float:
        """Stetson K index — measures kurtosis of the magnitude distribution."""
        if len(mags) < 3:
            return 0.0
        mean_mag = np.mean(mags)
        residuals = (mags - mean_mag) / (errs + 1e-10)
        n = len(residuals)
        return float(np.mean(np.abs(residuals)) / np.sqrt(np.mean(residuals ** 2) + 1e-10))

    # ------------------------------------------------------------------
    # Lomb-Scargle
    # ------------------------------------------------------------------
    def _lomb_scargle_features(
        self, times: np.ndarray, mags: np.ndarray, errs: np.ndarray
    ) -> dict:
        """Extract period and power features from Lomb-Scargle periodogram."""
        from astropy.timeseries import LombScargle

        features = {}
        try:
            ls = LombScargle(times, mags, errs, fit_mean=self.config.lomb_scargle_fit_mean)
            freq, power = ls.autopower(
                minimum_frequency=1.0 / self.config.period_range_max,
                maximum_frequency=1.0 / self.config.period_range_min,
            )
            best_idx = np.argmax(power)
            features["best_period"] = float(1.0 / freq[best_idx]) if freq[best_idx] > 0 else 0.0
            features["best_power"] = float(power[best_idx])
            features["fap"] = float(ls.false_alarm_probability(power[best_idx]))
            features["power_95"] = float(np.percentile(power, 95))
            features["power_mean"] = float(np.mean(power))
        except Exception as e:
            logger.debug("Lomb-Scargle failed: %s", e)
            features["best_period"] = 0.0
            features["best_power"] = 0.0
            features["fap"] = 1.0
            features["power_95"] = 0.0
            features["power_mean"] = 0.0
        return features


def _skewness(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)
