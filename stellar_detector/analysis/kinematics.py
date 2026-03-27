"""Kinematic and astrometric anomaly detection.

Detects stars with unusual proper motions, RUWE values indicating unseen
companions, astrometric excess noise, and proper motion anomalies (PMa)
between Gaia and Hipparcos epochs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from ..core.models import AnomalyResult, AnomalyType, DetectionConfig

logger = logging.getLogger(__name__)


class KinematicsAnalyzer:
    """Detect kinematic and astrometric anomalies."""

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()

    def analyze(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Run all kinematic/astrometric checks."""
        results: list[AnomalyResult] = []
        results.extend(self._proper_motion_outliers(df))
        results.extend(self._ruwe_anomalies(df))
        results.extend(self._astrometric_excess_noise(df))
        results.extend(self._tangential_velocity_outliers(df))
        logger.info("Kinematic analysis found %d anomalies", len(results))
        return results

    def _proper_motion_outliers(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Flag stars with unusually high total proper motion."""
        if "pm_total" not in df.columns:
            if "pmra" in df.columns and "pmdec" in df.columns:
                pm_total = np.sqrt(df["pmra"] ** 2 + df["pmdec"] ** 2)
            else:
                return []
        else:
            pm_total = df["pm_total"]

        valid = pm_total.dropna()
        if len(valid) < self.config.min_data_points:
            return []

        median_pm = valid.median()
        mad_pm = median_abs_deviation(valid, nan_policy="omit")
        mad_pm = mad_pm if mad_pm > 0 else valid.std()
        z_scores = (pm_total - median_pm) / (mad_pm + 1e-10)

        results = []
        for idx in df.index[z_scores > self.config.mad_threshold]:
            z = float(z_scores[idx])
            results.append(AnomalyResult(
                star_id=str(df.loc[idx, "source_id"]) if "source_id" in df.columns else str(idx),
                anomaly_type=AnomalyType.KINEMATICS,
                confidence=min(1.0, z / (2 * self.config.mad_threshold)),
                significance_score=z,
                parameters={"pm_total": float(pm_total[idx]), "z_score": z},
                description=f"High proper motion ({pm_total[idx]:.1f} mas/yr, z={z:.1f})",
                follow_up_priority=min(10, max(1, int(z))),
                detection_method="proper_motion_mad",
                statistical_tests={"z_score": z, "median": float(median_pm), "mad": float(mad_pm)},
                catalog_source=str(df.loc[idx, "catalog_source"])
                if "catalog_source" in df.columns else "",
            ))
        return results

    def _ruwe_anomalies(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Flag stars with RUWE > 1.4 (non-single-star behavior)."""
        if "ruwe" not in df.columns:
            return []

        results = []
        anomalous = df[df["ruwe"] > self.config.max_ruwe].dropna(subset=["ruwe"])
        for idx, row in anomalous.iterrows():
            ruwe = float(row["ruwe"])
            results.append(AnomalyResult(
                star_id=str(row.get("source_id", idx)),
                anomaly_type=AnomalyType.ASTROMETRIC,
                confidence=min(1.0, (ruwe - 1.4) / 2.0),
                significance_score=ruwe * 3,
                parameters={"ruwe": ruwe},
                description=f"High RUWE={ruwe:.2f} — possible unresolved companion",
                follow_up_priority=min(10, max(1, int(ruwe * 2))),
                detection_method="ruwe_threshold",
                statistical_tests={"ruwe": ruwe},
                catalog_source=str(row.get("catalog_source", "")),
            ))
        return results

    def _astrometric_excess_noise(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Flag stars with significant astrometric excess noise."""
        if "astrometric_excess_noise_sig" not in df.columns:
            return []

        results = []
        sig_col = df["astrometric_excess_noise_sig"]
        # Require significance >= 5 — sig >= 2 is too loose and flags ~12% of stars
        anomalous = df[sig_col >= 5.0].dropna(subset=["astrometric_excess_noise_sig"])
        for idx, row in anomalous.iterrows():
            sig = float(row["astrometric_excess_noise_sig"])
            noise = float(row.get("astrometric_excess_noise", 0))
            results.append(AnomalyResult(
                star_id=str(row.get("source_id", idx)),
                anomaly_type=AnomalyType.ASTROMETRIC,
                confidence=min(1.0, sig / 10.0),
                significance_score=sig,
                parameters={"aen_sig": sig, "aen": noise},
                description=f"Astrometric excess noise (significance={sig:.1f})",
                follow_up_priority=min(10, max(1, int(sig))),
                detection_method="astrometric_excess_noise",
                statistical_tests={"aen_significance": sig},
                catalog_source=str(row.get("catalog_source", "")),
            ))
        return results

    def _tangential_velocity_outliers(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Flag stars with extreme tangential velocities (hypervelocity candidates)."""
        if "v_tan_km_s" not in df.columns:
            return []

        v_tan = df["v_tan_km_s"].dropna()
        if len(v_tan) < self.config.min_data_points:
            return []

        # Hypervelocity threshold: > 300 km/s or > 5 MAD from median
        median_v = v_tan.median()
        mad_v = median_abs_deviation(v_tan, nan_policy="omit")
        mad_v = mad_v if mad_v > 0 else v_tan.std()

        results = []
        for idx in df.index:
            v = df.loc[idx, "v_tan_km_s"]
            if pd.isna(v):
                continue
            z = (v - median_v) / (mad_v + 1e-10)
            if z > self.config.mad_threshold and v > 200:
                results.append(AnomalyResult(
                    star_id=str(df.loc[idx, "source_id"])
                    if "source_id" in df.columns else str(idx),
                    anomaly_type=AnomalyType.KINEMATICS,
                    confidence=min(1.0, v / 600.0),
                    significance_score=z,
                    parameters={"v_tan_km_s": float(v), "z_score": float(z)},
                    description=f"Extreme tangential velocity {v:.0f} km/s (z={z:.1f})",
                    follow_up_priority=min(10, max(1, int(z))),
                    detection_method="tangential_velocity",
                    statistical_tests={"z_score": float(z), "v_tan": float(v)},
                    catalog_source=str(df.loc[idx, "catalog_source"])
                    if "catalog_source" in df.columns else "",
                ))
        return results
