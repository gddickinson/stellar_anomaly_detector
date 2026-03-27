"""Stellar lifetime anomaly detection.

Flags stars whose inferred age exceeds the expected main-sequence lifetime
for their mass and metallicity — a potential technosignature indicator.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..core.models import AnomalyResult, AnomalyType, DetectionConfig
from ..models.stellar_evolution import StellarEvolutionModels

logger = logging.getLogger(__name__)


class StellarLifetimeAnalyzer:
    """Detect stars with anomalous ages relative to their expected lifetimes."""

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()
        self.models = StellarEvolutionModels()

    def analyze(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Identify stars that appear older than physically expected."""
        results: list[AnomalyResult] = []

        required = {"abs_mag", "teff_gspphot"}
        if not required.issubset(df.columns):
            logger.warning("Missing columns for lifetime analysis: %s", required - set(df.columns))
            return results

        clean = df.dropna(subset=list(required)).copy()
        if len(clean) < self.config.min_data_points:
            return results

        # Estimate mass from luminosity (mass-luminosity relation)
        clean["mass_solar"] = clean["abs_mag"].apply(self.models.mass_from_abs_mag)

        # Expected main-sequence lifetime
        metallicity = clean.get("mh_gspphot", pd.Series(0.0, index=clean.index)).fillna(0.0)
        clean["expected_lifetime_gyr"] = self.models.main_sequence_lifetime(
            clean["mass_solar"].values, metallicity.values
        )

        # Estimate age via isochrone position (simplified: fraction of MS band traversed)
        clean["age_fraction"] = self._estimate_age_fraction(clean)

        # Flag stars where age_fraction > 0.9 (near end of expected MS lifetime)
        anomalous = clean[clean["age_fraction"] > 0.9]

        for idx, row in anomalous.iterrows():
            age_frac = row["age_fraction"]
            expected_lt = row["expected_lifetime_gyr"]
            mass = row["mass_solar"]

            confidence = min(1.0, (age_frac - 0.9) / 0.1 * 0.5 + 0.5)

            results.append(AnomalyResult(
                star_id=str(row.get("source_id", idx)),
                anomaly_type=AnomalyType.LIFETIME,
                confidence=confidence,
                significance_score=age_frac * 10,
                parameters={
                    "mass_solar": float(mass),
                    "expected_lifetime_gyr": float(expected_lt),
                    "age_fraction": float(age_frac),
                    "teff": float(row["teff_gspphot"]),
                },
                description=(
                    f"Star at {age_frac:.0%} of expected MS lifetime "
                    f"({expected_lt:.1f} Gyr for {mass:.2f} M_sun)"
                ),
                follow_up_priority=min(10, max(1, int(age_frac * 10))),
                detection_method="lifetime_isochrone",
                statistical_tests={"age_fraction": float(age_frac)},
                catalog_source=str(row.get("catalog_source", "")),
            ))

        logger.info("Lifetime analysis found %d anomalies", len(results))
        return results

    def _estimate_age_fraction(self, df: pd.DataFrame) -> pd.Series:
        """Estimate how far through the main-sequence band a star has evolved.

        Uses the deviation from the ZAMS (zero-age main sequence) relative to
        the TAMS (terminal-age main sequence) at the star's temperature.
        Returns a value in [0, 1] where 1 means at the TAMS.
        """
        zams_mag = df["teff_gspphot"].apply(self.models.zams_mag_from_teff)
        tams_mag = df["teff_gspphot"].apply(self.models.tams_mag_from_teff)

        band_width = zams_mag - tams_mag  # ZAMS is fainter (larger mag) for MS
        band_width = band_width.replace(0, np.nan)

        fraction = (zams_mag - df["abs_mag"]) / band_width
        return fraction.clip(0, 1.5).fillna(0.5)
