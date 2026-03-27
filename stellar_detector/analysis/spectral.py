"""Spectral and chemical abundance anomaly detection.

Methods:
- Metallicity outlier detection (robust z-score, Isolation Forest)
- Chemical abundance ratio anomalies
- Autoencoder-based spectral anomaly detection (optional, requires TensorFlow)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from ..core.models import AnomalyResult, AnomalyType, DetectionConfig

logger = logging.getLogger(__name__)


class SpectralAnalyzer:
    """Detect chemical and spectral anomalies."""

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()

    def analyze(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Run all spectral/chemical methods with internal consensus.

        Only emits a result if a star is flagged by 2+ of the 3 methods, since
        metallicity outlier + chemical IF + Teff-[M/H] LOF are correlated.
        """
        all_raw: list[AnomalyResult] = []
        all_raw.extend(self._metallicity_outliers(df))
        all_raw.extend(self._chemical_ratio_anomalies(df))
        all_raw.extend(self._temperature_metallicity_outliers(df))

        from collections import defaultdict
        by_star: dict[str, list[AnomalyResult]] = defaultdict(list)
        for r in all_raw:
            by_star[r.star_id].append(r)

        results = []
        for star_id, detections in by_star.items():
            methods = set(d.detection_method for d in detections)
            if len(methods) >= 2:
                best = max(detections, key=lambda d: d.significance_score)
                results.append(best)

        logger.info(
            "Spectral analysis: %d raw -> %d after consensus", len(all_raw), len(results)
        )
        return results

    def analyze_spectra_autoencoder(self, spectra: np.ndarray) -> np.ndarray:
        """Detect spectral anomalies using an autoencoder ensemble.

        Trains an ensemble of autoencoders and returns reconstruction error
        for each spectrum. High error = anomalous.

        Architecture follows the MaNGA Stellar Library approach:
        Input -> 2048 -> 512 -> 128 -> 32 -> 10 (latent) -> 32 -> 128 -> 512 -> 2048 -> Input

        Args:
            spectra: Array of shape (n_stars, n_wavelengths) with normalized flux.

        Returns:
            Array of reconstruction errors (MSE) for each spectrum.
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            logger.warning("TensorFlow not available — skipping autoencoder analysis")
            return np.zeros(len(spectra))

        input_dim = spectra.shape[1]
        errors_ensemble = []

        for i in range(self.config.autoencoder_n_ensemble):
            model = self._build_autoencoder(input_dim, keras)

            # Train on a random subset
            rng = np.random.default_rng(seed=i)
            train_idx = rng.choice(len(spectra), size=min(2000, len(spectra)), replace=False)
            val_idx = np.setdiff1d(np.arange(len(spectra)), train_idx)[:770]

            model.fit(
                spectra[train_idx], spectra[train_idx],
                validation_data=(spectra[val_idx], spectra[val_idx]),
                epochs=200, batch_size=32, verbose=0,
                callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)],
            )

            reconstructed = model.predict(spectra, verbose=0)
            mse = np.mean((spectra - reconstructed) ** 2, axis=1)
            errors_ensemble.append(mse)

        return np.median(errors_ensemble, axis=0)

    def _metallicity_outliers(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Flag stars with extreme metallicity [M/H]."""
        if "mh_gspphot" not in df.columns:
            return []

        mh = df["mh_gspphot"].dropna()
        if len(mh) < self.config.min_data_points:
            return []

        median_mh = mh.median()
        mad_mh = median_abs_deviation(mh, nan_policy="omit")
        mad_mh = mad_mh if mad_mh > 0 else mh.std()

        results = []
        for idx in df.index:
            val = df.loc[idx, "mh_gspphot"]
            if pd.isna(val):
                continue
            z = (val - median_mh) / (mad_mh + 1e-10)
            if abs(z) > self.config.mad_threshold:
                direction = "metal-rich" if val > median_mh else "metal-poor"
                results.append(AnomalyResult(
                    star_id=str(df.loc[idx, "source_id"])
                    if "source_id" in df.columns else str(idx),
                    anomaly_type=AnomalyType.METALLICITY,
                    confidence=min(1.0, abs(z) / (2 * self.config.mad_threshold)),
                    significance_score=abs(z),
                    parameters={"mh_gspphot": float(val), "z_score": float(z)},
                    description=f"Extreme metallicity [M/H]={val:.2f} ({direction}, z={z:.1f})",
                    follow_up_priority=min(10, max(1, int(abs(z)))),
                    detection_method="metallicity_mad",
                    statistical_tests={"z_score": float(z)},
                    catalog_source=str(df.loc[idx, "catalog_source"])
                    if "catalog_source" in df.columns else "",
                ))
        return results

    def _chemical_ratio_anomalies(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Detect anomalous chemical abundance ratios.

        Checks for unusual relationships between metallicity, temperature,
        and surface gravity that don't match standard stellar models.
        """
        required = {"mh_gspphot", "teff_gspphot", "logg_gspphot"}
        if not required.issubset(df.columns):
            return []

        clean = df.dropna(subset=list(required))
        if len(clean) < self.config.min_data_points:
            return []

        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return []

        features = StandardScaler().fit_transform(
            clean[["mh_gspphot", "teff_gspphot", "logg_gspphot"]].values
        )
        iso = IsolationForest(
            n_estimators=self.config.isolation_n_estimators,
            contamination=self.config.isolation_contamination,
            max_samples=min(self.config.isolation_max_samples, len(clean)),
            random_state=42,
        )
        labels = iso.fit_predict(features)
        scores = iso.score_samples(features)

        results = []
        for i, (idx, row) in enumerate(clean.iterrows()):
            if labels[i] == -1:
                results.append(AnomalyResult(
                    star_id=str(row.get("source_id", idx)),
                    anomaly_type=AnomalyType.CHEMICAL,
                    confidence=min(1.0, abs(scores[i]) * 2),
                    significance_score=abs(scores[i]) * 10,
                    parameters={
                        "mh": float(row["mh_gspphot"]),
                        "teff": float(row["teff_gspphot"]),
                        "logg": float(row["logg_gspphot"]),
                        "iso_score": float(scores[i]),
                    },
                    description=(
                        f"Anomalous chemistry: [M/H]={row['mh_gspphot']:.2f}, "
                        f"Teff={row['teff_gspphot']:.0f}K, logg={row['logg_gspphot']:.2f}"
                    ),
                    follow_up_priority=5,
                    detection_method="chemical_isolation_forest",
                    statistical_tests={"isolation_score": float(scores[i])},
                    catalog_source=str(row.get("catalog_source", "")),
                ))
        return results

    def _temperature_metallicity_outliers(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Flag stars that are unusual in the Teff vs [M/H] plane."""
        if "teff_gspphot" not in df.columns or "mh_gspphot" not in df.columns:
            return []

        clean = df.dropna(subset=["teff_gspphot", "mh_gspphot"])
        if len(clean) < 50:
            return []

        try:
            from sklearn.neighbors import LocalOutlierFactor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return []

        features = StandardScaler().fit_transform(
            clean[["teff_gspphot", "mh_gspphot"]].values
        )
        lof = LocalOutlierFactor(
            n_neighbors=min(self.config.lof_n_neighbors, len(clean) - 1),
            contamination=self.config.lof_contamination,
        )
        labels = lof.fit_predict(features)
        lof_scores = lof.negative_outlier_factor_

        results = []
        for i, (idx, row) in enumerate(clean.iterrows()):
            if labels[i] == -1:
                results.append(AnomalyResult(
                    star_id=str(row.get("source_id", idx)),
                    anomaly_type=AnomalyType.CHEMICAL,
                    confidence=min(1.0, abs(lof_scores[i]) / 3.0),
                    significance_score=abs(lof_scores[i]) * 5,
                    parameters={
                        "teff": float(row["teff_gspphot"]),
                        "mh": float(row["mh_gspphot"]),
                        "lof_score": float(lof_scores[i]),
                    },
                    description=(
                        f"Teff-metallicity outlier: Teff={row['teff_gspphot']:.0f}K, "
                        f"[M/H]={row['mh_gspphot']:.2f} (LOF={lof_scores[i]:.2f})"
                    ),
                    follow_up_priority=4,
                    detection_method="teff_metallicity_lof",
                    statistical_tests={"lof_score": float(lof_scores[i])},
                    catalog_source=str(row.get("catalog_source", "")),
                ))
        return results

    def _build_autoencoder(self, input_dim: int, keras):
        """Build the symmetric autoencoder model."""
        latent_dim = self.config.autoencoder_latent_dim
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(2048, activation="relu"),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(latent_dim, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(2048, activation="relu"),
            keras.layers.Dense(input_dim, activation="linear"),
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        return model
