"""Technosignature detection: Dyson spheres, stellar engines, megastructures.

Implements the Project Hephaistos methodology for Dyson sphere candidate detection
using SED fitting across optical (Gaia) + NIR (2MASS) + MIR (WISE) photometry.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..core.models import AnomalyResult, AnomalyType, DetectionConfig

logger = logging.getLogger(__name__)

# Effective wavelengths (microns) for SED fitting
BAND_WAVELENGTHS = {
    "phot_g_mean_mag": 0.622,
    "bp_rp": None,  # color, not a band
    "Jmag": 1.235,
    "Hmag": 1.662,
    "Kmag": 2.159,
    "W1mag": 3.353,
    "W2mag": 4.603,
    "W3mag": 11.561,
    "W4mag": 22.088,
}

# Zero-point fluxes (Jy) for magnitude-to-flux conversion
ZERO_POINTS_JY = {
    "phot_g_mean_mag": 3228.75,
    "Jmag": 1594.0,
    "Hmag": 1024.0,
    "Kmag": 666.7,
    "W1mag": 309.54,
    "W2mag": 171.79,
    "W3mag": 31.674,
    "W4mag": 8.363,
}


class TechnosignatureAnalyzer:
    """Detect potential technosignatures using multi-band photometry."""

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()

    def analyze(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Run all technosignature detection methods."""
        results: list[AnomalyResult] = []
        results.extend(self._dyson_sphere_candidates(df))
        results.extend(self._infrared_excess_detection(df))
        results.extend(self._stellar_engine_candidates(df))
        logger.info("Technosignature analysis found %d candidates", len(results))
        return results

    def _dyson_sphere_candidates(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Project Hephaistos-style Dyson sphere detection via SED grid search.

        For each star with optical + IR photometry, fit a grid of models:
          Model = Stellar blackbody + Dyson sphere blackbody
          Parameters: T_DS (100-700 K), covering factor f (0.1-0.9)

        Candidates have low SED fit residuals AND significant MIR excess.
        """
        # Check for required multi-band photometry
        required_optical = {"phot_g_mean_mag"}
        required_ir = {"W3mag", "W4mag"}

        has_optical = required_optical.issubset(df.columns)
        has_ir = required_ir.issubset(df.columns)

        if not (has_optical and has_ir):
            logger.info("Dyson sphere search requires optical + WISE W3/W4 — skipping")
            return []

        results = []
        candidates = df.dropna(subset=["phot_g_mean_mag", "W3mag", "W4mag"])
        if "teff_gspphot" in candidates.columns:
            candidates = candidates.dropna(subset=["teff_gspphot"])
        else:
            logger.info("No stellar temperature data — skipping Dyson sphere SED fitting")
            return []

        for idx, row in candidates.iterrows():
            result = self._fit_dyson_model(row)
            if result is not None:
                results.append(result)

        return results

    def _fit_dyson_model(self, row: pd.Series) -> AnomalyResult | None:
        """Fit Dyson sphere model to a single star's SED."""
        teff_star = row["teff_gspphot"]

        # Collect available photometric bands
        observed = {}
        for band, wavelength in BAND_WAVELENGTHS.items():
            if wavelength is not None and band in row.index and pd.notna(row[band]):
                flux = ZERO_POINTS_JY.get(band, 1.0) * 10 ** (-row[band] / 2.5)
                observed[band] = {"wavelength": wavelength, "flux": flux}

        if len(observed) < 4:
            return None

        wavelengths = np.array([v["wavelength"] for v in observed.values()])
        fluxes = np.array([v["flux"] for v in observed.values()])

        # Normalize fluxes
        norm = np.max(fluxes)
        fluxes_norm = fluxes / norm

        # Grid search over Dyson sphere parameters
        best_rmse = np.inf
        best_t_ds = 0
        best_f = 0

        t_ds_range = np.arange(
            self.config.dyson_temp_min_k, self.config.dyson_temp_max_k + 50, 50
        )
        f_range = np.arange(
            self.config.dyson_covering_min, self.config.dyson_covering_max + 0.05, 0.05
        )

        for t_ds in t_ds_range:
            for f_cover in f_range:
                model_flux = self._combined_sed(wavelengths, teff_star, t_ds, f_cover)
                model_norm = model_flux / np.max(model_flux) if np.max(model_flux) > 0 else model_flux
                rmse = np.sqrt(np.mean((fluxes_norm - model_norm) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_t_ds = t_ds
                    best_f = f_cover

        # Also fit star-only model
        star_flux = _planck_flux(wavelengths, teff_star)
        star_norm = star_flux / np.max(star_flux) if np.max(star_flux) > 0 else star_flux
        star_rmse = np.sqrt(np.mean((fluxes_norm - star_norm) ** 2))

        # Candidate if DS model is significantly better AND within RMSE threshold
        improvement = star_rmse - best_rmse
        if best_rmse < self.config.dyson_sed_rmse_max and improvement > 0.05:
            # Check W3/W4 SNR requirements
            w3_err = row.get("e_W3mag", 0.3)
            w4_err = row.get("e_W4mag", 0.3)
            w3_snr = 1.0 / (w3_err + 0.01) if pd.notna(w3_err) else 5.0
            w4_snr = 1.0 / (w4_err + 0.01) if pd.notna(w4_err) else 5.0

            if w3_snr >= self.config.dyson_snr_min and w4_snr >= self.config.dyson_snr_min:
                confidence = min(1.0, improvement / 0.2)
                return AnomalyResult(
                    star_id=str(row.get("source_id", row.name)),
                    anomaly_type=AnomalyType.DYSON_SPHERE,
                    confidence=confidence,
                    significance_score=improvement * 50,
                    parameters={
                        "dyson_temp_k": float(best_t_ds),
                        "covering_factor": float(best_f),
                        "sed_rmse": float(best_rmse),
                        "star_only_rmse": float(star_rmse),
                        "improvement": float(improvement),
                        "teff_star": float(teff_star),
                    },
                    description=(
                        f"Dyson sphere candidate: T_DS={best_t_ds:.0f}K, "
                        f"f={best_f:.2f}, RMSE={best_rmse:.3f} "
                        f"(star-only RMSE={star_rmse:.3f})"
                    ),
                    follow_up_priority=10,
                    detection_method="hephaistos_sed_grid",
                    statistical_tests={
                        "sed_rmse": float(best_rmse),
                        "improvement": float(improvement),
                    },
                    catalog_source=str(row.get("catalog_source", "")),
                    observational_recommendations=[
                        "High-resolution spectroscopy to rule out dust shell",
                        "Radio continuum observation to check for radio emission",
                        "Time-series photometry to check for transit-like dips",
                    ],
                )
        return None

    def _infrared_excess_detection(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Detect stars with significant infrared excess (W1-W2 or W2-W3 color anomaly).

        IR excess beyond normal photospheric emission can indicate circumstellar
        material, debris disks, or artificial structures.
        """
        results = []
        from scipy.stats import median_abs_deviation

        # Use robust statistical outlier detection on WISE colors
        # rather than fixed thresholds, since color distributions vary by field
        for color_col, min_absolute, label in [
            ("W1_W2", 0.5, "W1-W2"),
            ("W2_W3", 1.0, "W2-W3"),
        ]:
            if color_col not in df.columns:
                continue
            vals = df[color_col].dropna()
            if len(vals) < self.config.min_data_points:
                continue

            median_val = vals.median()
            mad = median_abs_deviation(vals, nan_policy="omit")
            mad = mad if mad > 0 else vals.std()
            threshold = self.config.mad_threshold

            for idx in df.index:
                val = df.loc[idx, color_col]
                if pd.isna(val):
                    continue
                z = (val - median_val) / (mad + 1e-10)
                # Must be a positive outlier AND exceed minimum absolute threshold
                if z > threshold and val > min_absolute:
                    priority = 7 if color_col == "W1_W2" else 8
                    results.append(AnomalyResult(
                        star_id=str(df.loc[idx, "source_id"])
                        if "source_id" in df.columns else str(idx),
                        anomaly_type=AnomalyType.INFRARED_EXCESS,
                        confidence=min(1.0, z / (2 * threshold)),
                        significance_score=z,
                        parameters={color_col: float(val), "z_score": float(z)},
                        description=f"{label} infrared excess: {val:.2f} mag (z={z:.1f})",
                        follow_up_priority=priority,
                        detection_method="wise_color_excess",
                        statistical_tests={color_col: float(val), "z_score": float(z)},
                        catalog_source=str(df.loc[idx, "catalog_source"])
                        if "catalog_source" in df.columns else "",
                    ))

        return results

    def _stellar_engine_candidates(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Flag stars with anomalous proper motion combined with IR excess.

        A Shkadov thruster would cause anomalous acceleration detectable as
        unusual proper motion for the star's distance and galactic location.
        """
        if "pm_total" not in df.columns or "distance_pc" not in df.columns:
            return []

        results = []
        clean = df.dropna(subset=["pm_total", "distance_pc"])

        for idx, row in clean.iterrows():
            pm = row["pm_total"]
            dist = row["distance_pc"]
            v_tan = 4.74047 * (pm / 1000.0) * dist

            has_ir_excess = False
            if "W1_W2" in row.index and pd.notna(row.get("W1_W2")):
                has_ir_excess = row["W1_W2"] > 0.3

            # Candidate: extreme tangential velocity + IR anomaly
            if v_tan > 200 and has_ir_excess:
                results.append(AnomalyResult(
                    star_id=str(row.get("source_id", idx)),
                    anomaly_type=AnomalyType.STELLAR_ENGINE,
                    confidence=min(1.0, v_tan / 500.0),
                    significance_score=v_tan / 50.0,
                    parameters={
                        "v_tan_km_s": float(v_tan),
                        "pm_total": float(pm),
                        "distance_pc": float(dist),
                    },
                    description=(
                        f"Stellar engine candidate: v_tan={v_tan:.0f} km/s with IR excess"
                    ),
                    follow_up_priority=9,
                    detection_method="kinematics_ir_combined",
                    statistical_tests={"v_tan": float(v_tan)},
                    catalog_source=str(row.get("catalog_source", "")),
                    observational_recommendations=[
                        "High-precision radial velocity monitoring",
                        "Check for asymmetric transit signatures",
                    ],
                ))

        return results

    @staticmethod
    def _combined_sed(
        wavelengths: np.ndarray, t_star: float, t_ds: float, f_cover: float
    ) -> np.ndarray:
        """Combined SED of a star partially enclosed by a Dyson sphere."""
        star_flux = (1 - f_cover) * _planck_flux(wavelengths, t_star)
        ds_flux = f_cover * _planck_flux(wavelengths, t_ds)
        return star_flux + ds_flux


def _planck_flux(wavelengths_um: np.ndarray, temperature: float) -> np.ndarray:
    """Planck function B_lambda in relative units.

    Args:
        wavelengths_um: Wavelengths in microns.
        temperature: Blackbody temperature in Kelvin.
    """
    h = 6.626e-34
    c = 2.998e8
    k_b = 1.381e-23
    wavelengths_m = wavelengths_um * 1e-6

    with np.errstate(over="ignore", divide="ignore"):
        exponent = h * c / (wavelengths_m * k_b * temperature)
        exponent = np.clip(exponent, 0, 500)
        return (2 * h * c ** 2 / wavelengths_m ** 5) / (np.exp(exponent) - 1 + 1e-30)
