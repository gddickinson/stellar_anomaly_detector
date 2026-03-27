"""Data preprocessing: quality filtering, derived quantities, normalization."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..core.models import DetectionConfig

logger = logging.getLogger(__name__)


def preprocess_catalog(df: pd.DataFrame, config: DetectionConfig | None = None) -> pd.DataFrame:
    """Run the full preprocessing pipeline on a catalog dataframe."""
    config = config or DetectionConfig()
    df = df.copy()
    df = compute_derived_quantities(df)
    df = apply_quality_filters(df, config)
    df = compute_quality_score(df)
    logger.info("Preprocessing complete: %d rows retained", len(df))
    return df


def compute_derived_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance, absolute magnitude, total proper motion, and color indices."""
    # Distance from parallax (mas -> pc)
    if "parallax" in df.columns:
        valid_plx = df["parallax"] > 0
        df.loc[valid_plx, "distance_pc"] = 1000.0 / df.loc[valid_plx, "parallax"]

        # Absolute magnitude: M = m - 5*log10(d) + 5
        if "phot_g_mean_mag" in df.columns:
            valid = valid_plx & df["phot_g_mean_mag"].notna()
            df.loc[valid, "abs_mag"] = (
                df.loc[valid, "phot_g_mean_mag"]
                - 5.0 * np.log10(df.loc[valid, "distance_pc"])
                + 5.0
            )

    # Total proper motion
    if "pmra" in df.columns and "pmdec" in df.columns:
        df["pm_total"] = np.sqrt(df["pmra"] ** 2 + df["pmdec"] ** 2)

    # Tangential velocity (km/s) = 4.74047 * pm(arcsec/yr) * distance(pc)
    if "pm_total" in df.columns and "distance_pc" in df.columns:
        df["v_tan_km_s"] = 4.74047 * (df["pm_total"] / 1000.0) * df["distance_pc"]

    # 2MASS color indices
    if "Jmag" in df.columns and "Hmag" in df.columns:
        df["J_H"] = df["Jmag"] - df["Hmag"]
    if "Hmag" in df.columns and "Kmag" in df.columns:
        df["H_K"] = df["Hmag"] - df["Kmag"]

    # WISE color indices
    if "W1mag" in df.columns and "W2mag" in df.columns:
        df["W1_W2"] = df["W1mag"] - df["W2mag"]
    if "W2mag" in df.columns and "W3mag" in df.columns:
        df["W2_W3"] = df["W2mag"] - df["W3mag"]

    # Luminosity from absolute magnitude (solar units)
    if "abs_mag" in df.columns:
        solar_abs_g = 4.67
        df["luminosity_solar"] = 10 ** ((solar_abs_g - df["abs_mag"]) / 2.5)

    return df


def apply_quality_filters(df: pd.DataFrame, config: DetectionConfig) -> pd.DataFrame:
    """Filter rows based on Gaia DR3 quality criteria."""
    n_before = len(df)

    if "parallax" in df.columns and "parallax_error" in df.columns:
        plx_snr = df["parallax"] / df["parallax_error"]
        mask = plx_snr >= config.min_parallax_over_error
        df = df[mask | plx_snr.isna()]

    if "ruwe" in df.columns:
        # Keep rows with RUWE <= threshold OR where RUWE is missing
        df = df[(df["ruwe"] <= config.max_ruwe) | df["ruwe"].isna()]

    if "phot_bp_rp_excess_factor" in df.columns:
        excess = df["phot_bp_rp_excess_factor"]
        in_range = (excess >= config.min_phot_bp_rp_excess) & (
            excess <= config.max_phot_bp_rp_excess
        )
        df = df[in_range | excess.isna()]

    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info("Quality filters removed %d / %d rows", n_removed, n_before)
    return df


def compute_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a 0-1 quality score based on available precision indicators."""
    score = pd.Series(0.5, index=df.index)

    if "parallax" in df.columns and "parallax_error" in df.columns:
        plx_snr = (df["parallax"] / df["parallax_error"]).clip(0, 100)
        score += 0.2 * (plx_snr / 100.0)

    if "ruwe" in df.columns:
        ruwe_score = (2.0 - df["ruwe"].clip(0.5, 2.0)) / 1.5
        score += 0.15 * ruwe_score

    if "phot_g_mean_flux_over_error" in df.columns:
        flux_snr = df["phot_g_mean_flux_over_error"].clip(0, 1000)
        score += 0.15 * (flux_snr / 1000.0)

    df["quality_score"] = score.clip(0, 1)
    return df


def normalize_features(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    """Robust normalization (median / MAD) for specified columns."""
    stats = {}
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 3:
            continue
        median = vals.median()
        mad = np.median(np.abs(vals - median))
        mad = mad if mad > 0 else vals.std()
        df[f"{col}_normalized"] = (df[col] - median) / (mad + 1e-10)
        stats[col] = {"median": median, "mad": mad}
    return df, stats
