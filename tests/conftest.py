"""Shared test fixtures for the stellar anomaly detector test suite."""

import numpy as np
import pandas as pd
import pytest

from stellar_detector.core.models import CatalogSource, DetectionConfig
from stellar_detector.data.fetcher import DataFetcher
from stellar_detector.data.preprocessing import preprocess_catalog


@pytest.fixture
def config():
    """Default detection configuration."""
    return DetectionConfig()


@pytest.fixture
def synthetic_raw():
    """Raw synthetic catalog (before preprocessing)."""
    fetcher = DataFetcher()
    return fetcher.fetch(CatalogSource.SYNTHETIC, n_stars=500)


@pytest.fixture
def synthetic_df(synthetic_raw, config):
    """Preprocessed synthetic catalog ready for analysis."""
    return preprocess_catalog(synthetic_raw, config)


@pytest.fixture
def small_df():
    """Minimal DataFrame for quick unit tests (50 stars)."""
    rng = np.random.default_rng(99)
    n = 50
    return pd.DataFrame({
        "source_id": [f"TEST_{i:04d}" for i in range(n)],
        "ra": rng.uniform(170, 190, n),
        "dec": rng.uniform(-10, 10, n),
        "parallax": rng.uniform(2, 40, n),
        "parallax_error": rng.uniform(0.01, 0.3, n),
        "pmra": rng.normal(0, 8, n),
        "pmdec": rng.normal(0, 8, n),
        "phot_g_mean_mag": rng.uniform(5, 16, n),
        "bp_rp": rng.uniform(-0.2, 3.5, n),
        "teff_gspphot": 10 ** rng.uniform(3.5, 4.5, n),
        "logg_gspphot": rng.uniform(2.0, 4.8, n),
        "mh_gspphot": rng.normal(-0.1, 0.4, n),
        "ruwe": rng.lognormal(0.0, 0.25, n),
        "catalog_source": "synthetic",
    })


@pytest.fixture
def small_processed(small_df, config):
    """Small preprocessed DataFrame."""
    return preprocess_catalog(small_df, config)


@pytest.fixture
def sample_light_curve():
    """Synthetic light curve with periodic variability."""
    rng = np.random.default_rng(7)
    n = 200
    times = np.sort(rng.uniform(0, 100, n))
    period = 5.3
    amplitude = 0.4
    mags = 12.0 + amplitude * np.sin(2 * np.pi * times / period) + rng.normal(0, 0.05, n)
    errs = np.full(n, 0.05)
    return times, mags, errs, period
