"""Tests for the Dyson-sphere SED fit (Project Hephaistos-style model).

The test photometry is built independently of the analyzer: a star (and optionally a warm shell that re-radiates a
fraction f of the star's light) as blackbodies in F_nu, converted to magnitudes with the analyzer's zero points.
"""

import numpy as np
import pandas as pd
import pytest

from stellar_detector.analysis.technosignature import (
    BAND_WAVELENGTHS, ZERO_POINTS_JY, TechnosignatureAnalyzer,
)

H, C, K = 6.626e-34, 2.998e8, 1.381e-23
BANDS = ["phot_g_mean_mag", "Jmag", "Hmag", "Kmag", "W1mag", "W2mag", "W3mag", "W4mag"]


def _bnu(lam_um, t):
    nu = C / (np.asarray(lam_um) * 1e-6)
    return 2 * H * nu ** 3 / C ** 2 / np.expm1(H * nu / (K * t))


def _star_row(teff, f_cover=0.0, t_ds=300.0, g_mag=12.0, name="S"):
    """Magnitudes of a star whose shell absorbs a fraction f_cover of its light and re-emits it at t_ds."""
    lam = np.array([BAND_WAVELENGTHS[b] for b in BANDS])
    fnu = (1 - f_cover) * _bnu(lam, teff) / teff ** 4 + f_cover * _bnu(lam, t_ds) / t_ds ** 4
    fnu = fnu / fnu[0] * ZERO_POINTS_JY["phot_g_mean_mag"] * 10 ** (-g_mag / 2.5)
    row = {b: -2.5 * np.log10(fnu[i] / ZERO_POINTS_JY[b]) for i, b in enumerate(BANDS)}
    row.update(source_id=name, teff_gspphot=teff, e_W3mag=0.05, e_W4mag=0.05, catalog_source="test")
    return row


@pytest.mark.parametrize("teff", [4000.0, 5800.0, 9000.0])
def test_bare_photosphere_is_not_a_candidate(teff):
    row = pd.Series(_star_row(teff))
    assert TechnosignatureAnalyzer()._fit_dyson_model(row) is None


@pytest.mark.parametrize("f_cover,t_ds", [(0.3, 300.0), (0.6, 400.0)])
def test_strong_warm_excess_is_found(f_cover, t_ds):
    row = pd.Series(_star_row(5800.0, f_cover, t_ds))
    res = TechnosignatureAnalyzer()._fit_dyson_model(row)
    assert res is not None
    assert res.parameters["dyson_temp_k"] == pytest.approx(t_ds, abs=50)
    assert res.parameters["covering_factor"] == pytest.approx(f_cover, abs=0.1)
    assert res.parameters["sed_rmse"] < 0.05


def test_star_only_model_fits_a_photosphere():
    """Observed fluxes are F_nu (Jy); the model must be in the same units, so a pure photosphere fits closely."""
    row = pd.Series(_star_row(5800.0))
    lam = np.array([BAND_WAVELENGTHS[b] for b in BANDS])
    obs = np.array([ZERO_POINTS_JY[b] * 10 ** (-row[b] / 2.5) for b in BANDS])
    model = TechnosignatureAnalyzer._combined_sed(lam, 5800.0, 300.0, 0.0)
    assert np.sqrt(np.mean((obs / obs.max() - model / model.max()) ** 2)) < 1e-6


def test_analyze_end_to_end_flags_only_the_excess_star():
    df = pd.DataFrame([_star_row(5800.0, name="plain"), _star_row(5800.0, 0.5, 350.0, name="shell")])
    found = [r for r in TechnosignatureAnalyzer().analyze(df) if r.detection_method == "hephaistos_sed_grid"]
    assert [r.star_id for r in found] == ["shell"]
