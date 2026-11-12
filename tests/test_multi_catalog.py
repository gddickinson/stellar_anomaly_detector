"""--multi must cross-match catalogues by position onto the first one, not stack their rows."""

import numpy as np
import pandas as pd

from stellar_detector.cli import build_parser, config_from_args, load_multi
from stellar_detector.core.models import CatalogSource
from stellar_detector.data.cross_match import merge_catalogs

ARCSEC = 1 / 3600


def _gaia():
    return pd.DataFrame({"source_id": ["G1", "G2", "G3", "G4"], "ra": [10.0, 10.01, 10.02, 10.03],
                         "dec": [5.0, 5.0, 5.0, 5.0], "parallax": [5.0, 4.0, 3.0, 2.0], "pmra": 0.0, "pmdec": 0.0,
                         "phot_g_mean_mag": [10.0, 11, 12, 13], "catalog_source": "gaia_dr3"})


def _wise():
    return pd.DataFrame({
        # W1 matches G1 at 1"; W2 is 8" from G2 (outside 5"); W3 sits 1" from G3 and 2" from G4's neighbour
        "source_id": ["W1", "W2", "W3"], "ra": [10.0 + ARCSEC, 10.01 + 8 * ARCSEC, 10.02 + ARCSEC],
        "dec": [5.0, 5.0, 5.0], "W3mag": [8.0, 9.0, 7.0], "W4mag": [7.5, 8.5, 6.5], "catalog_source": "allwise"})


def test_merge_adds_photometry_to_matched_rows_only():
    out = merge_catalogs(_gaia(), {"allwise": _wise()}, radius_arcsec=5.0, epochs={"gaia_dr3": 2016.0, "allwise": 2010.5},
                         primary_name="gaia_dr3")
    assert list(out["source_id"]) == ["G1", "G2", "G3", "G4"]          # one row per primary star
    assert out.loc[0, "W3mag"] == 8.0 and out.loc[2, "W3mag"] == 7.0
    assert np.isnan(out.loc[1, "W3mag"]) and np.isnan(out.loc[3, "W3mag"])  # 8" is beyond the radius
    assert out.loc[0, "allwise_id"] == "W1" and out.loc[0, "allwise_sep_arcsec"] < 1.5


def test_secondary_source_is_assigned_to_one_primary_star_only():
    g = _gaia()
    g.loc[1, "ra"] = 10.0 + 2 * ARCSEC                                  # G2 also within 5" of W1, but farther
    out = merge_catalogs(g, {"allwise": _wise()}, radius_arcsec=5.0, epochs={}, primary_name="gaia_dr3")
    assert out.loc[0, "allwise_id"] == "W1" and pd.isna(out.loc[1, "allwise_id"])


def test_overlapping_columns_keep_primary_values():
    w = _wise().assign(parallax=99.0)
    out = merge_catalogs(_gaia(), {"allwise": w}, radius_arcsec=5.0, epochs={}, primary_name="gaia_dr3")
    assert out.loc[0, "parallax"] == 5.0 and out.loc[0, "parallax_allwise"] == 99.0


class FakeFetcher:
    def fetch(self, source, **k):
        return {CatalogSource.GAIA_DR3: _gaia(), CatalogSource.ALLWISE: _wise()}[source]


def test_cli_multi_cross_matches():
    args = build_parser().parse_args(["--multi", "gaia,allwise", "--match-radius", "5"])
    df = load_multi(FakeFetcher(), args, config_from_args(args))
    assert len(df) == 4 and df["W3mag"].notna().sum() == 2
