"""Cross-catalog matching using sky coordinates and epoch propagation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

logger = logging.getLogger(__name__)


class CrossCatalogMatcher:
    """Match stars across catalogs using positional and proper-motion-aware matching.

    Implements the Gaia-recommended approach: epoch propagation + angular separation
    with a Figure of Merit (FoM) for best-neighbour selection.
    """

    def __init__(self, max_separation_arcsec: float = 5.0, confidence_sigma: float = 5.0):
        self.max_sep = max_separation_arcsec
        self.confidence_sigma = confidence_sigma

    def match(
        self,
        catalog_a: pd.DataFrame,
        catalog_b: pd.DataFrame,
        epoch_a: float = 2016.0,
        epoch_b: float = 2016.0,
        common_epoch: float = 2016.0,
    ) -> pd.DataFrame:
        """Cross-match two catalogs. Returns a DataFrame of matched pairs.

        Each row contains columns from both catalogs with suffixes ``_a`` and ``_b``,
        plus ``separation_arcsec`` and ``match_fom`` (Figure of Merit).
        """
        coords_a = self._build_skycoord(catalog_a, epoch_a, common_epoch)
        coords_b = self._build_skycoord(catalog_b, epoch_b, common_epoch)

        if coords_a is None or coords_b is None:
            logger.warning("Could not build SkyCoord for one or both catalogs")
            return pd.DataFrame()

        idx, sep2d, _ = coords_a.match_to_catalog_sky(coords_b)
        sep_arcsec = sep2d.arcsec

        mask = sep_arcsec <= self.max_sep
        logger.info(
            "Cross-match: %d / %d sources within %.1f arcsec",
            mask.sum(), len(catalog_a), self.max_sep,
        )

        a_idx = np.where(mask)[0]
        b_idx = idx[mask]

        df_a = catalog_a.iloc[a_idx].reset_index(drop=True).add_suffix("_a")
        df_b = catalog_b.iloc[b_idx].reset_index(drop=True).add_suffix("_b")

        matched = pd.concat([df_a, df_b], axis=1)
        matched["separation_arcsec"] = sep_arcsec[mask]
        matched["match_fom"] = self._figure_of_merit(matched, sep_arcsec[mask])

        return matched.sort_values("match_fom", ascending=False).reset_index(drop=True)

    def match_multiple(self, catalogs: dict[str, pd.DataFrame], epochs: dict[str, float]) -> dict:
        """Pairwise cross-match of multiple catalogs.

        Returns a dict keyed by ``"name_a__name_b"`` -> matched DataFrame.
        """
        names = list(catalogs.keys())
        results = {}
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                key = f"{name_a}__{name_b}"
                logger.info("Cross-matching %s x %s", name_a, name_b)
                results[key] = self.match(
                    catalogs[name_a],
                    catalogs[name_b],
                    epoch_a=epochs.get(name_a, 2016.0),
                    epoch_b=epochs.get(name_b, 2016.0),
                )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_skycoord(
        self, df: pd.DataFrame, catalog_epoch: float, target_epoch: float
    ) -> SkyCoord | None:
        """Build SkyCoord, optionally propagating proper motions to a common epoch."""
        if "ra" not in df.columns or "dec" not in df.columns:
            return None

        ra = df["ra"].values
        dec = df["dec"].values

        has_pm = "pmra" in df.columns and "pmdec" in df.columns
        if has_pm:
            pmra = df["pmra"].fillna(0).values * u.mas / u.yr
            pmdec = df["pmdec"].fillna(0).values * u.mas / u.yr
        else:
            pmra = np.zeros(len(df)) * u.mas / u.yr
            pmdec = np.zeros(len(df)) * u.mas / u.yr

        coord = SkyCoord(
            ra=ra * u.deg,
            dec=dec * u.deg,
            pm_ra_cosdec=pmra,
            pm_dec=pmdec,
            obstime=Time(catalog_epoch, format="jyear"),
        )

        if abs(catalog_epoch - target_epoch) > 0.01 and has_pm:
            coord = coord.apply_space_motion(new_obstime=Time(target_epoch, format="jyear"))

        return coord

    @staticmethod
    def _figure_of_merit(matched: pd.DataFrame, separation: np.ndarray) -> np.ndarray:
        """Compute a Figure of Merit combining separation and positional errors.

        Higher FoM = better match. Penalizes large separations and rewards
        small positional uncertainties.
        """
        # Start with inverse separation (closer = better)
        fom = 1.0 / (separation + 0.01)

        # Boost FoM if parallax errors are small
        for suffix in ("_a", "_b"):
            col = f"parallax_error{suffix}"
            if col in matched.columns:
                plx_err = matched[col].fillna(1.0).values
                fom *= 1.0 / (plx_err + 0.01)

        return fom
