"""Data fetching from astronomical catalogs (Gaia, Hipparcos, Tycho-2, 2MASS, WISE).

Includes a local file cache so identical queries are served from disk
instead of re-downloading. Cache files are stored in ``<output_dir>/cache/``
as CSV files keyed by (source, n_stars, ra, dec, radius).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

if TYPE_CHECKING:
    from ..core.models import CatalogSource

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetch and standardize data from multiple astronomical catalogs.

    Fetched data is cached locally so repeat runs with the same parameters
    are served from disk. Use ``use_cache=False`` to force a fresh download.
    """

    def __init__(self, output_dir: str = "./stellar_data", cache_dir: str | None = None):
        self.output_dir = output_dir
        self.cache_dir = Path(cache_dir or output_dir) / "cache"

    def fetch(
        self,
        source: CatalogSource,
        n_stars: int = 2000,
        ra_center: float = 180.0,
        dec_center: float = 0.0,
        radius_deg: float = 5.0,
        random_sample: bool = True,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch data from the specified catalog source.

        Args:
            use_cache: If True (default), return cached data when available
                       and save new downloads to the cache.

        Cache behaviour:
        - Cache is keyed by **region** (source, ra, dec, radius) — not n_stars.
        - If the cache has >= n_stars rows for this region, a sample is returned.
        - If the cache has fewer rows than requested, a fresh download is done
          and the cache is updated with the larger result.
        """
        from ..core.models import CatalogSource

        # Try cache first (keyed by region, not by n_stars)
        region_key = self._region_key(source, ra_center, dec_center, radius_deg)
        if use_cache and source != CatalogSource.SYNTHETIC:
            cached = self._load_cache(region_key)
            if cached is not None and len(cached) >= n_stars:
                result = cached.sample(n=n_stars, random_state=42).reset_index(drop=True)
                logger.info(
                    "Cache hit for %s: returning %d / %d cached rows",
                    source.value, n_stars, len(cached),
                )
                return result
            elif cached is not None:
                logger.info(
                    "Cache has %d rows but %d requested — downloading fresh",
                    len(cached), n_stars,
                )

        dispatch = {
            CatalogSource.GAIA_DR3: self._fetch_gaia,
            CatalogSource.HIPPARCOS: self._fetch_hipparcos,
            CatalogSource.TYCHO2: self._fetch_tycho2,
            CatalogSource.TWOMASS: self._fetch_2mass,
            CatalogSource.ALLWISE: self._fetch_allwise,
            CatalogSource.SYNTHETIC: self._generate_synthetic,
        }
        fetcher = dispatch.get(source)
        if fetcher is None:
            raise ValueError(f"Unsupported catalog source: {source}")

        logger.info("Fetching %d stars from %s", n_stars, source.value)
        df = fetcher(
            n_stars=n_stars,
            ra_center=ra_center,
            dec_center=dec_center,
            radius_deg=radius_deg,
        )
        df["catalog_source"] = source.value
        logger.info("Fetched %d rows from %s", len(df), source.value)

        # Save to cache (only if this is the largest download for this region)
        if use_cache and source != CatalogSource.SYNTHETIC:
            existing = self._load_cache(region_key)
            if existing is None or len(df) > len(existing):
                self._save_cache(region_key, df)

        return df

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _region_key(source: CatalogSource, ra: float, dec: float, radius: float) -> str:
        """Cache key based on sky region only (not n_stars)."""
        raw = f"{source.value}_{ra:.4f}_{dec:.4f}_{radius:.4f}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.csv"

    def _load_cache(self, key: str) -> pd.DataFrame | None:
        path = self._cache_path(key)
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception as e:
                logger.warning("Cache read failed for %s: %s", key, e)
        return None

    def _save_cache(self, key: str, df: pd.DataFrame) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path(key).write_text(df.to_csv(index=False))
            logger.info("Cached %d rows -> %s", len(df), self._cache_path(key))
        except Exception as e:
            logger.warning("Cache write failed: %s", e)

    # ------------------------------------------------------------------
    # Gaia DR3
    # ------------------------------------------------------------------
    def _fetch_gaia(
        self, n_stars: int, ra_center: float, dec_center: float, radius_deg: float
    ) -> pd.DataFrame:
        from astroquery.gaia import Gaia

        # Strategy: try async full query -> async basic -> sync minimal
        queries = [
            (
                "full async",
                f"""SELECT TOP {n_stars}
                    source_id, ra, dec, parallax, parallax_error,
                    pmra, pmdec, phot_g_mean_mag, bp_rp,
                    teff_gspphot, logg_gspphot, mh_gspphot,
                    ruwe, astrometric_excess_noise, astrometric_excess_noise_sig,
                    phot_bp_rp_excess_factor, radial_velocity,
                    phot_g_mean_flux_over_error
                FROM gaiadr3.gaia_source
                WHERE 1=CONTAINS(
                    POINT('ICRS', ra, dec),
                    CIRCLE('ICRS', {ra_center}, {dec_center}, {radius_deg})
                )
                AND parallax IS NOT NULL
                AND parallax > 0
                AND phot_g_mean_mag IS NOT NULL""",
            ),
            (
                "basic async",
                f"""SELECT TOP {n_stars}
                    source_id, ra, dec, parallax, parallax_error,
                    pmra, pmdec, phot_g_mean_mag, bp_rp,
                    teff_gspphot, logg_gspphot, mh_gspphot, ruwe
                FROM gaiadr3.gaia_source
                WHERE 1=CONTAINS(
                    POINT('ICRS', ra, dec),
                    CIRCLE('ICRS', {ra_center}, {dec_center}, {radius_deg})
                )
                AND parallax IS NOT NULL AND parallax > 0""",
            ),
            (
                "minimal async",
                f"""SELECT TOP {n_stars}
                    source_id, ra, dec, parallax, parallax_error,
                    pmra, pmdec, phot_g_mean_mag, bp_rp
                FROM gaiadr3.gaia_source
                WHERE ra BETWEEN {ra_center - radius_deg} AND {ra_center + radius_deg}
                AND dec BETWEEN {dec_center - radius_deg} AND {dec_center + radius_deg}
                AND parallax IS NOT NULL AND parallax > 0""",
            ),
        ]

        for label, query in queries:
            try:
                logger.info("Gaia: trying %s query...", label)
                job = Gaia.launch_job_async(query)
                table = job.get_results()
                df = table.to_pandas()
                if len(df) > 0:
                    logger.info("Gaia %s query returned %d rows", label, len(df))
                    return df
            except Exception as e:
                logger.warning("Gaia %s query failed: %s", label, e)

        raise RuntimeError(
            "All Gaia queries failed. The archive may be down — "
            "try --source hipparcos or --source synthetic"
        )

    # ------------------------------------------------------------------
    # Hipparcos-2
    # ------------------------------------------------------------------
    def _fetch_hipparcos(
        self, n_stars: int, ra_center: float, dec_center: float, radius_deg: float
    ) -> pd.DataFrame:
        from astroquery.vizier import Vizier

        coord = SkyCoord(ra=ra_center, dec=dec_center, unit=(u.deg, u.deg))
        viz = Vizier(
            columns=["HIP", "RArad", "DErad", "Plx", "e_Plx", "pmRA", "pmDE", "Hpmag", "B-V"],
            row_limit=n_stars,
        )
        result = viz.query_region(coord, radius=radius_deg * u.deg, catalog="I/311/hip2")
        if not result:
            raise RuntimeError("Hipparcos query returned no results")

        df = result[0].to_pandas()
        return df.rename(columns={
            "HIP": "source_id",
            "RArad": "ra",
            "DErad": "dec",
            "Plx": "parallax",
            "e_Plx": "parallax_error",
            "pmRA": "pmra",
            "pmDE": "pmdec",
            "Hpmag": "phot_g_mean_mag",
            "B-V": "bp_rp",
        })

    # ------------------------------------------------------------------
    # Tycho-2
    # ------------------------------------------------------------------
    def _fetch_tycho2(
        self, n_stars: int, ra_center: float, dec_center: float, radius_deg: float
    ) -> pd.DataFrame:
        from astroquery.vizier import Vizier

        coord = SkyCoord(ra=ra_center, dec=dec_center, unit=(u.deg, u.deg))
        viz = Vizier(
            columns=["TYC1", "TYC2", "TYC3", "RAmdeg", "DEmdeg", "pmRA", "pmDE", "BTmag", "VTmag"],
            row_limit=n_stars,
        )
        result = viz.query_region(coord, radius=radius_deg * u.deg, catalog="I/259/tyc2")
        if not result:
            raise RuntimeError("Tycho-2 query returned no results")

        df = result[0].to_pandas()

        # Build proper TYC identifier from the three component columns
        id_parts = [df[c].astype(str) for c in ["TYC1", "TYC2", "TYC3"] if c in df.columns]
        if id_parts:
            df["source_id"] = "TYC_" + id_parts[0].str.cat(id_parts[1:], sep="-")
        else:
            df["source_id"] = [f"TYC_{i}" for i in range(len(df))]

        df["bp_rp"] = df.get("BTmag", np.nan) - df.get("VTmag", np.nan)
        df = df.rename(columns={
            "RAmdeg": "ra",
            "DEmdeg": "dec",
            "pmRA": "pmra",
            "pmDE": "pmdec",
            "VTmag": "phot_g_mean_mag",
        })
        return df

    # ------------------------------------------------------------------
    # 2MASS
    # ------------------------------------------------------------------
    def _fetch_2mass(
        self, n_stars: int, ra_center: float, dec_center: float, radius_deg: float
    ) -> pd.DataFrame:
        from astroquery.vizier import Vizier

        coord = SkyCoord(ra=ra_center, dec=dec_center, unit=(u.deg, u.deg))
        viz = Vizier(
            columns=["_2MASS", "RAJ2000", "DEJ2000", "Jmag", "Hmag", "Kmag",
                      "e_Jmag", "e_Hmag", "e_Kmag"],
            row_limit=n_stars,
        )
        result = viz.query_region(coord, radius=radius_deg * u.deg, catalog="II/246/out")
        if not result:
            raise RuntimeError("2MASS query returned no results")

        df = result[0].to_pandas()
        if "_2MASS" in df.columns:
            df = df.rename(columns={"_2MASS": "source_id"})
        else:
            df["source_id"] = [f"2M_{i}" for i in range(len(df))]
        return df.rename(columns={"RAJ2000": "ra", "DEJ2000": "dec"})

    # ------------------------------------------------------------------
    # AllWISE
    # ------------------------------------------------------------------
    def _fetch_allwise(
        self, n_stars: int, ra_center: float, dec_center: float, radius_deg: float
    ) -> pd.DataFrame:
        from astroquery.vizier import Vizier

        coord = SkyCoord(ra=ra_center, dec=dec_center, unit=(u.deg, u.deg))
        viz = Vizier(
            columns=[
                "AllWISE", "RAJ2000", "DEJ2000",
                "W1mag", "W2mag", "W3mag", "W4mag",
                "e_W1mag", "e_W2mag", "e_W3mag", "e_W4mag",
            ],
            row_limit=n_stars,
        )
        result = viz.query_region(coord, radius=radius_deg * u.deg, catalog="II/328/allwise")
        if not result:
            raise RuntimeError("AllWISE query returned no results")

        df = result[0].to_pandas()
        if "AllWISE" in df.columns:
            df = df.rename(columns={"AllWISE": "source_id"})
        else:
            df["source_id"] = [f"WISE_{i}" for i in range(len(df))]
        return df.rename(columns={"RAJ2000": "ra", "DEJ2000": "dec"})

    # ------------------------------------------------------------------
    # Synthetic (for testing)
    # ------------------------------------------------------------------
    def _generate_synthetic(
        self, n_stars: int, ra_center: float, dec_center: float, radius_deg: float
    ) -> pd.DataFrame:
        rng = np.random.default_rng(42)

        temperatures = 10 ** rng.uniform(3.4, 4.7, n_stars)
        main_seq_mag = _synthetic_main_sequence_mag(temperatures)
        scatter = rng.normal(0, 0.3, n_stars)

        df = pd.DataFrame({
            "source_id": [f"SYN_{i:06d}" for i in range(n_stars)],
            "ra": ra_center + rng.uniform(-radius_deg, radius_deg, n_stars),
            "dec": dec_center + rng.uniform(-radius_deg, radius_deg, n_stars),
            "parallax": rng.uniform(1.0, 50.0, n_stars),
            "parallax_error": rng.uniform(0.01, 0.5, n_stars),
            "pmra": rng.normal(0, 10, n_stars),
            "pmdec": rng.normal(0, 10, n_stars),
            "phot_g_mean_mag": rng.uniform(3, 18, n_stars),
            "bp_rp": rng.uniform(-0.5, 4.0, n_stars),
            "teff_gspphot": temperatures,
            "logg_gspphot": rng.uniform(1.0, 5.0, n_stars),
            "mh_gspphot": rng.normal(-0.1, 0.4, n_stars),
            "ruwe": rng.lognormal(0.0, 0.3, n_stars),
            "abs_mag": main_seq_mag + scatter,
        })

        # Inject known anomalies (5% of sample)
        n_anomalies = max(1, n_stars // 20)
        anomaly_idx = rng.choice(n_stars, n_anomalies, replace=False)
        df.loc[anomaly_idx, "abs_mag"] += rng.uniform(2, 6, n_anomalies)
        df.loc[anomaly_idx, "pmra"] += rng.uniform(50, 200, n_anomalies)
        df["is_injected_anomaly"] = False
        df.loc[anomaly_idx, "is_injected_anomaly"] = True

        return df


def _synthetic_main_sequence_mag(temperatures: np.ndarray) -> np.ndarray:
    """Approximate absolute magnitude for main-sequence stars given temperature."""
    log_t = np.log10(temperatures)
    return -10.0 * log_t + 40.0
