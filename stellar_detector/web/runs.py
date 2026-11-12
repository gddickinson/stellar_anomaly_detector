"""Runs served by the web app: the committed demo field (analysed once, cached) and recent local runs (small LRU).

A run is the dict from pipeline.run_pipeline. app.py serves slices of it: summary, stars, detections, candidates and
per-star detail. New runs fetch a catalogue with the repo's DataFetcher (network) or read an uploaded CSV.
"""
from __future__ import annotations

import io
import uuid
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..core.models import CatalogSource
from ..data.fetcher import DataFetcher
from .pipeline import make_config, run_pipeline

DEMO_DIR = Path(__file__).resolve().parent / "demo"
DEMO_CSV = DEMO_DIR / "gaia_dr3_kic8462852_field.csv"
BOYAJIAN_ID = "2081900940499099136"  # KIC 8462852 (Boyajian's star) in Gaia DR3
DEMO = {
    "id": "demo",
    "title": "Gaia DR3 field around KIC 8462852 (Boyajian's star)",
    "catalog": "gaia_dr3",
    "file": "stellar_detector/web/demo/" + DEMO_CSV.name,
    "ra": 301.5644, "dec": 44.4568, "radius_deg": 0.15,
    "query": "DataFetcher().fetch(CatalogSource.GAIA_DR3, n_stars=20000, ra_center=301.5644, dec_center=44.4568, "
             "radius_deg=0.15) -- the 'full async' ADQL query in data/fetcher.py: gaiadr3.gaia_source within the "
             "circle, parallax > 0 and phot_g_mean_mag not null (TOP 20000 was not reached, so the cone is complete)",
    "retrieved": "2026-11-11",
    "credit": "ESA/Gaia/DPAC, Gaia Data Release 3 (CC BY-SA 3.0 IGO)",
    "featured": {"id": BOYAJIAN_ID, "name": "KIC 8462852 (Boyajian's star)"},
}

SOURCES = {
    "gaia": CatalogSource.GAIA_DR3, "hipparcos": CatalogSource.HIPPARCOS, "tycho2": CatalogSource.TYCHO2,
    "2mass": CatalogSource.TWOMASS, "allwise": CatalogSource.ALLWISE, "synthetic": CatalogSource.SYNTHETIC,
}
MAX_STARS = 50000
MAX_RUNS = 6
RUNS: "OrderedDict[str, dict]" = OrderedDict()


class RunError(ValueError):
    pass


def demo_raw() -> pd.DataFrame:
    return pd.read_csv(DEMO_CSV)


@lru_cache(maxsize=1)
def demo_run() -> dict:
    return run_pipeline(demo_raw(), make_config(), {**DEMO, "n_rows": None})


def get_run(run_id: str) -> dict | None:
    return demo_run() if run_id == "demo" else RUNS.get(run_id)


def _store(run: dict) -> str:
    run_id = uuid.uuid4().hex[:10]
    run["source"]["id"] = run_id
    RUNS[run_id] = run
    while len(RUNS) > MAX_RUNS:
        RUNS.popitem(last=False)
    return run_id


def _num(params: dict, key: str, default: float, lo: float, hi: float) -> float:
    v = params.get(key, default)
    try:
        v = float(default if v in (None, "") else v)
    except (TypeError, ValueError):
        raise RunError(f"{key} must be a number")
    if not lo <= v <= hi:
        raise RunError(f"{key} must be between {lo} and {hi}")
    return v


def new_run(params: dict, fetcher: DataFetcher | None = None) -> str:
    """Fetch a catalogue (or reuse the demo CSV) and analyse it. params: source, n_stars, ra, dec, radius, config."""
    try:
        config = make_config(params.get("config") or {})
    except ValueError as e:
        raise RunError(str(e))
    source = params.get("source", "demo")
    if source == "demo":
        raw, meta = demo_raw(), {**DEMO, "title": DEMO["title"] + " (re-run with your settings)"}
    elif source in SOURCES:
        n = int(_num(params, "n_stars", 2000, 10, MAX_STARS))
        ra = _num(params, "ra", 180.0, 0.0, 360.0)
        dec = _num(params, "dec", 0.0, -90.0, 90.0)
        rad = _num(params, "radius", 1.0, 0.001, 10.0)
        fetcher = fetcher or DataFetcher()
        try:
            raw = fetcher.fetch(SOURCES[source], n_stars=n, ra_center=ra, dec_center=dec, radius_deg=rad,
                                use_cache=bool(params.get("use_cache", True)))
        except Exception as e:
            raise RunError(f"Fetching {source} failed: {e}")
        meta = {"title": f"{source} · {n} stars · RA {ra} Dec {dec} r={rad} deg", "catalog": SOURCES[source].value,
                "ra": ra, "dec": dec, "radius_deg": rad}
    else:
        raise RunError(f"Unknown source {source!r}")
    return _store(run_pipeline(raw, config, meta))


def upload_run(data: bytes, name: str, config_overrides: dict) -> str:
    """Analyse an uploaded catalogue CSV (as `cli --file`); columns follow the Gaia DR3 names used by the fetcher."""
    try:
        config = make_config(config_overrides)
        raw = pd.read_csv(io.BytesIO(data))
    except ValueError as e:
        raise RunError(str(e))
    except Exception as e:
        raise RunError(f"Could not read the CSV: {e}")
    if len(raw) > MAX_STARS:
        raise RunError(f"Too many rows ({len(raw)}; limit {MAX_STARS}); use the command line.")
    if not {"bp_rp", "parallax", "phot_g_mean_mag"} & set(raw.columns):
        raise RunError("The CSV needs Gaia-style columns (e.g. source_id, ra, dec, parallax, phot_g_mean_mag, bp_rp).")
    return _store(run_pipeline(raw, config, {"title": f"Uploaded file {name}", "catalog": "upload", "file": name}))


def summary(run: dict) -> dict:
    keys = ("source", "config", "n_input", "n_stars", "modules", "stats", "tiers", "hr", "seconds")
    out = {k: run[k] for k in keys}
    out["n_detections"] = len(run["detections"])
    out["n_candidates"] = len(run["candidates"])
    out["n_flagged_stars"] = len({d["star_id"] for d in run["detections"]})
    out["columns"] = list(run["stars"])
    return out


def star_detail(run: dict, star_id: str) -> dict | None:
    ids = run["stars"].get("source_id", [])
    try:
        i = ids.index(star_id)
    except ValueError:
        return None
    props = {c: v[i] for c, v in run["stars"].items()}
    cand = next((c for c in run["candidates"] if c["star_id"] == star_id), None)
    return {
        "star_id": star_id, "run": run["source"].get("id"), "properties": props,
        "detections": [d for d in run["detections"] if d["star_id"] == star_id],
        "candidate": cand, "n_candidates": len(run["candidates"]), "n_stars": run["n_stars"],
    }
