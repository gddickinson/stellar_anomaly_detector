"""Run the repo's own analysis pipeline (as cli.main does) and turn the results into JSON-ready dicts.

Nothing here re-implements detection: preprocessing, the six analyzers and EnsembleScorer are the package's own code.
The only additions are bookkeeping for the web pages: which module produced each raw detection, the ensemble tier of
each candidate (EnsembleScorer._evaluate_star, the same call aggregate() makes), and the HR main-sequence band.
"""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, fields

import numpy as np
import pandas as pd

from ..analysis.robust import robust_center_scale
from ..analysis.ensemble import (
    HIGH_IMPORTANCE_TYPES, SINGLE_METHOD_MIN_SIGNIFICANCE, TIER1_MIN_TYPE_FAMILIES, TIER2_MIN_METHODS,
    TYPE_FAMILIES, EnsembleScorer,
)
from ..analysis.hr_diagram import HRDiagramAnalyzer
from ..analysis.kinematics import KinematicsAnalyzer
from ..analysis.ml_pipeline import MLPipeline
from ..analysis.spectral import SpectralAnalyzer
from ..analysis.stellar_lifetime import StellarLifetimeAnalyzer
from ..analysis.technosignature import TechnosignatureAnalyzer
from ..core.models import AnomalyResult, AnomalyType, DetectionConfig
from ..data.preprocessing import preprocess_catalog
from ..models.stellar_evolution import StellarEvolutionModels

# The analyzers cli.main runs, in its order: (id, label, class).
MODULES = [
    ("hr", "HR diagram", HRDiagramAnalyzer),
    ("lifetime", "Stellar lifetime", StellarLifetimeAnalyzer),
    ("kinematics", "Kinematics", KinematicsAnalyzer),
    ("spectral", "Spectral", SpectralAnalyzer),
    ("technosignature", "Technosignature", TechnosignatureAnalyzer),
    ("ml", "ML pipeline", MLPipeline),
]

# Per-star columns sent to the browser (those present in the data).
STAR_COLUMNS = [
    "source_id", "ra", "dec", "parallax", "parallax_error", "distance_pc", "pmra", "pmdec", "pm_total",
    "v_tan_km_s", "phot_g_mean_mag", "bp_rp", "abs_mag", "luminosity_solar", "teff_gspphot", "logg_gspphot",
    "mh_gspphot", "ruwe", "astrometric_excess_noise", "astrometric_excess_noise_sig", "phot_bp_rp_excess_factor",
    "radial_velocity", "quality_score", "Jmag", "Hmag", "Kmag", "W1mag", "W2mag", "W3mag", "W4mag",
    "J_H", "H_K", "W1_W2", "W2_W3", "is_injected_anomaly",
]

# DetectionConfig fields the web form can change: (name, label, min, max, step, help).
CONFIG_FIELDS = [
    ("mad_threshold", "Robust z-score threshold (sigma = 1.4826 MAD)", 2.0, 10.0, 0.5,
     "Used by the HR main-sequence, proper-motion, velocity, metallicity and WISE-colour detectors."),
    ("min_parallax_over_error", "Minimum parallax / error", 1.0, 50.0, 1.0, "Quality cut before any analysis."),
    ("max_ruwe", "Maximum RUWE", 1.0, 5.0, 0.1, "Quality cut; also the RUWE-anomaly threshold."),
    ("isolation_contamination", "Isolation Forest contamination", 0.001, 0.2, 0.001,
     "Expected outlier fraction (ML pipeline and chemical Isolation Forest)."),
    ("lof_contamination", "LOF contamination", 0.001, 0.2, 0.001, "Expected outlier fraction for LOF."),
    ("ensemble_threshold", "ML ensemble score threshold", 0.3, 0.99, 0.01,
     "Weighted mean of min-max-normalised ML scores above which a star is flagged."),
    ("gmm_n_components", "GMM components (HR)", 2, 20, 1, "Gaussian mixture components fitted to the CMD."),
    ("dbscan_min_samples", "DBSCAN min samples (HR)", 3, 30, 1, "Neighbourhood size for DBSCAN noise points."),
]


def clean(v):
    """JSON-safe scalar/list/dict: numpy -> python, NaN/inf -> None, enums -> names."""
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, np.ndarray)):
        return [clean(x) for x in v]
    if isinstance(v, AnomalyType):
        return v.anomaly_name
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    return v


def make_config(overrides: dict | None = None) -> DetectionConfig:
    """DetectionConfig with validated overrides; mad_threshold also sets outlier_threshold (as the CLI does)."""
    config = DetectionConfig()
    for name, _label, lo, hi, _step, _help in CONFIG_FIELDS:
        if overrides and overrides.get(name) not in (None, ""):
            v = float(overrides[name])
            if not (lo <= v <= hi):
                raise ValueError(f"{name} must be between {lo} and {hi}")
            setattr(config, name, int(v) if isinstance(getattr(config, name), int) else v)
    config.outlier_threshold = config.mad_threshold
    return config


def result_dict(r: AnomalyResult, module: str | None = None) -> dict:
    d = {
        "star_id": r.star_id, "anomaly_type": r.anomaly_type.anomaly_name,
        "family": TYPE_FAMILIES.get(r.anomaly_type, r.anomaly_type.anomaly_name),
        "confidence": r.confidence, "significance_score": r.significance_score,
        "follow_up_priority": r.follow_up_priority, "detection_method": r.detection_method,
        "description": r.description, "parameters": r.parameters, "statistical_tests": r.statistical_tests,
        "observational_recommendations": r.observational_recommendations,
    }
    if module:
        d["module"] = module
    return clean(d)


def ensemble_with_tiers(raw: list[AnomalyResult]) -> list[tuple[int, AnomalyResult]]:
    """EnsembleScorer.aggregate, keeping each selected star's tier (1-3). Same order as aggregate()."""
    scorer = EnsembleScorer()
    by_star: dict[str, list[AnomalyResult]] = defaultdict(list)
    for r in raw:
        by_star[r.star_id].append(r)
    out = [scorer._evaluate_star(sid, dets) for sid, dets in by_star.items()]
    out = [(t, r) for t, r in out if r is not None]
    out.sort(key=lambda tr: (tr[1].follow_up_priority, tr[1].significance_score), reverse=True)
    return out


def hr_band(df: pd.DataFrame, config: DetectionConfig) -> dict:
    """Main-sequence model curve and the +/- threshold band of HRDiagramAnalyzer._main_sequence_deviation."""
    models = StellarEvolutionModels()
    grid = np.round(np.linspace(-0.5, 4.5, 101), 3)
    curve = [models.main_sequence_mag_from_color(x) for x in grid]
    d = df.dropna(subset=["bp_rp", "abs_mag"]) if {"bp_rp", "abs_mag"} <= set(df.columns) else df.iloc[:0]
    if len(d) < 3:
        return {"bp_rp": grid.tolist(), "ms_mag": curve, "center": None, "scale": None,
                "threshold": config.mad_threshold}
    dev = d["abs_mag"].values - d["bp_rp"].apply(models.main_sequence_mag_from_color).values
    center, scale = robust_center_scale(dev)
    return clean({"bp_rp": grid.tolist(), "ms_mag": curve, "center": center, "scale": scale,
                  "threshold": config.mad_threshold})


def run_pipeline(raw: pd.DataFrame, config: DetectionConfig, source: dict) -> dict:
    """Preprocess, run every module, aggregate. Returns the full run record (see app.py for how it is served)."""
    t0 = time.time()
    df = preprocess_catalog(raw, config).reset_index(drop=True)
    detections, modules = [], []
    for mid, label, cls in MODULES:
        try:
            res = cls(config).analyze(df)
            modules.append({"id": mid, "label": label, "n": len(res), "error": None,
                            "methods": dict(Counter(r.detection_method for r in res))})
            detections += [(mid, r) for r in res]
        except Exception as e:  # cli.main logs and continues the same way
            modules.append({"id": mid, "label": label, "n": 0, "error": str(e), "methods": {}})
    tiers = ensemble_with_tiers([r for _, r in detections])
    stats = EnsembleScorer().summary_stats([r for _, r in tiers])
    # Stars that failed only the RUWE cut: not analysed, but the RUWE test may flag them, so list them too.
    rejects = df.attrs.get("ruwe_rejects")
    rejects = rejects if isinstance(rejects, pd.DataFrame) else df.iloc[:0]
    table = pd.concat([df.assign(analysed=True), rejects.assign(analysed=False)], ignore_index=True)
    cols = [c for c in STAR_COLUMNS if c in table.columns] + ["analysed"]
    stars = {c: clean(table[c].astype(str).tolist() if c == "source_id" else table[c].tolist()) for c in cols}
    cands = []
    for rank, (tier, r) in enumerate(tiers, 1):
        d = result_dict(r)
        d.update(rank=rank, tier=tier)
        cands.append(d)
    return {
        "source": source,
        "config": clean(asdict(config)),
        "n_input": int(len(raw)), "n_stars": int(len(df)), "n_ruwe_set_aside": int(len(rejects)),
        "modules": modules,
        "detections": [result_dict(r, mid) for mid, r in detections],
        "candidates": cands,
        "stats": clean(stats),
        "tiers": dict(Counter(str(c["tier"]) for c in cands)),
        "hr": hr_band(df, config),
        "stars": stars,
        "seconds": round(time.time() - t0, 2),
    }


def ensemble_rules() -> dict:
    """The EnsembleScorer constants, for the in-browser explorer."""
    return {
        "tier1_min_families": TIER1_MIN_TYPE_FAMILIES, "tier2_min_methods": TIER2_MIN_METHODS,
        "single_min_significance": SINGLE_METHOD_MIN_SIGNIFICANCE,
        "high_importance_types": sorted(t.anomaly_name for t in HIGH_IMPORTANCE_TYPES),
        "high_importance_min_confidence": 0.7,
        "families": {t.anomaly_name: f for t, f in TYPE_FAMILIES.items()},
    }


def config_fields() -> list[dict]:
    defaults = DetectionConfig()
    names = {f.name for f in fields(DetectionConfig)}
    return [{"name": n, "label": lab, "min": lo, "max": hi, "step": st, "help": h, "default": getattr(defaults, n)}
            for n, lab, lo, hi, st, h in CONFIG_FIELDS if n in names]
