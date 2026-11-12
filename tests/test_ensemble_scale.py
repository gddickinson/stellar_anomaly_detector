"""The ensemble's single-method (Tier 3) cut must compare scores on one scale (sigma), not raw module scores."""

import numpy as np
import pandas as pd

from stellar_detector.analysis.ensemble import EnsembleScorer, common_significance
from stellar_detector.analysis.ml_pipeline import MLPipeline
from stellar_detector.analysis.stellar_lifetime import StellarLifetimeAnalyzer
from stellar_detector.core.models import AnomalyResult, AnomalyType, DetectionConfig


def _r(method, sig, atype=AnomalyType.KINEMATICS, tests=None):
    return AnomalyResult(star_id="S", anomaly_type=atype, confidence=0.6, significance_score=sig, parameters={},
                         description="t", follow_up_priority=5, detection_method=method,
                         statistical_tests=tests or {})


def test_lifetime_and_ml_scores_do_not_pass_automatically():
    # 10 x age fraction and 10 x ML score are not sigmas: alone they must not reach Tier 3
    assert EnsembleScorer().aggregate([_r("lifetime_isochrone", 15.0, AnomalyType.LIFETIME)]) == []
    assert EnsembleScorer().aggregate([_r("ml_ensemble", 9.5, AnomalyType.PHOTOMETRY)]) == []
    assert EnsembleScorer().aggregate([_r("ruwe_threshold", 9.0, AnomalyType.ASTROMETRIC)]) == []


def test_z_scores_and_declared_sigma_still_pass():
    assert len(EnsembleScorer().aggregate([_r("proper_motion_mad", 7.0)])) == 1
    assert EnsembleScorer().aggregate([_r("proper_motion_mad", 5.0)]) == []
    assert len(EnsembleScorer().aggregate([_r("lifetime_isochrone", 15.0, AnomalyType.LIFETIME,
                                              {"sigma_equiv": 6.5})])) == 1


def test_common_significance_of_analyzer_output_is_a_rank_sigma():
    """Lifetime and ML detections carry a rank-percentile sigma, bounded by the sample size."""
    rng = np.random.default_rng(0)
    n = 400
    t = 10 ** rng.uniform(3.6, 3.9, n)
    df = pd.DataFrame({"source_id": [f"S{i}" for i in range(n)], "teff_gspphot": t,
                       "abs_mag": -8 * np.log10(t) + 34.5 - rng.uniform(0, 1.3, n),
                       "bp_rp": rng.normal(1, 0.3, n), "pmra": rng.normal(0, 5, n), "pmdec": rng.normal(0, 5, n)})
    cfg = DetectionConfig()
    res = StellarLifetimeAnalyzer(cfg).analyze(df) + MLPipeline(cfg).analyze(df)
    assert res
    for r in res:
        s = common_significance(r)
        assert 0 < s < 3.5  # max for n=400 is isf(0.5/400) ~ 3.0
