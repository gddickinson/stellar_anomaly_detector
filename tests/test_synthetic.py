"""The synthetic catalogue must be a clean population for the package's own models, plus the injected anomalies."""

import pytest

from stellar_detector.analysis.ensemble import EnsembleScorer
from stellar_detector.analysis.stellar_lifetime import StellarLifetimeAnalyzer
from stellar_detector.core.models import CatalogSource, DetectionConfig
from stellar_detector.data.fetcher import DataFetcher
from stellar_detector.data.preprocessing import preprocess_catalog
from stellar_detector.web.pipeline import MODULES


@pytest.fixture(scope="module")
def synth():
    cfg = DetectionConfig()
    return cfg, preprocess_catalog(DataFetcher().fetch(CatalogSource.SYNTHETIC, n_stars=2000), cfg)


def _clean_rate(df, ids):
    clean = df[~df["is_injected_anomaly"].astype(bool)]
    return clean["source_id"].isin(ids).mean()


def test_lifetime_false_positive_rate_is_low(synth):
    cfg, df = synth
    ids = {r.star_id for r in StellarLifetimeAnalyzer(cfg).analyze(df)}
    assert _clean_rate(df, ids) < 0.05


def test_ensemble_false_positive_rate_is_low(synth):
    cfg, df = synth
    raw = [r for _, _, cls in MODULES for r in cls(cfg).analyze(df)]
    ids = {r.star_id for r in EnsembleScorer().aggregate(raw)}
    assert _clean_rate(df, ids) < 0.05


def test_injected_anomalies_are_mostly_recovered(synth):
    cfg, df = synth
    raw = [r for _, _, cls in MODULES for r in cls(cfg).analyze(df)]
    ids = {r.star_id for r in EnsembleScorer().aggregate(raw)}
    injected = df[df["is_injected_anomaly"].astype(bool)]
    assert injected["source_id"].isin(ids).mean() > 0.8
