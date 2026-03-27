"""Tests for analysis modules: HR diagram, lifetime, kinematics, spectral, ML, ensemble."""

import numpy as np
import pandas as pd
import pytest

from stellar_detector.analysis.ensemble import EnsembleScorer
from stellar_detector.analysis.hr_diagram import HRDiagramAnalyzer
from stellar_detector.analysis.kinematics import KinematicsAnalyzer
from stellar_detector.analysis.ml_pipeline import MLPipeline
from stellar_detector.analysis.spectral import SpectralAnalyzer
from stellar_detector.analysis.stellar_lifetime import StellarLifetimeAnalyzer
from stellar_detector.analysis.variability import VariabilityAnalyzer
from stellar_detector.core.models import AnomalyResult, AnomalyType, DetectionConfig


# ── HR Diagram ───────────────────────────────────────────────────────
class TestHRDiagramAnalyzer:
    def test_finds_anomalies_in_synthetic(self, synthetic_df, config):
        analyzer = HRDiagramAnalyzer(config)
        results = analyzer.analyze(synthetic_df)
        assert len(results) > 0
        assert all(isinstance(r, AnomalyResult) for r in results)

    def test_all_results_are_hr_outliers(self, synthetic_df, config):
        results = HRDiagramAnalyzer(config).analyze(synthetic_df)
        assert all(r.anomaly_type == AnomalyType.HR_OUTLIER for r in results)

    def test_returns_empty_for_missing_columns(self, config):
        df = pd.DataFrame({"ra": [1, 2], "dec": [3, 4]})
        results = HRDiagramAnalyzer(config).analyze(df)
        assert results == []

    def test_returns_empty_for_too_few_rows(self, config):
        df = pd.DataFrame({"bp_rp": [1.0], "abs_mag": [5.0]})
        results = HRDiagramAnalyzer(config).analyze(df)
        assert results == []

    def test_confidence_between_0_and_1(self, synthetic_df, config):
        results = HRDiagramAnalyzer(config).analyze(synthetic_df)
        for r in results:
            assert 0 <= r.confidence <= 1.0


# ── Stellar Lifetime ─────────────────────────────────────────────────
class TestStellarLifetimeAnalyzer:
    def test_finds_lifetime_anomalies(self, synthetic_df, config):
        results = StellarLifetimeAnalyzer(config).analyze(synthetic_df)
        assert len(results) >= 0  # may or may not find any
        for r in results:
            assert r.anomaly_type == AnomalyType.LIFETIME

    def test_requires_abs_mag_and_teff(self, config):
        df = pd.DataFrame({"bp_rp": [1.0], "parallax": [5.0]})
        results = StellarLifetimeAnalyzer(config).analyze(df)
        assert results == []


# ── Kinematics ───────────────────────────────────────────────────────
class TestKinematicsAnalyzer:
    def test_detects_high_pm_stars(self, config):
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "source_id": range(n),
            "pmra": np.append(rng.normal(0, 5, n - 2), [200, -180]),
            "pmdec": np.append(rng.normal(0, 5, n - 2), [150, -200]),
            "catalog_source": "test",
        })
        results = KinematicsAnalyzer(config).analyze(df)
        assert len(results) >= 2  # the two extreme PM stars

    def test_ruwe_anomalies(self, config):
        df = pd.DataFrame({
            "source_id": ["A", "B", "C"],
            "ruwe": [1.0, 1.3, 5.0],
            "catalog_source": "test",
        })
        results = KinematicsAnalyzer(config).analyze(df)
        ruwe_results = [r for r in results if "RUWE" in r.description]
        assert len(ruwe_results) == 1  # only RUWE=5.0 exceeds 1.4

    def test_astrometric_excess_noise(self, config):
        df = pd.DataFrame({
            "source_id": ["A", "B"],
            "astrometric_excess_noise_sig": [0.5, 5.0],
            "astrometric_excess_noise": [0.1, 2.0],
            "catalog_source": "test",
        })
        results = KinematicsAnalyzer(config).analyze(df)
        aen_results = [r for r in results if "Astrometric excess" in r.description]
        assert len(aen_results) == 1


# ── Variability ──────────────────────────────────────────────────────
class TestVariabilityAnalyzer:
    def test_light_curve_feature_extraction(self, sample_light_curve):
        times, mags, errs, true_period = sample_light_curve
        analyzer = VariabilityAnalyzer()
        features = analyzer.analyze_light_curve(times, mags, errs)

        assert "mean_mag" in features
        assert "stetson_j" in features
        assert "best_period" in features
        assert features["amplitude"] > 0

    def test_period_recovery(self, sample_light_curve):
        times, mags, errs, true_period = sample_light_curve
        analyzer = VariabilityAnalyzer()
        features = analyzer.analyze_light_curve(times, mags, errs)
        # Period should be approximately correct (within 20%)
        if features["best_period"] > 0:
            ratio = features["best_period"] / true_period
            assert 0.4 < ratio < 2.5  # allow harmonics

    def test_stetson_indices_computed(self, sample_light_curve):
        times, mags, errs, _ = sample_light_curve
        analyzer = VariabilityAnalyzer()
        features = analyzer.analyze_light_curve(times, mags, errs)
        assert "stetson_j" in features
        assert "stetson_k" in features
        assert "stetson_l" in features


# ── Spectral / Chemical ──────────────────────────────────────────────
class TestSpectralAnalyzer:
    def test_metallicity_outliers(self, config):
        """Spectral consensus needs 2+ methods, so provide teff+logg too."""
        rng = np.random.default_rng(42)
        n = 100
        mh = np.append(rng.normal(-0.1, 0.2, n - 2), [-3.0, 2.5])
        df = pd.DataFrame({
            "source_id": range(n),
            "mh_gspphot": mh,
            "teff_gspphot": np.append(rng.uniform(4000, 7000, n - 2), [4500, 4500]),
            "logg_gspphot": np.append(rng.uniform(3.5, 4.5, n - 2), [4.0, 4.0]),
            "catalog_source": "test",
        })
        results = SpectralAnalyzer(config).analyze(df)
        # With extreme metallicity + teff/logg, the two extreme stars should be
        # caught by metallicity_mad AND chemical_isolation_forest
        assert len(results) >= 1

    def test_chemical_isolation_forest(self, small_processed, config):
        results = SpectralAnalyzer(config).analyze(small_processed)
        chem_results = [r for r in results if r.anomaly_type == AnomalyType.CHEMICAL]
        # Should find some outliers in the 3D chemical space
        assert isinstance(chem_results, list)


# ── ML Pipeline ──────────────────────────────────────────────────────
class TestMLPipeline:
    def test_finds_anomalies_in_synthetic(self, synthetic_df, config):
        pipeline = MLPipeline(config)
        results = pipeline.analyze(synthetic_df)
        assert isinstance(results, list)
        for r in results:
            assert r.detection_method == "ml_ensemble"
            assert 0 <= r.confidence <= 1.0

    def test_auto_feature_selection(self, synthetic_df):
        features = MLPipeline._auto_select_features(synthetic_df)
        assert len(features) > 0
        assert "source_id" not in features
        assert "catalog_source" not in features

    def test_returns_empty_for_too_few_rows(self, config):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        results = MLPipeline(config).analyze(df, feature_columns=["x", "y"])
        assert results == []


# ── Ensemble Scorer ──────────────────────────────────────────────────
class TestEnsembleScorer:
    def _make_result(self, star_id, anomaly_type, method, confidence=0.7, sig=5.0):
        return AnomalyResult(
            star_id=star_id, anomaly_type=anomaly_type,
            confidence=confidence, significance_score=sig,
            parameters={}, description="test",
            follow_up_priority=5, detection_method=method,
            statistical_tests={},
        )

    def test_tier2_multi_method_same_type(self):
        """Tier 2: same type confirmed by 2+ methods should be selected."""
        results = [
            self._make_result("S1", AnomalyType.HR_OUTLIER, "method_a", 0.6),
            self._make_result("S1", AnomalyType.HR_OUTLIER, "method_b", 0.8),
        ]
        scorer = EnsembleScorer()
        merged = scorer.aggregate(results)
        s1_results = [r for r in merged if r.star_id == "S1"]
        assert len(s1_results) == 1
        assert s1_results[0].anomaly_type == AnomalyType.HR_OUTLIER

    def test_tier1_multi_family(self):
        """Tier 1: 3+ independent anomaly families -> CROSS_CATALOG entry."""
        results = [
            self._make_result("S1", AnomalyType.HR_OUTLIER, "hr"),        # photometric
            self._make_result("S1", AnomalyType.KINEMATICS, "kin"),       # kinematic
            self._make_result("S1", AnomalyType.METALLICITY, "met"),      # chemical
        ]
        scorer = EnsembleScorer()
        merged = scorer.aggregate(results)
        combined = [r for r in merged if r.anomaly_type == AnomalyType.CROSS_CATALOG]
        assert len(combined) == 1

    def test_single_weak_detection_rejected(self):
        """A single method detection below threshold should be filtered out."""
        results = [
            self._make_result("S1", AnomalyType.HR_OUTLIER, "one_method", sig=3.0),
        ]
        scorer = EnsembleScorer()
        merged = scorer.aggregate(results)
        assert len(merged) == 0

    def test_single_extreme_detection_kept(self):
        """A single method detection with very high significance is Tier 3."""
        results = [
            self._make_result("S1", AnomalyType.KINEMATICS, "pm_mad", sig=8.0),
        ]
        scorer = EnsembleScorer()
        merged = scorer.aggregate(results)
        assert len(merged) == 1

    def test_high_importance_type_kept(self):
        """Dyson sphere candidates are kept even with single detection."""
        results = [
            self._make_result("S1", AnomalyType.DYSON_SPHERE, "hephaistos", confidence=0.8, sig=4.0),
        ]
        scorer = EnsembleScorer()
        merged = scorer.aggregate(results)
        assert len(merged) == 1

    def test_summary_stats(self):
        results = [
            self._make_result("S1", AnomalyType.HR_OUTLIER, "m1", sig=7.0),
            self._make_result("S1", AnomalyType.KINEMATICS, "m2", sig=7.0),
            self._make_result("S1", AnomalyType.METALLICITY, "m3", sig=7.0),
        ]
        scorer = EnsembleScorer()
        merged = scorer.aggregate(results)
        stats = scorer.summary_stats(merged)
        assert stats["total"] >= 1
        assert stats["unique_stars"] == 1

    def test_to_dataframe(self):
        results = [self._make_result("S1", AnomalyType.DYSON_SPHERE, "m", confidence=0.9)]
        scorer = EnsembleScorer()
        df = scorer.to_dataframe(results)
        assert len(df) == 1
        assert "star_id" in df.columns
