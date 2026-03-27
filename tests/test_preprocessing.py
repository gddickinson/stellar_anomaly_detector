"""Tests for data preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from stellar_detector.core.models import DetectionConfig
from stellar_detector.data.preprocessing import (
    apply_quality_filters,
    compute_derived_quantities,
    compute_quality_score,
    normalize_features,
    preprocess_catalog,
)


class TestComputeDerivedQuantities:
    def test_distance_from_parallax(self):
        df = pd.DataFrame({"parallax": [10.0, 5.0, 1.0]})
        result = compute_derived_quantities(df)
        np.testing.assert_allclose(result["distance_pc"], [100.0, 200.0, 1000.0])

    def test_negative_parallax_excluded(self):
        df = pd.DataFrame({"parallax": [-1.0, 0.0, 5.0]})
        result = compute_derived_quantities(df)
        assert pd.isna(result.loc[0, "distance_pc"])
        assert pd.isna(result.loc[1, "distance_pc"])
        assert result.loc[2, "distance_pc"] == pytest.approx(200.0)

    def test_absolute_magnitude_computed(self):
        df = pd.DataFrame({"parallax": [10.0], "phot_g_mean_mag": [10.0]})
        result = compute_derived_quantities(df)
        # M = 10 - 5*log10(100) + 5 = 10 - 10 + 5 = 5
        assert result.loc[0, "abs_mag"] == pytest.approx(5.0)

    def test_proper_motion_total(self):
        df = pd.DataFrame({"pmra": [3.0], "pmdec": [4.0]})
        result = compute_derived_quantities(df)
        assert result.loc[0, "pm_total"] == pytest.approx(5.0)

    def test_color_indices_2mass(self):
        df = pd.DataFrame({"Jmag": [10.0], "Hmag": [9.5], "Kmag": [9.0]})
        result = compute_derived_quantities(df)
        assert result.loc[0, "J_H"] == pytest.approx(0.5)
        assert result.loc[0, "H_K"] == pytest.approx(0.5)

    def test_wise_color_indices(self):
        df = pd.DataFrame({"W1mag": [8.0], "W2mag": [7.5], "W3mag": [6.0]})
        result = compute_derived_quantities(df)
        assert result.loc[0, "W1_W2"] == pytest.approx(0.5)
        assert result.loc[0, "W2_W3"] == pytest.approx(1.5)

    def test_luminosity_from_abs_mag(self):
        df = pd.DataFrame({"parallax": [10.0], "phot_g_mean_mag": [4.67]})
        result = compute_derived_quantities(df)
        # abs_mag = 4.67 - 5*log10(100) + 5 = -0.33 -> luminous
        assert result.loc[0, "luminosity_solar"] > 1.0


class TestQualityFilters:
    def test_low_parallax_snr_removed(self):
        df = pd.DataFrame({
            "parallax": [10.0, 0.5],
            "parallax_error": [1.0, 1.0],
        })
        config = DetectionConfig(min_parallax_over_error=5.0)
        result = apply_quality_filters(df, config)
        assert len(result) == 1

    def test_high_ruwe_removed(self):
        df = pd.DataFrame({"ruwe": [1.0, 1.2, 2.0, 3.5]})
        config = DetectionConfig(max_ruwe=1.4)
        result = apply_quality_filters(df, config)
        assert len(result) == 2

    def test_missing_values_retained(self):
        df = pd.DataFrame({"ruwe": [1.0, np.nan, 2.0]})
        config = DetectionConfig(max_ruwe=1.4)
        result = apply_quality_filters(df, config)
        assert len(result) == 2  # 1.0 and NaN kept


class TestQualityScore:
    def test_score_between_0_and_1(self, small_df):
        result = compute_quality_score(small_df)
        assert result["quality_score"].min() >= 0
        assert result["quality_score"].max() <= 1


class TestNormalizeFeatures:
    def test_median_centered(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
        result, stats = normalize_features(df, ["x"])
        assert "x_normalized" in result.columns
        assert stats["x"]["median"] == pytest.approx(3.5)

    def test_empty_column_skipped(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result, stats = normalize_features(df, ["x", "nonexistent"])
        assert "nonexistent_normalized" not in result.columns


class TestPreprocessCatalog:
    def test_full_pipeline_adds_derived_columns(self, synthetic_raw, config):
        result = preprocess_catalog(synthetic_raw, config)
        assert "distance_pc" in result.columns
        assert "abs_mag" in result.columns
        assert "pm_total" in result.columns
        assert "quality_score" in result.columns

    def test_pipeline_reduces_row_count(self, synthetic_raw, config):
        result = preprocess_catalog(synthetic_raw, config)
        # Quality filters should remove some rows
        assert len(result) <= len(synthetic_raw)
