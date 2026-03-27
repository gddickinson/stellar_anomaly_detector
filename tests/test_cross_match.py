"""Tests for cross-catalog matching."""

import numpy as np
import pandas as pd
import pytest

from stellar_detector.data.cross_match import CrossCatalogMatcher


class TestCrossCatalogMatcher:
    def test_exact_position_match(self):
        cat_a = pd.DataFrame({"ra": [180.0], "dec": [0.0], "source_id": ["A1"]})
        cat_b = pd.DataFrame({"ra": [180.0], "dec": [0.0], "source_id": ["B1"]})
        matcher = CrossCatalogMatcher(max_separation_arcsec=10.0)
        result = matcher.match(cat_a, cat_b)
        assert len(result) == 1
        assert result["separation_arcsec"].iloc[0] < 0.1

    def test_no_match_beyond_threshold(self):
        cat_a = pd.DataFrame({"ra": [180.0], "dec": [0.0]})
        cat_b = pd.DataFrame({"ra": [181.0], "dec": [0.0]})  # ~1 degree away
        matcher = CrossCatalogMatcher(max_separation_arcsec=5.0)
        result = matcher.match(cat_a, cat_b)
        assert len(result) == 0

    def test_multiple_matches(self):
        cat_a = pd.DataFrame({
            "ra": [180.0, 180.001, 190.0],
            "dec": [0.0, 0.0, 0.0],
        })
        cat_b = pd.DataFrame({
            "ra": [180.0005, 190.0001],
            "dec": [0.0, 0.0],
        })
        matcher = CrossCatalogMatcher(max_separation_arcsec=10.0)
        result = matcher.match(cat_a, cat_b)
        assert len(result) >= 1

    def test_epoch_propagation_with_pm(self):
        # Star with high proper motion: position shifts between epochs
        cat_a = pd.DataFrame({
            "ra": [180.0], "dec": [0.0],
            "pmra": [100.0], "pmdec": [0.0],  # 100 mas/yr
        })
        cat_b = pd.DataFrame({
            "ra": [180.0], "dec": [0.0],
        })
        matcher = CrossCatalogMatcher(max_separation_arcsec=5.0)
        # Same epoch: should match
        result_same = matcher.match(cat_a, cat_b, epoch_a=2016.0, epoch_b=2016.0)
        assert len(result_same) == 1

    def test_match_multiple(self):
        cats = {
            "A": pd.DataFrame({"ra": [180.0, 181.0], "dec": [0.0, 0.0]}),
            "B": pd.DataFrame({"ra": [180.0001], "dec": [0.0]}),
            "C": pd.DataFrame({"ra": [181.0001], "dec": [0.0]}),
        }
        epochs = {"A": 2016.0, "B": 2016.0, "C": 2016.0}
        matcher = CrossCatalogMatcher(max_separation_arcsec=5.0)
        results = matcher.match_multiple(cats, epochs)
        assert "A__B" in results
        assert "A__C" in results
        assert "B__C" in results

    def test_missing_columns_returns_empty(self):
        cat_a = pd.DataFrame({"x": [1]})
        cat_b = pd.DataFrame({"ra": [180.0], "dec": [0.0]})
        matcher = CrossCatalogMatcher()
        result = matcher.match(cat_a, cat_b)
        assert len(result) == 0
