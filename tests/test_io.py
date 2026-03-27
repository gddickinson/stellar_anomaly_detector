"""Tests for I/O utilities."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from stellar_detector.analysis.ensemble import EnsembleScorer
from stellar_detector.core.models import AnomalyResult, AnomalyType
from stellar_detector.utils.io import export_report, load_catalog_csv, save_results


def _make_result(star_id="S1"):
    return AnomalyResult(
        star_id=star_id, anomaly_type=AnomalyType.HR_OUTLIER,
        confidence=0.9, significance_score=7.0,
        parameters={"z": 4.5}, description="Test anomaly",
        follow_up_priority=8, detection_method="test",
        statistical_tests={"z": 4.5},
    )


class TestSaveResults:
    def test_save_csv(self, tmp_path):
        results = [_make_result("A"), _make_result("B")]
        path = save_results(results, str(tmp_path), filename="test", fmt="csv")
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 2

    def test_save_json(self, tmp_path):
        results = [_make_result()]
        path = save_results(results, str(tmp_path), filename="test", fmt="json")
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["star_id"] == "S1"

    def test_invalid_format_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported"):
            save_results([_make_result()], str(tmp_path), fmt="xml_invalid")


class TestLoadCatalogCSV:
    def test_load_existing_file(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"ra": [1, 2], "dec": [3, 4]}).to_csv(csv_path, index=False)
        df = load_catalog_csv(str(csv_path))
        assert len(df) == 2

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_catalog_csv("/nonexistent/path.csv")


class TestExportReport:
    def test_html_report_created(self, tmp_path):
        results = [_make_result("A"), _make_result("B")]
        df = pd.DataFrame({"source_id": ["A", "B", "C"], "ra": [1, 2, 3]})
        path = export_report(results, df, str(tmp_path))
        assert path.exists()
        html = path.read_text()
        assert "Stellar Anomaly Detection Report" in html
        assert "Stars Analyzed" in html
