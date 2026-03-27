"""Tests for core data models and stellar evolution models."""

import pytest

from stellar_detector.core.models import (
    AnomalyResult,
    AnomalyType,
    CatalogSource,
    DetectionConfig,
)
from stellar_detector.models.stellar_evolution import StellarEvolutionModels


class TestAnomalyType:
    def test_all_types_have_significance(self):
        for t in AnomalyType:
            assert 1 <= t.significance <= 10

    def test_dyson_sphere_highest_significance(self):
        assert AnomalyType.DYSON_SPHERE.significance == 10

    def test_anomaly_name_attribute(self):
        assert AnomalyType.HR_OUTLIER.anomaly_name == "hr_diagram_outlier"


class TestAnomalyResult:
    def test_to_dict_contains_key_fields(self):
        r = AnomalyResult(
            star_id="STAR_001",
            anomaly_type=AnomalyType.HR_OUTLIER,
            confidence=0.85,
            significance_score=5.2,
            parameters={"z_score": 4.1},
            description="Test anomaly",
            follow_up_priority=7,
            detection_method="test",
            statistical_tests={"z": 4.1},
        )
        d = r.to_dict()
        assert d["star_id"] == "STAR_001"
        assert d["anomaly_type"] == "hr_diagram_outlier"
        assert d["confidence"] == 0.85

    def test_default_timestamp_set(self):
        r = AnomalyResult(
            star_id="X", anomaly_type=AnomalyType.COLOR, confidence=0.5,
            significance_score=1.0, parameters={}, description="",
            follow_up_priority=1, detection_method="t", statistical_tests={},
        )
        assert r.timestamp is not None


class TestDetectionConfig:
    def test_default_values(self):
        c = DetectionConfig()
        assert c.outlier_threshold == 4.0
        assert c.mad_threshold == 4.0
        assert c.isolation_n_estimators == 200
        assert c.max_ruwe == 1.4
        assert c.dyson_temp_min_k == 100.0

    def test_custom_override(self):
        c = DetectionConfig(outlier_threshold=5.0, isolation_contamination=0.1)
        assert c.outlier_threshold == 5.0
        assert c.isolation_contamination == 0.1


class TestCatalogSource:
    def test_gaia_value(self):
        assert CatalogSource.GAIA_DR3.value == "gaia_dr3"

    def test_all_sources_have_string_value(self):
        for s in CatalogSource:
            assert isinstance(s.value, str)


class TestStellarEvolutionModels:
    @pytest.fixture
    def models(self):
        return StellarEvolutionModels()

    def test_ms_mag_increases_with_redder_color(self, models):
        blue = models.main_sequence_mag_from_color(0.0)
        red = models.main_sequence_mag_from_color(2.0)
        assert red > blue  # redder stars are fainter (higher mag)

    def test_mass_from_abs_mag_sun(self, models):
        solar_mass = models.mass_from_abs_mag(models.SOLAR_ABS_MAG_G)
        assert 0.8 < solar_mass < 1.3  # should be near 1.0

    def test_mass_luminosity_roundtrip(self, models):
        for mass in [0.5, 1.0, 2.0, 5.0]:
            lum = models.luminosity_from_mass(mass)
            recovered = models.mass_from_luminosity(lum)
            assert abs(recovered - mass) < 0.01

    def test_lifetime_decreases_with_mass(self, models):
        lt_low = models.main_sequence_lifetime(0.5)
        lt_high = models.main_sequence_lifetime(5.0)
        assert lt_low > lt_high  # low-mass stars live longer

    def test_zams_fainter_than_tams(self, models):
        for teff in [4000, 6000, 10000]:
            zams = models.zams_mag_from_teff(teff)
            tams = models.tams_mag_from_teff(teff)
            assert zams > tams  # ZAMS is fainter (higher mag)

    def test_temperature_from_color_monotonic(self, models):
        t_blue = models.temperature_from_color(0.0)
        t_red = models.temperature_from_color(3.0)
        assert t_blue > t_red  # bluer = hotter
