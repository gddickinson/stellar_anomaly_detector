"""CLI argument handling (no network, no analysis run)."""

from stellar_detector.cli import build_parser, config_from_args
from stellar_detector.core.models import DetectionConfig


def test_outlier_threshold_reaches_the_analyzers():
    """The analyzers read mad_threshold; --outlier-threshold must set it (as the GUI presets do)."""
    cfg = config_from_args(build_parser().parse_args(["--outlier-threshold", "3.0"]))
    assert cfg.mad_threshold == 3.0 and cfg.outlier_threshold == 3.0


def test_defaults_keep_detection_config_thresholds():
    cfg = config_from_args(build_parser().parse_args([]))
    assert cfg.mad_threshold == DetectionConfig().mad_threshold
    assert cfg.isolation_contamination == 0.03
