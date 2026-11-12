"""Robust z-scores must be in units of sigma: (x - median) / (1.4826 * MAD)."""

import numpy as np
import pandas as pd
import pytest

from stellar_detector.analysis.kinematics import KinematicsAnalyzer
from stellar_detector.analysis.spectral import SpectralAnalyzer
from stellar_detector.core.models import DetectionConfig


def _gaussian(n=4001, seed=3):
    x = np.random.default_rng(seed).normal(0.0, 1.0, n)
    return (x - np.median(x)) / (1.4826 * np.median(np.abs(x - np.median(x))))  # exactly median 0, robust sigma 1


def test_metallicity_z_is_in_sigma_units():
    mh = np.append(_gaussian(), [3.5, 6.0])  # 3.5 sigma: below the 4-sigma threshold; 6 sigma: above
    df = pd.DataFrame({"source_id": [f"S{i}" for i in range(len(mh))], "mh_gspphot": mh})
    res = SpectralAnalyzer(DetectionConfig())._metallicity_outliers(df)
    flagged = {r.star_id: r.parameters["z_score"] for r in res}
    assert f"S{len(mh) - 2}" not in flagged
    assert flagged[f"S{len(mh) - 1}"] == pytest.approx(6.0, rel=0.01)


def test_proper_motion_z_is_in_sigma_units():
    pm = np.append(10 + _gaussian(), [13.5, 16.0])
    df = pd.DataFrame({"source_id": [f"S{i}" for i in range(len(pm))], "pm_total": pm})
    res = KinematicsAnalyzer(DetectionConfig())._proper_motion_outliers(df)
    flagged = {r.star_id: r.parameters["z_score"] for r in res}
    assert f"S{len(pm) - 2}" not in flagged
    assert flagged[f"S{len(pm) - 1}"] == pytest.approx(6.0, rel=0.01)
