"""Robust location/scale shared by the z-score detectors.

z = (x - median) / sigma_MAD with sigma_MAD = 1.4826 * MAD, the consistent estimate of the standard deviation for
Gaussian data, so a threshold of 4 means about 4 sigma. Falls back to the standard deviation when the MAD is 0.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import median_abs_deviation

MAD_TO_SIGMA = 1.4826


def robust_center_scale(values) -> tuple[float, float]:
    """(median, 1.4826 * MAD) of the finite values; std if the MAD is zero."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    center = float(np.median(v))
    scale = float(median_abs_deviation(v, scale="normal"))
    return center, scale if scale > 0 else float(np.std(v))


def rank_sigma(population, value: float) -> float:
    """Gaussian-equivalent sigma of ``value``'s upper-tail rank in ``population`` (higher = more extreme).

    p = (k - 0.5) / N with k the number of population values >= value; sigma = isf(p). The largest
    possible value is isf(0.5 / N), so a rank says nothing more extreme than the sample size allows.
    """
    from scipy.stats import norm

    pop = np.asarray(population, dtype=float)
    pop = pop[np.isfinite(pop)]
    if pop.size == 0 or not np.isfinite(value):
        return 0.0
    k = max(1, int(np.sum(pop >= value)))
    return float(norm.isf((k - 0.5) / pop.size))
