"""Ensemble scoring: strict anomaly selection from multiple analysis modules.

Design philosophy: Individual analyzers detect broadly (for visualization and
exploration). The ensemble's job is to be *selective* — only surfacing stars
that are genuinely exceptional. In a random field of 1,000-5,000 stars, we
should expect 0-20 real anomalies, not hundreds.

Scoring tiers:
  - Tier 1 (highest): Multi-dimensional anomaly — flagged across 3+ independent
    anomaly categories (e.g., HR outlier + kinematic anomaly + chemical anomaly).
    These are the strongest candidates.
  - Tier 2: High-confidence single-category detection confirmed by 2+ independent
    methods within that category (e.g., DBSCAN + KDE + z-score all agree on HR outlier).
  - Tier 3: Extreme single-method detection — only if significance is very high
    (z > 6) or the anomaly type has intrinsic scientific importance (e.g., Dyson
    sphere candidate, stellar engine).

Stars that are merely in the statistical tail of one distribution are NOT reported
as anomalies — that's expected population diversity.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from ..core.models import AnomalyResult, AnomalyType

logger = logging.getLogger(__name__)

# Anomaly types that are scientifically important enough to report even
# with a single detection method, if the confidence is high
HIGH_IMPORTANCE_TYPES = {
    AnomalyType.DYSON_SPHERE,
    AnomalyType.STELLAR_ENGINE,
    AnomalyType.MEGASTRUCTURE,
    AnomalyType.INFRARED_EXCESS,
}

# Minimum significance for a single-method detection to be reported on its own
SINGLE_METHOD_MIN_SIGNIFICANCE = 6.0

# Minimum number of independent methods within one anomaly type for Tier 2
TIER2_MIN_METHODS = 2

# Minimum number of distinct anomaly type families for Tier 1
TIER1_MIN_TYPE_FAMILIES = 3

# Map anomaly types to independent "families" — detections within the same family
# are correlated and shouldn't be double-counted
TYPE_FAMILIES = {
    AnomalyType.HR_OUTLIER: "photometric",
    AnomalyType.LUMINOSITY: "photometric",
    AnomalyType.COLOR: "photometric",
    AnomalyType.PHOTOMETRY: "photometric",
    AnomalyType.KINEMATICS: "kinematic",
    AnomalyType.GALACTIC_ORBIT: "kinematic",
    AnomalyType.ASTROMETRIC: "astrometric",
    AnomalyType.BINARY_ANOMALY: "astrometric",
    AnomalyType.METALLICITY: "chemical",
    AnomalyType.CHEMICAL: "chemical",
    AnomalyType.VARIABILITY: "variability",
    AnomalyType.TEMPORAL: "variability",
    AnomalyType.ROTATION: "variability",
    AnomalyType.LIFETIME: "evolutionary",
    AnomalyType.MAGNETIC: "magnetic",
    AnomalyType.DYSON_SPHERE: "technosignature",
    AnomalyType.STELLAR_ENGINE: "technosignature",
    AnomalyType.MEGASTRUCTURE: "technosignature",
    AnomalyType.INFRARED_EXCESS: "infrared",
    AnomalyType.ISOLATION: "spatial",
    AnomalyType.CLUSTERING: "spatial",
    AnomalyType.GEOMETRIC: "spatial",
}


class EnsembleScorer:
    """Strict anomaly selection from multiple detection methods."""

    def __init__(self, min_methods: int = 1, boost_multi_method: bool = True):
        self.min_methods = min_methods
        self.boost_multi_method = boost_multi_method

    def aggregate(self, all_results: list[AnomalyResult]) -> list[AnomalyResult]:
        """Select genuinely exceptional stars from all raw detections.

        Applies tiered filtering to suppress population-diversity noise and
        surface only the most scientifically interesting targets.
        """
        if not all_results:
            return []

        by_star: dict[str, list[AnomalyResult]] = defaultdict(list)
        for r in all_results:
            by_star[r.star_id].append(r)

        selected: list[AnomalyResult] = []

        for star_id, detections in by_star.items():
            tier, result = self._evaluate_star(star_id, detections)
            if result is not None:
                selected.append(result)

        selected.sort(
            key=lambda r: (r.follow_up_priority, r.significance_score), reverse=True
        )

        n_raw_stars = len(by_star)
        logger.info(
            "Ensemble: %d raw detections across %d stars -> %d selected anomalies (%.1f%%)",
            len(all_results), n_raw_stars, len(selected),
            len(selected) / n_raw_stars * 100 if n_raw_stars else 0,
        )
        return selected

    def _evaluate_star(
        self, star_id: str, detections: list[AnomalyResult]
    ) -> tuple[int, AnomalyResult | None]:
        """Evaluate a star's detections and decide whether it qualifies as anomalous.

        Returns (tier, merged_result) where tier is 1-3 or 0 if rejected.
        """
        # Count independent anomaly families
        families = set()
        for d in detections:
            family = TYPE_FAMILIES.get(d.anomaly_type, d.anomaly_type.anomaly_name)
            families.add(family)

        # Group by anomaly type
        by_type: dict[AnomalyType, list[AnomalyResult]] = defaultdict(list)
        for d in detections:
            by_type[d.anomaly_type].append(d)

        # --- Tier 1: Multi-family anomaly ---
        if len(families) >= TIER1_MIN_TYPE_FAMILIES:
            return 1, self._build_tier1_result(star_id, detections, families)

        # --- Tier 2: Single type confirmed by multiple independent methods ---
        for atype, type_dets in by_type.items():
            methods = set(d.detection_method for d in type_dets)
            if len(methods) >= TIER2_MIN_METHODS:
                best = max(type_dets, key=lambda d: d.significance_score)
                return 2, self._build_tier2_result(star_id, atype, type_dets, methods)

        # --- Tier 3: Extreme single detection or high-importance type ---
        best = max(detections, key=lambda d: d.significance_score)
        if best.anomaly_type in HIGH_IMPORTANCE_TYPES and best.confidence >= 0.7:
            return 3, best
        if best.significance_score >= SINGLE_METHOD_MIN_SIGNIFICANCE:
            return 3, best

        # Not anomalous enough to report
        return 0, None

    def _build_tier1_result(
        self, star_id: str, detections: list[AnomalyResult], families: set[str]
    ) -> AnomalyResult:
        """Build result for a Tier 1 (multi-family) anomaly."""
        best = max(detections, key=lambda d: d.significance_score)
        family_list = sorted(families)
        all_methods = sorted(set(d.detection_method for d in detections))

        combined_params = {}
        for d in detections:
            combined_params.update(d.parameters)
        combined_params["n_families"] = len(families)
        combined_params["families"] = family_list

        combined_tests = {}
        for d in detections:
            for k, v in d.statistical_tests.items():
                combined_tests[f"{d.detection_method}_{k}"] = v

        return AnomalyResult(
            star_id=star_id,
            anomaly_type=AnomalyType.CROSS_CATALOG,
            confidence=min(1.0, best.confidence + 0.1 * (len(families) - 2)),
            significance_score=best.significance_score * (1 + 0.15 * (len(families) - 2)),
            parameters=combined_params,
            description=(
                f"Multi-dimensional anomaly across {len(families)} independent families "
                f"({', '.join(family_list)})"
            ),
            follow_up_priority=min(10, best.follow_up_priority + len(families) - 2),
            detection_method=f"tier1_ensemble({', '.join(all_methods[:3])}...)",
            statistical_tests=combined_tests,
            catalog_source=best.catalog_source,
            observational_recommendations=list(
                set(rec for d in detections for rec in d.observational_recommendations)
            ),
        )

    def _build_tier2_result(
        self,
        star_id: str,
        anomaly_type: AnomalyType,
        detections: list[AnomalyResult],
        methods: set[str],
    ) -> AnomalyResult:
        """Build result for a Tier 2 (multi-method confirmed) anomaly."""
        best = max(detections, key=lambda d: d.significance_score)
        method_list = sorted(methods)

        combined_params = {}
        for d in detections:
            combined_params.update(d.parameters)
        combined_params["n_confirming_methods"] = len(methods)

        combined_tests = {}
        for d in detections:
            for k, v in d.statistical_tests.items():
                combined_tests[f"{d.detection_method}_{k}"] = v

        return AnomalyResult(
            star_id=star_id,
            anomaly_type=anomaly_type,
            confidence=min(1.0, best.confidence + 0.05 * (len(methods) - 1)),
            significance_score=best.significance_score * (1 + 0.1 * (len(methods) - 1)),
            parameters=combined_params,
            description=(
                f"{best.description} [confirmed by {len(methods)} methods: "
                f"{', '.join(method_list)}]"
            ),
            follow_up_priority=min(10, best.follow_up_priority + len(methods) - 1),
            detection_method=f"tier2_ensemble({', '.join(method_list)})",
            statistical_tests=combined_tests,
            catalog_source=best.catalog_source,
            observational_recommendations=list(
                set(rec for d in detections for rec in d.observational_recommendations)
            ),
        )

    def to_dataframe(self, results: list[AnomalyResult]) -> pd.DataFrame:
        """Convert a list of AnomalyResults to a summary DataFrame."""
        if not results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in results])

    def summary_stats(self, results: list[AnomalyResult]) -> dict:
        """Compute summary statistics for a set of results."""
        if not results:
            return {"total": 0}

        type_counts = defaultdict(int)
        for r in results:
            type_counts[r.anomaly_type.anomaly_name] += 1

        confidences = [r.confidence for r in results]
        return {
            "total": len(results),
            "unique_stars": len({r.star_id for r in results}),
            "by_type": dict(type_counts),
            "mean_confidence": float(np.mean(confidences)),
            "max_confidence": float(np.max(confidences)),
            "high_priority_count": sum(1 for r in results if r.follow_up_priority >= 7),
        }
