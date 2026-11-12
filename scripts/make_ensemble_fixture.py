"""Write tests/fixtures/ensemble_fixture.json: the demo field's raw detections and the Python EnsembleScorer output
for several rule settings. ensemble_engine.test.mjs checks the JS port against it; tests/test_web.py checks it is
current.

Run: python scripts/make_ensemble_fixture.py
"""
import json
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stellar_detector.analysis import ensemble  # noqa: E402
from stellar_detector.core.models import AnomalyResult, AnomalyType  # noqa: E402
from stellar_detector.web.pipeline import ensemble_with_tiers  # noqa: E402
from stellar_detector.web.runs import demo_run  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "ensemble_fixture.json"
KEYS = ("star_id", "anomaly_type", "family", "detection_method", "confidence", "significance_score",
        "follow_up_priority", "description", "module", "common_significance")
CASES = [
    {"name": "defaults", "opts": {}},
    {"name": "strict single-method", "opts": {"single_min_significance": 12.0}},
    {"name": "two families", "opts": {"tier1_min_families": 2}},
    {"name": "no lifetime, no ML", "opts": {"modules": ["hr", "kinematics", "spectral", "technosignature"]}},
]
BY_NAME = {t.anomaly_name: t for t in AnomalyType}


def to_result(d: dict) -> AnomalyResult:
    return AnomalyResult(
        star_id=d["star_id"], anomaly_type=BY_NAME[d["anomaly_type"]], confidence=d["confidence"],
        significance_score=d["significance_score"], parameters={}, description=d["description"],
        follow_up_priority=d["follow_up_priority"], detection_method=d["detection_method"],
        statistical_tests={"sigma_equiv": d["common_significance"]})


def python_output(detections: list[dict], opts: dict) -> list[dict]:
    mods = set(opts.get("modules") or [d["module"] for d in detections])
    raw = [to_result(d) for d in detections if d["module"] in mods]
    with mock.patch.multiple(
            ensemble,
            TIER1_MIN_TYPE_FAMILIES=opts.get("tier1_min_families", ensemble.TIER1_MIN_TYPE_FAMILIES),
            TIER2_MIN_METHODS=opts.get("tier2_min_methods", ensemble.TIER2_MIN_METHODS),
            SINGLE_METHOD_MIN_SIGNIFICANCE=opts.get("single_min_significance",
                                                    ensemble.SINGLE_METHOD_MIN_SIGNIFICANCE)):
        tiers = ensemble_with_tiers(raw)
    return [{"star_id": r.star_id, "tier": t, "anomaly_type": r.anomaly_type.anomaly_name,
             "confidence": r.confidence, "significance_score": r.significance_score,
             "follow_up_priority": r.follow_up_priority, "detection_method": r.detection_method,
             "description": r.description} for t, r in tiers]


def build() -> dict:
    detections = [{k: d[k] for k in KEYS} for d in demo_run()["detections"]]
    return {"detections": detections,
            "cases": [{**c, "expected": python_output(detections, c["opts"])} for c in CASES]}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = build()
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT} ({len(data['detections'])} detections, "
          f"{[len(c['expected']) for c in data['cases']]} candidates per case)")


if __name__ == "__main__":
    main()
