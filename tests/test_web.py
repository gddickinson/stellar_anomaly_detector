"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stellar_detector.web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific API tests (never touch the network) -----------------------------------------------------------
import io  # noqa: E402
import json  # noqa: E402
import socket  # noqa: E402

from stellar_detector.analysis.ensemble import EnsembleScorer  # noqa: E402
from stellar_detector.data.preprocessing import preprocess_catalog  # noqa: E402
from stellar_detector.web import runs  # noqa: E402
from stellar_detector.web.pipeline import MODULES, make_config  # noqa: E402


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise RuntimeError("tests must not use the network")
    monkeypatch.setattr(socket.socket, "connect", refuse)


def test_demo_counts_consistent():
    s = client.get("/api/runs/demo/summary").json()
    assert s["n_input"] == 4112 and 0 < s["n_stars"] < s["n_input"]
    assert s["n_detections"] == sum(m["n"] for m in s["modules"])
    assert s["n_candidates"] == sum(s["tiers"].values()) == s["stats"]["total"]
    assert all(m["error"] is None for m in s["modules"])
    stars = client.get("/api/runs/demo/stars").json()
    assert len(stars["source_id"]) == s["n_stars"] + s["n_ruwe_set_aside"] and all(isinstance(x, str) for x in stars["source_id"])


def test_demo_candidates_match_ensemble_aggregate():
    """The web ranking (with tiers) is exactly EnsembleScorer.aggregate on the package's own analyzers."""
    cfg = make_config()
    df = preprocess_catalog(runs.demo_raw(), cfg).reset_index(drop=True)
    raw = [r for _, _, cls in MODULES for r in cls(cfg).analyze(df)]
    expected = EnsembleScorer().aggregate(raw)
    got = client.get("/api/runs/demo/candidates").json()
    assert [c["star_id"] for c in got] == [r.star_id for r in expected]
    assert [c["rank"] for c in got] == list(range(1, len(got) + 1))
    for c, r in zip(got, expected):
        assert c["significance_score"] == pytest.approx(r.significance_score, rel=1e-12)
        assert c["tier"] in (1, 2, 3)


def test_star_detail_and_featured_star():
    d = client.get(f"/api/runs/demo/star/{runs.BOYAJIAN_ID}").json()
    assert d["properties"]["source_id"] == runs.BOYAJIAN_ID
    top = client.get("/api/runs/demo/candidates").json()[0]
    t = client.get(f"/api/runs/demo/star/{top['star_id']}").json()
    assert t["candidate"]["rank"] == 1 and t["detections"]
    assert client.get("/api/runs/demo/star/123").status_code == 404
    assert client.get("/api/runs/nope/summary").status_code == 404


def test_run_synthetic_and_demo_with_settings():
    r = client.post("/api/run", json={"source": "synthetic", "n_stars": 400, "config": {"mad_threshold": 5}})
    assert r.status_code == 200, r.text
    s = client.get(f"/api/runs/{r.json()['run_id']}/summary").json()
    assert s["config"]["mad_threshold"] == 5 and s["config"]["outlier_threshold"] == 5
    assert "is_injected_anomaly" in s["columns"]
    r = client.post("/api/run", json={"source": "demo", "config": {"min_parallax_over_error": 10}})
    s2 = client.get(f"/api/runs/{r.json()['run_id']}/summary").json()
    assert s2["n_stars"] < client.get("/api/runs/demo/summary").json()["n_stars"]


def test_run_catalogue_fetch_is_called_with_the_region(monkeypatch):
    seen = {}

    def fake_fetch(self, source, n_stars, ra_center, dec_center, radius_deg, use_cache=True, **k):
        seen.update(source=source.value, n=n_stars, ra=ra_center, dec=dec_center, r=radius_deg)
        return runs.demo_raw().head(n_stars)
    monkeypatch.setattr(runs.DataFetcher, "fetch", fake_fetch)
    r = client.post("/api/run", json={"source": "gaia", "n_stars": 1500, "ra": 10, "dec": -5, "radius": 0.5})
    assert r.status_code == 200, r.text
    assert seen == {"source": "gaia_dr3", "n": 1500, "ra": 10.0, "dec": -5.0, "r": 0.5}


@pytest.mark.parametrize("body", [{"source": "nope"}, {"source": "demo", "config": {"mad_threshold": 99}},
                                  {"source": "gaia", "n_stars": "x"}])
def test_run_rejects_bad_input(body):
    assert client.post("/api/run", json=body).status_code == 400


def test_upload_csv():
    buf = io.BytesIO()
    runs.demo_raw().head(600).to_csv(buf, index=False)
    r = client.post("/api/run/upload", files={"file": ("field.csv", buf.getvalue(), "text/csv")},
                    data={"config": json.dumps({"mad_threshold": 4})})
    assert r.status_code == 200, r.text
    assert client.get(f"/api/runs/{r.json()['run_id']}/summary").json()["n_input"] == 600
    bad = client.post("/api/run/upload", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert bad.status_code == 400


def test_ensemble_fixture_is_current():
    """tests/fixtures/ensemble_fixture.json (used by the JS test) must match the Python code (tolerance: libm)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_ensemble_fixture
    saved = json.loads((ROOT / "tests" / "fixtures" / "ensemble_fixture.json").read_text())
    now = make_ensemble_fixture.build()
    assert [d["star_id"] for d in saved["detections"]] == [d["star_id"] for d in now["detections"]]
    for a, b in zip(saved["cases"], now["cases"]):
        assert [e["star_id"] for e in a["expected"]] == [e["star_id"] for e in b["expected"]], a["name"]
        for x, y in zip(a["expected"], b["expected"]):
            assert x["tier"] == y["tier"]
            assert x["significance_score"] == pytest.approx(y["significance_score"], rel=1e-9)
