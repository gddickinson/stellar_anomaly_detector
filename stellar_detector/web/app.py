"""FastAPI web app for the Stellar Anomaly Detector: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8233, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below.

All analysis is the package's own pipeline (see pipeline.py); runs.py holds the demo field and recent runs.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import MODULES, config_fields, ensemble_rules
from .runs import DEMO, SOURCES, RunError, demo_run, get_run, new_run, star_detail, summary, upload_run

STATIC = Path(__file__).resolve().parent / "static"
ROOT = Path(__file__).resolve().parents[2]
TITLE = "Stellar Anomaly Detector"
MAX_UPLOAD = 50 * 1024 * 1024

app = FastAPI(title=TITLE)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Project summary: demo dataset, modules, sources, editable settings and the ensemble rules."""
    d = demo_run()
    return {
        "title": TITLE, "version": "6.0.0",
        "demo": {**DEMO, "n_input": d["n_input"], "n_stars": d["n_stars"], "n_detections": len(d["detections"]),
                 "n_candidates": len(d["candidates"])},
        "modules": [{"id": m, "label": lab} for m, lab, _ in MODULES],
        "sources": list(SOURCES),
        "config_fields": config_fields(),
        "ensemble": ensemble_rules(),
    }


def _run(run_id: str) -> dict:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(404, "run not found (the local app keeps the last few runs only)")
    return run


@app.get("/api/runs/{run_id}/summary")
def run_summary(run_id: str):
    """Counts per module, ensemble statistics, settings used and the HR main-sequence band."""
    return summary(_run(run_id))


@app.get("/api/runs/{run_id}/stars")
def run_stars(run_id: str):
    """Every star that passed the quality cuts, column-wise ({column: [values]}); source_id as strings."""
    return _run(run_id)["stars"]


@app.get("/api/runs/{run_id}/detections")
def run_detections(run_id: str):
    """Every raw detection from every module, before the ensemble."""
    return _run(run_id)["detections"]


@app.get("/api/runs/{run_id}/candidates")
def run_candidates(run_id: str):
    """Stars selected by EnsembleScorer, ranked as aggregate() ranks them, with their tier."""
    return _run(run_id)["candidates"]


@app.get("/api/runs/{run_id}/star/{star_id}")
def run_star(run_id: str, star_id: str):
    """One star: properties, raw detections and ensemble result."""
    d = star_detail(_run(run_id), star_id)
    if d is None:
        raise HTTPException(404, "star not in this run (it may have been removed by the quality cuts)")
    return d


@app.get("/api/docs/readme")
def readme():
    return {"title": "README", "markdown": (ROOT / "README.md").read_text(encoding="utf-8")}


@app.post("/api/run")
async def run(body: dict):
    """New run: {source: demo|gaia|hipparcos|tycho2|2mass|allwise|synthetic, n_stars, ra, dec, radius, config}."""
    try:
        run_id = await run_in_threadpool(new_run, body)
    except RunError as e:
        raise HTTPException(400, str(e))
    return {"run_id": run_id}


@app.post("/api/run/upload")
async def run_upload(file: UploadFile = File(...), config: str = Form("{}")):
    """Analyse an uploaded catalogue CSV (like `cli --file`)."""
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(400, "File too large (limit 50 MB); use the command line.")
    try:
        overrides = json.loads(config or "{}")
        run_id = await run_in_threadpool(upload_run, data, Path(file.filename or "upload.csv").name, overrides)
    except json.JSONDecodeError:
        raise HTTPException(400, "config must be JSON")
    except RunError as e:
        raise HTTPException(400, str(e))
    return {"run_id": run_id}


def _demo_star_paths() -> list[str]:
    return [f"/runs/demo/star/{s}" for s in demo_run()["stars"]["source_id"]]


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/docs/readme"] + [
    f"/runs/demo/{p}" for p in ("summary", "stars", "detections", "candidates")] + _demo_star_paths()
