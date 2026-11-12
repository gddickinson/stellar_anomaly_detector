"""Web interface (FastAPI + plain ES-module front end) and static-site data for the Stellar Anomaly Detector.

- app.py       routes under /api and STATIC_ENDPOINTS (exported by scripts/build_static.py)
- pipeline.py  runs the package's own preprocessing, analyzers and EnsembleScorer; JSON conversion
- runs.py      the demo field (demo/*.csv, Gaia DR3) and recent local runs
- static/      the front end (index.html, core.js, page modules, ensemble_engine.js)
"""
