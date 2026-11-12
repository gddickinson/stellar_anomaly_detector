# Session log

## 2026-11-11 — Web interface, static site and bug fixes (branch `web-interface`)

**Bug fixes** (each with a regression test that failed first, in its own commit):
- Dyson-sphere SED fit (`analysis/technosignature.py`) could never flag a star: observed fluxes are F_nu (Jy zero
  points) but the model was B_lambda, and the shell term was f·B(T_DS) per unit area (no energy conservation), so a
  100–700 K shell added no flux. Now F_nu with each blackbody normalised to its share of the bolometric flux.
- Dyson-sphere SNR check: a missing W3/W4 error (AllWISE upper limit) counted as SNR 5; now not detected.
- ML pipeline auto-features included `ra`/`dec` despite the docstring; excluded by exact name.
- Synthetic generator: injected +2..6 mag HR offsets were erased by preprocessing (G was independent of abs_mag);
  G is now derived from abs_mag and parallax.
- CLI `--outlier-threshold` set a field no analyzer reads; it now sets `mad_threshold` (default unchanged).

**Web interface** (`stellar_detector/web/`, `run_web.py`, port 8233): FastAPI app running the package's own pipeline
(fetch any catalogue/region, CSV upload, editable DetectionConfig fields), pages Run, Plots, Candidates, Scores, Star,
Method, About, README. Static site (GitHub Pages) from `scripts/build_static.py`: demo field precomputed, ensemble
explorer in the browser (`ensemble_engine.js`, matches Python exactly on 4 rule settings).

**Demo data:** Gaia DR3 cone of 0.15° around KIC 8462852 (4,112 sources; 785 after quality cuts; 181 raw detections;
120 ensemble candidates, 5 Tier 1). KIC 8462852 itself is not flagged. CDS VizieR was down (the CfA mirror returned
mislabelled columns), so no AllWISE/2MASS photometry: the technosignature module has nothing to test on the demo.

**Needs George's decision (not changed):**
- Ensemble Tier 3 compares one significance cut (6) across scales that differ by module: the lifetime test emits
  10 × age fraction (always ≥ 9 when flagged) and the ML pipeline 10 × score (always ≥ 7), so every such detection is
  "extreme". In the demo, 59 of 120 candidates are lifetime-only (mostly giants).
- The quality cut removes RUWE > 1.4 before the RUWE-anomaly test (threshold 1.4), so that test never fires.
- Synthetic generator main sequence (M = −10 log T + 40) is brighter than the lifetime ZAMS (−8 log T + 34.5), so the
  lifetime test flags ~95% of synthetic stars.
- `cli --multi` concatenates catalogues row-wise instead of cross-matching, so Gaia + AllWISE rows never share a
  star and the Dyson/IR tests cannot run from the CLI; the intended merge (esp. for overlapping catalogues like
  Hipparcos) is unclear.
- Robust z-scores use the unscaled MAD (4 MAD ≈ 2.7 σ) while descriptions say "z".
- The Dyson thresholds (RMSE < 0.2, improvement > 0.05) are uncalibrated now that the fit works.

Tests: 79 unit tests + 805 web tests pass; node ensemble test passes; screenshots smoke test all ok (static and local).
The Mac mini was not used.
