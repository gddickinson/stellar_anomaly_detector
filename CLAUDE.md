# Stellar Anomaly Detector v6.0

Advanced stellar anomaly detector and technosignature search engine.

## Quick Reference

- **Main entry point**: `stellar_detector/cli.py:main()`
- **GUI entry point**: `stellar_detector/gui/main_window.py:launch_app()`
- **Navigation map**: @INTERFACE.md
- **Development roadmap**: @ROADMAP.md

## Architecture

Modular Python package (`stellar_detector/`) with 7 subpackages:
- `core/` — Data models, config, constants
- `data/` — Catalog fetching, preprocessing, cross-matching
- `analysis/` — 8 anomaly detection modules (HR, lifetime, kinematics, variability, spectral, technosignature, ML, ensemble)
- `models/` — Astrophysical models (stellar evolution)
- `visualization/` — Matplotlib plots
- `gui/` — PySide6 GUI (Phase 2+)
- `utils/` — Logging, I/O

## Running

```bash
# CLI with synthetic data
python -m stellar_detector.cli --source synthetic --stars 2000

# Multi-catalog
python -m stellar_detector.cli --multi gaia,hipparcos --stars 1000 --report

# GUI
python -m stellar_detector.cli --gui
```

## Key Design Decisions

- Every analysis module follows the same interface: `analyzer.analyze(df) -> list[AnomalyResult]`
- `DetectionConfig` centralizes all tunable parameters
- `EnsembleScorer` deduplicates and merges results from all modules
- Quality filters follow Gaia DR3 best practices (RUWE < 1.4, parallax_over_error > 5)
- Technosignature detection implements the Project Hephaistos methodology
