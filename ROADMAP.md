# Stellar Anomaly Detector -- Roadmap

## Current State
A mature, well-architected modular Python package with 7 subpackages, 8 anomaly
detection modules, a PySide6 GUI with dockable panels, and 64 passing unit tests.
Supports 5+ astronomical catalogs (Gaia DR3, Hipparcos, Tycho-2, 2MASS, AllWISE),
ensemble ML scoring, SHAP explanations, and publication-quality exports. Has both
CLI and GUI entry points, a `pyproject.toml`, `INTERFACE.md`, and `CLAUDE.md`.
This is the most complete project in the collection. Phases 1-4 of the original
roadmap are done; Phases 5-6 are partially complete.

## Short-term Improvements
- [ ] Increase test coverage: add integration tests for the full CLI pipeline
      (fetch -> preprocess -> analyze -> ensemble -> export)
- [ ] Add property-based tests for `cross_match.py` epoch propagation edge cases
- [ ] Pin exact dependency versions in `pyproject.toml` for reproducible installs
- [ ] Add CI/CD via GitHub Actions: lint, test, build wheel
- [ ] Validate `DetectionConfig` parameters on construction (e.g., contamination
      must be in [0, 1], n_estimators > 0)
- [ ] Add docstrings to all GUI modules (`gui/*.py`)

## Feature Enhancements
- [ ] SAMP integration: interoperate with TOPCAT, Aladin, DS9 for astronomer
      workflows (listed in Phase 5 but not started)
- [ ] Visual ADQL/TAP query builder for custom catalog queries
- [ ] Batch processing queue for large catalog regions (Phase 4 backlog)
- [ ] Cross-catalog comparison view in the GUI: side-by-side or overlay plots
      (Phase 3 backlog)
- [ ] Add a "similar anomalies" search: given a selected star, find others with
      matching anomaly profiles using cosine similarity on feature vectors
- [ ] Light curve fetcher: pull ZTF/TESS light curves for selected stars directly
      from the GUI

## Long-term Vision
- [ ] Async data loading with QThread workers for all I/O operations (Phase 6)
- [ ] Data virtualization for 100k+ star catalogs with lazy loading
- [ ] Cross-platform packaging via PyInstaller or briefcase
- [ ] Citizen science mode: simplified UI for non-astronomers to review and
      classify anomaly candidates
- [ ] Real-time alert integration: subscribe to VOEvent streams for transient
      follow-up

## Technical Debt
- [ ] Some GUI modules may exceed 500 lines (`main_window.py`,
      `interactive_hr.py`) -- audit and split if needed
- [ ] `dimensionality.py` bundles t-SNE, UMAP, SHAP, and XGBoost -- consider
      splitting into focused modules
- [ ] The `stellar_data/` cache directory has no size limits or expiry policy
- [ ] Optional dependencies (coniferest, tensorflow, umap-learn) fail silently
      -- add clear startup warnings when advanced features are unavailable
- [ ] Undo/redo for analysis parameter changes (Phase 6 backlog)
- [ ] Keyboard shortcuts and command palette not yet implemented
