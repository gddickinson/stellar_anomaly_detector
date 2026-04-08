# Stellar Anomaly Detector v6.0 — Interface Map

## Package: `stellar_detector/`

### `core/` — Data Models & Configuration
- **`models.py`** — Central type definitions
  - `AnomalyType(Enum)` — 23 anomaly types with significance ratings (1-10)
  - `CatalogSource(Enum)` — Supported catalog sources (Gaia DR3, Hipparcos, Tycho-2, 2MASS, AllWISE, etc.)
  - `AnomalyResult` — Dataclass holding a single detection result (star_id, type, confidence, method, etc.)
  - `DetectionConfig` — Dataclass with ~50 tunable parameters for all detection methods
- **`constants.py`** — Physical constants, quality thresholds, catalog metadata, display names

### `data/` — Data Acquisition & Preprocessing
- **`fetcher.py`** — `DataFetcher` class
  - `fetch(source, n_stars, ra, dec, radius)` — Dispatch to catalog-specific fetchers
  - `_fetch_gaia()`, `_fetch_hipparcos()`, `_fetch_tycho2()`, `_fetch_2mass()`, `_fetch_allwise()`
  - `_generate_synthetic()` — Test data with injected anomalies
- **`preprocessing.py`** — Pipeline functions
  - `preprocess_catalog(df, config)` — Full pipeline: derived quantities + quality filters + scoring
  - `compute_derived_quantities(df)` — Distance, abs_mag, pm_total, v_tan, color indices, luminosity
  - `apply_quality_filters(df, config)` — Gaia DR3 quality cuts (parallax SNR, RUWE, BP-RP excess)
  - `normalize_features(df, columns)` — Robust median/MAD normalization
- **`cross_match.py`** — `CrossCatalogMatcher` class
  - `match(catalog_a, catalog_b, epochs)` — Positional matching with epoch propagation + FoM scoring
  - `match_multiple(catalogs, epochs)` — Pairwise cross-match of multiple catalogs

### `analysis/` — Anomaly Detection Modules
- **`hr_diagram.py`** — `HRDiagramAnalyzer`
  - `analyze(df)` — Main-sequence deviation (robust z-score), DBSCAN noise, KDE low-density, GMM low-likelihood
- **`stellar_lifetime.py`** — `StellarLifetimeAnalyzer`
  - `analyze(df)` — Flags stars near end of expected MS lifetime using mass/metallicity models
- **`kinematics.py`** — `KinematicsAnalyzer`
  - `analyze(df)` — Proper motion outliers, RUWE > 1.4, astrometric excess noise, hypervelocity candidates
- **`variability.py`** — `VariabilityAnalyzer`
  - `analyze(df)` — Photometric scatter outliers, Stetson J index anomalies
  - `analyze_light_curve(times, mags, errs)` — 30+ feature extraction (Stetson J/K/L, Lomb-Scargle, stats)
- **`spectral.py`** — `SpectralAnalyzer`
  - `analyze(df)` — Metallicity outliers, chemical ratio anomalies (IF), Teff-[M/H] outliers (LOF)
  - `analyze_spectra_autoencoder(spectra)` — 10-model ensemble autoencoder for spectral anomalies
- **`technosignature.py`** — `TechnosignatureAnalyzer`
  - `analyze(df)` — Dyson sphere SED grid search (Project Hephaistos), IR excess (WISE), stellar engines
- **`ml_pipeline.py`** — `MLPipeline`
  - `analyze(df, feature_columns)` — Ensemble ML: Isolation Forest + LOF + OCSVM + Adaptive IF
- **`ensemble.py`** — `EnsembleScorer`
  - `aggregate(results)` — Deduplicate, merge multi-method detections, boost confidence
  - `to_dataframe(results)` — Convert to pandas DataFrame
  - `summary_stats(results)` — Compute summary statistics
- **`dimensionality.py`** — Advanced ML utilities
  - `compute_tsne_embedding(df, features)` — t-SNE (openTSNE or sklearn fallback)
  - `compute_umap_embedding(df, features)` — UMAP dimensionality reduction
  - `explain_anomalies_shap(df, features, results)` — SHAP feature contributions per anomaly
  - `xgboost_chemical_classifier(df)` — XGBoost chemical peculiar star classification
- **`persistence.py`** — `ResultStore`
  - `save_run(run_id, results)` — Store results in DuckDB/SQLite
  - `load_run(run_id)` / `list_runs()` / `query(sql)` — Retrieve and query stored results

### `models/` — Astrophysical Models
- **`stellar_evolution.py`** — `StellarEvolutionModels`
  - Main sequence mag from color (solar/low-Z/high-Z metallicity models)
  - Mass-luminosity and luminosity-mass relations
  - Main-sequence lifetime (mass + metallicity dependent)
  - ZAMS/TAMS magnitude from temperature
  - Color-temperature calibrations (Gaia BP-RP)

### `visualization/` — Static Plotting
- **`plots.py`** — Matplotlib figure generators
  - `plot_hr_diagram(df, anomalies)` — CMD with anomaly overlay
  - `plot_sky_map(df, anomalies)` — Mollweide/Aitoff projection
  - `plot_anomaly_dashboard(df, anomalies)` — 6-panel summary dashboard
  - `plot_anomaly_distribution(anomalies)` — Bar chart by anomaly type

### `gui/` — PySide6 GUI (13 modules)
- **`main_window.py`** — `MainWindow(QMainWindow)` + `launch_app()`
  - Dockable panels, menu bar (File/Data/Analysis/View/Help), toolbar, status bar
  - Integrates all panels: catalog browser, data table, property inspector, config, jobs, dashboard
- **`catalog_browser.py`** — `CatalogBrowserWidget` — catalog tree + fetch controls
- **`data_table.py`** — `DataTableWidget` / `PandasTableModel` — sortable, filterable QTableView
- **`property_inspector.py`** — `PropertyInspectorWidget` — star detail panel with anomaly flags
- **`analysis_config.py`** — `AnalysisConfigWidget` — parameter editor with 4 presets
- **`job_manager.py`** — `JobManagerWidget` — progress bars, cancel, log output
- **`workers.py`** — `FetchWorker` / `AnalysisWorker` — QThread background workers
- **`theme.py`** — Dark (Catppuccin Mocha) and Light (Catppuccin Latte) stylesheets
- **`dashboard.py`** — `DashboardWidget` — tabbed container for all visualization panels
- **`interactive_hr.py`** — `InteractiveHRWidget` — pyqtgraph HR diagram (zoom, hover, color modes)
- **`sky_map.py`** — `SkyMapWidget` — Mollweide/Aitoff/Hammer/Lambert sky projection
- **`galactic_3d.py`** — `Galactic3DWidget` — pyqtgraph 3D scatter in galactic coordinates
- **`light_curve_viewer.py`** — `LightCurveWidget` — time-series + Lomb-Scargle + phase folding

### `utils/` — Utilities
- **`logging_config.py`** — `setup_logging(level, log_file)` — Console + file logging
- **`io.py`** — File I/O and reports
  - `save_results(results, output_dir, fmt)` — Export to CSV/JSON/VOTable
  - `load_catalog_csv(filepath)` — Load catalog from CSV
  - `export_report(results, df, output_dir)` — Generate HTML analysis report
- **`session.py`** — `Session` class — workspace save/restore (JSON + CSV)
- **`annotations.py`** — `AnnotationStore` — tag/comment/classify anomalies with persistence
- **`export.py`** — Publication exports
  - `export_votable(df, filepath)` — VOTable export
  - `export_fits(df, filepath)` — FITS binary table export
  - `export_latex_table(results, filepath)` — LaTeX table for papers
  - `export_publication_plot(fig, filepath)` — SVG/PDF publication-ready figures

### `cli.py` — Command-Line Interface
- `main()` — Entry point: parse args, fetch data, run all analyzers, aggregate, output
- Supports `--gui`, `--source`, `--multi`, `--file`, `--stars`, `--report`, `--format`

## Module Dependencies

```
cli.py / gui/main_window.py
  ├── core/ (models, constants)
  ├── data/ (fetcher -> preprocessing -> cross_match)
  ├── analysis/ (hr_diagram, stellar_lifetime, kinematics, variability,
  │              spectral, technosignature, ml_pipeline -> ensemble)
  │     ├── models/stellar_evolution.py
  │     ├── dimensionality.py (t-SNE, UMAP, SHAP, XGBoost)
  │     └── persistence.py (DuckDB/SQLite)
  ├── visualization/plots.py
  ├── gui/ (dashboard -> interactive_hr, sky_map, galactic_3d, light_curve_viewer)
  │     ├── catalog_browser, data_table, property_inspector, analysis_config
  │     ├── job_manager, workers
  │     └── theme
  └── utils/ (io, logging_config, session, annotations, export)
```
