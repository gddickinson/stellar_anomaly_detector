"""PySide6 GUI for the Stellar Anomaly Detector.

Panels:
- main_window: QMainWindow with dockable layout, menus, toolbar
- catalog_browser: Catalog selection and fetch controls
- data_table: QTableView with pandas model, filtering, sorting
- property_inspector: Star detail panel with anomaly flags
- analysis_config: Parameter editor with presets
- job_manager: Job queue, progress bars, log output
- workers: QThread workers for non-blocking operations
- theme: Dark/light theme stylesheets
"""
