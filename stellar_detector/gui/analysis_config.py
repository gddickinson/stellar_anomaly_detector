"""Analysis configuration panel — parameter editor with presets and validation."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.models import DetectionConfig

# Presets for common use cases
PRESETS = {
    "Default": DetectionConfig(),
    "Conservative": DetectionConfig(
        outlier_threshold=5.0, mad_threshold=5.0,
        isolation_contamination=0.01, lof_contamination=0.01,
        ensemble_threshold=0.85,
    ),
    "Sensitive": DetectionConfig(
        outlier_threshold=3.0, mad_threshold=3.0,
        isolation_contamination=0.05, lof_contamination=0.05,
        ensemble_threshold=0.5,
    ),
    "Technosignature Focus": DetectionConfig(
        dyson_sed_rmse_max=0.3, dyson_snr_min=2.5,
        dyson_temp_min_k=80, dyson_temp_max_k=800,
        dyson_covering_min=0.05, dyson_covering_max=0.95,
    ),
}

# Tooltips for key parameters
TOOLTIPS = {
    "outlier_threshold": "Z-score threshold for flagging statistical outliers",
    "mad_threshold": "Median Absolute Deviation threshold for robust outlier detection",
    "isolation_n_estimators": "Number of trees in Isolation Forest ensemble",
    "isolation_contamination": "Expected fraction of anomalies (0.01 = 1%)",
    "lof_n_neighbors": "Number of neighbors for Local Outlier Factor",
    "dbscan_eps": "DBSCAN neighborhood radius (auto-tuned if too small)",
    "max_ruwe": "Maximum RUWE for clean astrometry (Gaia DR3: 1.4)",
    "min_parallax_over_error": "Minimum parallax SNR for reliable distances",
    "dyson_temp_min_k": "Minimum Dyson sphere temperature to search (Kelvin)",
    "dyson_temp_max_k": "Maximum Dyson sphere temperature to search (Kelvin)",
    "dyson_sed_rmse_max": "Maximum SED fit residual for Dyson sphere candidates",
    "ensemble_threshold": "Minimum ensemble score to flag as anomaly",
}

# Group parameters by category
PARAM_GROUPS = {
    "Statistical Thresholds": [
        "outlier_threshold", "robust_outlier_threshold", "mad_threshold",
    ],
    "Machine Learning": [
        "isolation_n_estimators", "isolation_contamination", "isolation_max_samples",
        "lof_n_neighbors", "lof_contamination", "ocsvm_nu", "ensemble_threshold",
    ],
    "Clustering": [
        "dbscan_eps", "dbscan_min_samples", "hdbscan_min_cluster_size", "gmm_n_components",
    ],
    "Data Quality (Gaia)": [
        "min_parallax_over_error", "max_ruwe", "max_phot_bp_rp_excess", "min_phot_bp_rp_excess",
    ],
    "Technosignature": [
        "dyson_temp_min_k", "dyson_temp_max_k", "dyson_covering_min", "dyson_covering_max",
        "dyson_sed_rmse_max", "dyson_snr_min",
    ],
    "Variability": [
        "variability_threshold", "period_range_min", "period_range_max", "stetson_j_threshold",
    ],
    "Processing": [
        "min_data_points", "max_missing_fraction", "bootstrap_iterations", "n_jobs",
    ],
}


class AnalysisConfigWidget(QWidget):
    """Parameter editor panel with presets and grouped controls."""

    config_changed = Signal(DetectionConfig)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._widgets: dict[str, Any] = {}
        self._config = DetectionConfig()
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)

        # Preset selector
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(PRESETS.keys())
        self._preset_combo.currentTextChanged.connect(self._apply_preset)
        preset_row.addWidget(self._preset_combo)
        outer.addLayout(preset_row)

        # Scrollable parameter area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)

        for group_name, param_names in PARAM_GROUPS.items():
            group = QGroupBox(group_name)
            form = QFormLayout(group)
            for name in param_names:
                widget = self._create_widget(name)
                if widget:
                    tooltip = TOOLTIPS.get(name, "")
                    if tooltip:
                        widget.setToolTip(tooltip)
                    form.addRow(name.replace("_", " ").title() + ":", widget)
                    self._widgets[name] = widget
            layout.addWidget(group)

        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

        # Apply button
        apply_btn = QPushButton("Apply Configuration")
        apply_btn.setProperty("primary", True)
        apply_btn.clicked.connect(self._emit_config)
        outer.addWidget(apply_btn)

    def _create_widget(self, name: str) -> QWidget | None:
        """Create the appropriate widget for a config field."""
        for f in fields(DetectionConfig):
            if f.name != name:
                continue
            value = getattr(self._config, name)
            if f.type == "float" or isinstance(value, float):
                w = QDoubleSpinBox()
                w.setRange(-1000, 10000)
                w.setDecimals(3)
                w.setValue(value)
                return w
            elif f.type == "int" or isinstance(value, int):
                w = QSpinBox()
                w.setRange(-1, 100000)
                w.setValue(value)
                return w
            elif f.type == "bool" or isinstance(value, bool):
                w = QCheckBox()
                w.setChecked(value)
                return w
        return None

    def _apply_preset(self, preset_name: str):
        preset = PRESETS.get(preset_name)
        if not preset:
            return
        self._config = preset
        for name, widget in self._widgets.items():
            value = getattr(self._config, name, None)
            if value is None:
                continue
            if isinstance(widget, QDoubleSpinBox):
                widget.setValue(value)
            elif isinstance(widget, QSpinBox):
                widget.setValue(value)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(value)

    def _emit_config(self):
        config = self.get_config()
        self.config_changed.emit(config)

    def get_config(self) -> DetectionConfig:
        """Read current values from all widgets into a DetectionConfig."""
        kwargs = {}
        for name, widget in self._widgets.items():
            if isinstance(widget, QDoubleSpinBox):
                kwargs[name] = widget.value()
            elif isinstance(widget, QSpinBox):
                kwargs[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                kwargs[name] = widget.isChecked()
        return DetectionConfig(**kwargs)
