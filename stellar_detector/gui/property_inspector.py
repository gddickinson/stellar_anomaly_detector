"""Property inspector — side panel showing all details for a selected star."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..core.constants import ANOMALY_DISPLAY_NAMES


class PropertyInspectorWidget(QWidget):
    """Display all properties and anomaly flags for a selected star."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Star Properties")
        title.setProperty("heading", True)
        outer.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._content = QWidget()
        self._layout = QVBoxLayout(self._content)
        self._layout.setAlignment(Qt.AlignTop)
        self._placeholder = QLabel("Select a star to view properties")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet("color: gray; padding: 20px;")
        self._layout.addWidget(self._placeholder)

        scroll.setWidget(self._content)
        outer.addWidget(scroll)

    def show_star(self, row_data: dict[str, Any], anomalies: list | None = None):
        """Populate the inspector with data for a selected star."""
        self._clear()

        # Identification
        id_group = self._add_group("Identification")
        for key in ["source_id", "catalog_source"]:
            if key in row_data:
                self._add_property(id_group, key, row_data[key])

        # Position
        pos_group = self._add_group("Position")
        for key in ["ra", "dec", "parallax", "parallax_error", "distance_pc"]:
            if key in row_data:
                self._add_property(pos_group, key, row_data[key])

        # Photometry
        phot_group = self._add_group("Photometry")
        for key in [
            "phot_g_mean_mag", "bp_rp", "abs_mag", "luminosity_solar",
            "Jmag", "Hmag", "Kmag", "W1mag", "W2mag", "W3mag", "W4mag",
            "J_H", "H_K", "W1_W2", "W2_W3",
        ]:
            if key in row_data:
                self._add_property(phot_group, key, row_data[key])

        # Kinematics
        kin_group = self._add_group("Kinematics")
        for key in ["pmra", "pmdec", "pm_total", "v_tan_km_s", "radial_velocity"]:
            if key in row_data:
                self._add_property(kin_group, key, row_data[key])

        # Stellar parameters
        stellar_group = self._add_group("Stellar Parameters")
        for key in [
            "teff_gspphot", "logg_gspphot", "mh_gspphot",
            "ruwe", "astrometric_excess_noise", "astrometric_excess_noise_sig",
        ]:
            if key in row_data:
                self._add_property(stellar_group, key, row_data[key])

        # Quality
        qual_group = self._add_group("Quality")
        for key in ["quality_score", "phot_bp_rp_excess_factor", "phot_g_mean_flux_over_error"]:
            if key in row_data:
                self._add_property(qual_group, key, row_data[key])

        # Anomalies
        if anomalies:
            anom_group = self._add_group(f"Anomalies ({len(anomalies)})")
            for anom in anomalies:
                display_name = ANOMALY_DISPLAY_NAMES.get(
                    anom.anomaly_type.anomaly_name, anom.anomaly_type.anomaly_name
                )
                text = (
                    f"<b>{display_name}</b> "
                    f"(confidence: {anom.confidence:.2f}, priority: {anom.follow_up_priority})<br>"
                    f"<i>{anom.description[:100]}</i><br>"
                    f"Method: {anom.detection_method}"
                )
                label = QLabel(text)
                label.setWordWrap(True)
                label.setTextFormat(Qt.RichText)
                label.setStyleSheet("padding: 4px; margin: 2px 0;")
                anom_group.layout().addWidget(label)

        self._layout.addStretch()

    def clear(self):
        self._clear()
        self._placeholder = QLabel("Select a star to view properties")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet("color: gray; padding: 20px;")
        self._layout.addWidget(self._placeholder)

    def _clear(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def _add_group(self, title: str) -> QGroupBox:
        group = QGroupBox(title)
        group.setLayout(QVBoxLayout())
        self._layout.addWidget(group)
        return group

    @staticmethod
    def _add_property(group: QGroupBox, key: str, value: Any):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            display = "—"
        elif isinstance(value, float):
            display = f"{value:.4g}"
        else:
            display = str(value)

        label = QLabel(f"<b>{key}:</b> {display}")
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        group.layout().addWidget(label)
