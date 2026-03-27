"""Catalog browser panel — tree view of available catalogs with fetch controls."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.constants import CATALOG_METADATA
from ..core.models import CatalogSource

_SOURCE_ITEMS = [
    (CatalogSource.GAIA_DR3, "Gaia DR3", "~1.8 billion sources"),
    (CatalogSource.HIPPARCOS, "Hipparcos-2", "~118k stars"),
    (CatalogSource.TYCHO2, "Tycho-2", "~2.5 million stars"),
    (CatalogSource.TWOMASS, "2MASS", "~470 million sources"),
    (CatalogSource.ALLWISE, "AllWISE", "~747 million sources"),
    (CatalogSource.SYNTHETIC, "Synthetic (test)", "Generated with injected anomalies"),
]


class CatalogBrowserWidget(QWidget):
    """Panel for selecting and fetching astronomical catalog data."""

    fetch_requested = Signal(CatalogSource, int, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._catalog_status: dict[str, str] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Catalog Browser")
        title.setProperty("heading", True)
        layout.addWidget(title)

        # Catalog tree
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Catalog", "Status", "Description"])
        self._tree.setColumnWidth(0, 150)
        self._tree.setColumnWidth(1, 80)

        for source, name, desc in _SOURCE_ITEMS:
            item = QTreeWidgetItem([name, "Not loaded", desc])
            item.setData(0, Qt.UserRole, source)
            self._tree.addTopLevelItem(item)
        layout.addWidget(self._tree)

        # Fetch parameters
        params = QGroupBox("Fetch Parameters")
        params_layout = QVBoxLayout(params)

        # Source selector
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source:"))
        self._source_combo = QComboBox()
        for source, name, _ in _SOURCE_ITEMS:
            self._source_combo.addItem(name, source)
        src_row.addWidget(self._source_combo)
        params_layout.addLayout(src_row)

        # Star count
        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Stars:"))
        self._star_count = QSpinBox()
        self._star_count.setRange(100, 50000)
        self._star_count.setValue(2000)
        self._star_count.setSingleStep(500)
        count_row.addWidget(self._star_count)
        params_layout.addLayout(count_row)

        # Coordinates
        coord_row = QHBoxLayout()
        coord_row.addWidget(QLabel("RA:"))
        self._ra = QDoubleSpinBox()
        self._ra.setRange(0, 360)
        self._ra.setValue(180.0)
        self._ra.setDecimals(2)
        coord_row.addWidget(self._ra)
        coord_row.addWidget(QLabel("Dec:"))
        self._dec = QDoubleSpinBox()
        self._dec.setRange(-90, 90)
        self._dec.setValue(0.0)
        self._dec.setDecimals(2)
        coord_row.addWidget(self._dec)
        params_layout.addLayout(coord_row)

        # Radius
        rad_row = QHBoxLayout()
        rad_row.addWidget(QLabel("Radius (deg):"))
        self._radius = QDoubleSpinBox()
        self._radius.setRange(0.1, 30.0)
        self._radius.setValue(5.0)
        self._radius.setDecimals(1)
        rad_row.addWidget(self._radius)
        params_layout.addLayout(rad_row)

        layout.addWidget(params)

        # Fetch button
        self._fetch_btn = QPushButton("Fetch Data")
        self._fetch_btn.setProperty("primary", True)
        self._fetch_btn.clicked.connect(self._on_fetch)
        layout.addWidget(self._fetch_btn)

        layout.addStretch()

    def _on_fetch(self):
        source = self._source_combo.currentData()
        self.fetch_requested.emit(
            source,
            self._star_count.value(),
            self._ra.value(),
            self._dec.value(),
            self._radius.value(),
        )

    def update_catalog_status(self, source: CatalogSource, status: str, row_count: int = 0):
        """Update the status display for a catalog in the tree."""
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == source:
                if row_count > 0:
                    item.setText(1, f"{row_count} rows")
                else:
                    item.setText(1, status)
                break

    def set_fetching(self, is_fetching: bool):
        """Enable/disable the fetch button."""
        self._fetch_btn.setEnabled(not is_fetching)
        self._fetch_btn.setText("Fetching..." if is_fetching else "Fetch Data")
