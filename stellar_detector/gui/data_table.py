"""QTableView with pandas model — virtual scrolling, sorting, filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTableView,
    QVBoxLayout,
    QWidget,
)


class PandasTableModel(QAbstractTableModel):
    """High-performance table model backed by a pandas DataFrame.

    Uses Qt's model/view architecture for efficient display of large datasets
    without loading all data into widgets.
    """

    def __init__(self, df: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    def set_dataframe(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            value = self._df.iloc[index.row(), index.column()]
            if isinstance(value, float):
                if np.isnan(value):
                    return ""
                return f"{value:.4g}"
            return str(value)
        if role == Qt.TextAlignmentRole:
            value = self._df.iloc[index.row(), index.column()]
            if isinstance(value, (int, float, np.integer, np.floating)):
                return int(Qt.AlignRight | Qt.AlignVCenter)
            return int(Qt.AlignLeft | Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)

    def get_row_data(self, row: int) -> dict:
        """Return all column values for a given row as a dict."""
        if 0 <= row < len(self._df):
            return self._df.iloc[row].to_dict()
        return {}

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df


class DataTableWidget(QWidget):
    """Complete data table widget with filter bar and row count display."""

    row_selected = Signal(int, dict)  # (row_index, row_data)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = PandasTableModel()
        self._proxy = QSortFilterProxyModel()
        self._proxy.setSourceModel(self._model)
        self._proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self._proxy.setFilterKeyColumn(-1)  # search all columns
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Filter bar
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self._filter_input = QLineEdit()
        self._filter_input.setPlaceholderText("Type to filter across all columns...")
        self._filter_input.textChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self._filter_input)
        self._count_label = QLabel("0 rows")
        filter_row.addWidget(self._count_label)
        layout.addLayout(filter_row)

        # Table view
        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QTableView.SelectRows)
        self._table.setSelectionMode(QTableView.SingleSelection)
        self._table.verticalHeader().setDefaultSectionSize(24)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.clicked.connect(self._on_row_clicked)
        layout.addWidget(self._table)

    def set_dataframe(self, df: pd.DataFrame):
        self._model.set_dataframe(df)
        self._update_count()
        self._table.resizeColumnsToContents()

    def _on_filter_changed(self, text: str):
        self._proxy.setFilterFixedString(text)
        self._update_count()

    def _on_row_clicked(self, index: QModelIndex):
        source_index = self._proxy.mapToSource(index)
        row_data = self._model.get_row_data(source_index.row())
        self.row_selected.emit(source_index.row(), row_data)

    def _update_count(self):
        total = self._model.rowCount()
        shown = self._proxy.rowCount()
        if shown == total:
            self._count_label.setText(f"{total} rows")
        else:
            self._count_label.setText(f"{shown} / {total} rows")

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._model.dataframe
