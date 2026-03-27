"""Job manager panel — analysis queue with progress, cancel, and log output."""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class JobManagerWidget(QWidget):
    """Panel showing running/completed analysis jobs with progress and logs."""

    cancel_requested = Signal(str)  # job_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._jobs: dict[str, QTreeWidgetItem] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Current job progress
        progress_row = QHBoxLayout()
        self._progress_label = QLabel("Idle")
        progress_row.addWidget(self._progress_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        progress_row.addWidget(self._progress_bar)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        progress_row.addWidget(self._cancel_btn)
        layout.addLayout(progress_row)

        # Splitter: job list + log
        splitter = QSplitter(Qt.Vertical)

        # Job history
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Job", "Status", "Started", "Duration", "Results"])
        self._tree.setColumnWidth(0, 150)
        self._tree.setColumnWidth(1, 80)
        splitter.addWidget(self._tree)

        # Log output
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        self._log.setPlaceholderText("Analysis log output...")
        splitter.addWidget(self._log)

        splitter.setSizes([200, 150])
        layout.addWidget(splitter)

    def add_job(self, job_id: str, description: str) -> None:
        """Register a new job in the list."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        item = QTreeWidgetItem([description, "Running", timestamp, "—", "—"])
        self._tree.insertTopLevelItem(0, item)
        self._jobs[job_id] = item
        self._progress_label.setText(description)
        self._progress_bar.setValue(0)
        self._cancel_btn.setEnabled(True)
        self._current_job_id = job_id
        self.log(f"Started: {description}")

    def update_progress(self, job_id: str, percent: int, message: str) -> None:
        """Update progress for a running job."""
        self._progress_bar.setValue(percent)
        self._progress_label.setText(message)
        item = self._jobs.get(job_id)
        if item:
            item.setText(1, f"{percent}%")

    def complete_job(self, job_id: str, summary: str) -> None:
        """Mark a job as completed."""
        item = self._jobs.get(job_id)
        if item:
            item.setText(1, "Done")
            item.setText(4, summary)
        self._progress_bar.setValue(100)
        self._progress_label.setText("Complete")
        self._cancel_btn.setEnabled(False)
        self.log(f"Completed: {summary}")

    def fail_job(self, job_id: str, error: str) -> None:
        """Mark a job as failed."""
        item = self._jobs.get(job_id)
        if item:
            item.setText(1, "Failed")
            item.setText(4, error[:50])
        self._progress_bar.setValue(0)
        self._progress_label.setText("Failed")
        self._cancel_btn.setEnabled(False)
        self.log(f"ERROR: {error}")

    def log(self, message: str) -> None:
        """Append a line to the log output."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log.appendPlainText(f"[{timestamp}] {message}")

    def _on_cancel(self):
        if hasattr(self, "_current_job_id"):
            self.cancel_requested.emit(self._current_job_id)
            self._cancel_btn.setEnabled(False)
            self._progress_label.setText("Cancelling...")
