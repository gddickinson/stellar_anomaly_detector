"""Result persistence using DuckDB (or SQLite fallback) for cross-session storage."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from ..core.models import AnomalyResult

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS anomaly_results (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL,
    run_timestamp TEXT NOT NULL,
    star_id TEXT NOT NULL,
    anomaly_type TEXT NOT NULL,
    confidence REAL,
    significance_score REAL,
    follow_up_priority INTEGER,
    detection_method TEXT,
    description TEXT,
    catalog_source TEXT,
    parameters TEXT,
    statistical_tests TEXT
)
"""

_INSERT = """
INSERT INTO anomaly_results
    (run_id, run_timestamp, star_id, anomaly_type, confidence,
     significance_score, follow_up_priority, detection_method,
     description, catalog_source, parameters, statistical_tests)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class ResultStore:
    """Persistent storage for anomaly detection results across sessions."""

    def __init__(self, db_path: str = "stellar_results.db"):
        self._path = Path(db_path)
        self._conn = None
        self._backend = "duckdb"
        self._connect()

    def _connect(self):
        try:
            import duckdb
            self._conn = duckdb.connect(str(self._path))
            self._backend = "duckdb"
        except ImportError:
            import sqlite3
            self._conn = sqlite3.connect(str(self._path))
            self._backend = "sqlite"

        self._conn.execute(_CREATE_TABLE)
        logger.info("ResultStore: %s backend at %s", self._backend, self._path)

    def save_run(self, run_id: str, results: list[AnomalyResult]) -> int:
        """Save a batch of results from an analysis run."""
        timestamp = datetime.now().isoformat()
        rows = []
        for r in results:
            rows.append((
                run_id, timestamp, r.star_id, r.anomaly_type.anomaly_name,
                r.confidence, r.significance_score, r.follow_up_priority,
                r.detection_method, r.description, r.catalog_source,
                json.dumps(r.parameters, default=str),
                json.dumps(r.statistical_tests, default=str),
            ))

        self._conn.executemany(_INSERT, rows)
        if self._backend == "sqlite":
            self._conn.commit()
        logger.info("Saved %d results for run %s", len(rows), run_id)
        return len(rows)

    def load_run(self, run_id: str) -> list[dict]:
        """Load all results from a specific run."""
        cursor = self._conn.execute(
            "SELECT * FROM anomaly_results WHERE run_id = ? ORDER BY significance_score DESC",
            (run_id,),
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def list_runs(self) -> list[dict]:
        """List all stored analysis runs with summary stats."""
        cursor = self._conn.execute("""
            SELECT run_id, run_timestamp,
                   COUNT(*) as n_results,
                   COUNT(DISTINCT star_id) as n_stars,
                   AVG(confidence) as avg_confidence
            FROM anomaly_results
            GROUP BY run_id, run_timestamp
            ORDER BY run_timestamp DESC
        """)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def query(self, sql: str) -> list[dict]:
        """Run an arbitrary SQL query against the results database."""
        cursor = self._conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def delete_run(self, run_id: str) -> int:
        """Delete all results from a specific run."""
        cursor = self._conn.execute(
            "DELETE FROM anomaly_results WHERE run_id = ?", (run_id,)
        )
        if self._backend == "sqlite":
            self._conn.commit()
        count = cursor.rowcount
        logger.info("Deleted %d results for run %s", count, run_id)
        return count

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
