"""Project/session save and restore — workspace state serialization."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.models import AnomalyResult, AnomalyType, DetectionConfig

logger = logging.getLogger(__name__)


class Session:
    """Serializable workspace state: loaded data, config, results, annotations."""

    def __init__(self):
        self.name: str = ""
        self.created: str = datetime.now().isoformat()
        self.config: DetectionConfig = DetectionConfig()
        self.data_path: str = ""
        self.catalog_source: str = ""
        self.n_stars: int = 0
        self.annotations: dict[str, dict[str, Any]] = {}  # star_id -> annotation
        self.result_summary: dict = {}

    def save(self, directory: str, df: pd.DataFrame | None = None, results: list[AnomalyResult] | None = None):
        """Save session to a directory (JSON metadata + CSV data)."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        # Save data
        if df is not None and not df.empty:
            data_path = out / "catalog_data.csv"
            df.to_csv(data_path, index=False)
            self.data_path = str(data_path)
            self.n_stars = len(df)

        # Save results
        if results:
            results_records = [r.to_dict() for r in results]
            (out / "results.json").write_text(json.dumps(results_records, indent=2, default=str))

        # Save session metadata
        meta = {
            "name": self.name,
            "created": self.created,
            "saved": datetime.now().isoformat(),
            "catalog_source": self.catalog_source,
            "n_stars": self.n_stars,
            "config": _config_to_dict(self.config),
            "annotations": self.annotations,
            "result_summary": self.result_summary,
        }
        (out / "session.json").write_text(json.dumps(meta, indent=2, default=str))
        logger.info("Session saved to %s", out)

    @classmethod
    def load(cls, directory: str) -> tuple["Session", pd.DataFrame | None, list[dict]]:
        """Load a session from a directory.

        Returns (session, dataframe_or_none, results_dicts).
        """
        out = Path(directory)
        session = cls()

        # Load metadata
        meta_path = out / "session.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            session.name = meta.get("name", "")
            session.created = meta.get("created", "")
            session.catalog_source = meta.get("catalog_source", "")
            session.n_stars = meta.get("n_stars", 0)
            session.annotations = meta.get("annotations", {})
            session.result_summary = meta.get("result_summary", {})
            if "config" in meta:
                session.config = _config_from_dict(meta["config"])

        # Load data
        df = None
        data_path = out / "catalog_data.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)

        # Load results
        results = []
        results_path = out / "results.json"
        if results_path.exists():
            results = json.loads(results_path.read_text())

        logger.info("Session loaded from %s: %d stars", out, session.n_stars)
        return session, df, results


def _config_to_dict(config: DetectionConfig) -> dict:
    from dataclasses import asdict
    return asdict(config)


def _config_from_dict(d: dict) -> DetectionConfig:
    from dataclasses import fields
    valid = {f.name for f in fields(DetectionConfig)}
    return DetectionConfig(**{k: v for k, v in d.items() if k in valid})
