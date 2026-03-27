"""Annotation system — tag, comment, classify anomalies."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AnnotationStatus:
    UNREVIEWED = "unreviewed"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    NEEDS_FOLLOWUP = "needs_followup"
    INTERESTING = "interesting"


class AnnotationStore:
    """In-memory annotation store with JSON persistence.

    Each annotation is keyed by star_id and contains:
    - status: one of AnnotationStatus values
    - tags: list of user-defined tags
    - comment: free-text note
    - timestamp: when last updated
    """

    def __init__(self):
        self._annotations: dict[str, dict[str, Any]] = {}

    def annotate(
        self,
        star_id: str,
        status: str = AnnotationStatus.UNREVIEWED,
        tags: list[str] | None = None,
        comment: str = "",
    ):
        """Add or update an annotation for a star."""
        self._annotations[star_id] = {
            "status": status,
            "tags": tags or [],
            "comment": comment,
            "timestamp": datetime.now().isoformat(),
        }

    def get(self, star_id: str) -> dict[str, Any] | None:
        return self._annotations.get(star_id)

    def remove(self, star_id: str):
        self._annotations.pop(star_id, None)

    def filter_by_status(self, status: str) -> list[str]:
        """Return star_ids matching a given status."""
        return [sid for sid, a in self._annotations.items() if a["status"] == status]

    def filter_by_tag(self, tag: str) -> list[str]:
        return [sid for sid, a in self._annotations.items() if tag in a.get("tags", [])]

    def all_tags(self) -> set[str]:
        """Return all unique tags across all annotations."""
        tags = set()
        for a in self._annotations.values():
            tags.update(a.get("tags", []))
        return tags

    @property
    def count(self) -> int:
        return len(self._annotations)

    @property
    def annotations(self) -> dict[str, dict[str, Any]]:
        return self._annotations

    def save(self, filepath: str):
        """Save annotations to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._annotations, indent=2))
        logger.info("Saved %d annotations to %s", len(self._annotations), filepath)

    def load(self, filepath: str):
        """Load annotations from JSON file."""
        path = Path(filepath)
        if path.exists():
            self._annotations = json.loads(path.read_text())
            logger.info("Loaded %d annotations from %s", len(self._annotations), filepath)

    def summary(self) -> dict[str, int]:
        """Count annotations by status."""
        counts: dict[str, int] = {}
        for a in self._annotations.values():
            status = a.get("status", AnnotationStatus.UNREVIEWED)
            counts[status] = counts.get(status, 0) + 1
        return counts
