"""File I/O utilities: CSV, VOTable, and report export."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..core.models import AnomalyResult

logger = logging.getLogger(__name__)


def save_results(
    results: list[AnomalyResult],
    output_dir: str,
    filename: str = "anomalies",
    fmt: str = "csv",
) -> Path:
    """Save anomaly results to file.

    Args:
        results: List of AnomalyResult objects.
        output_dir: Directory to write to.
        filename: Base filename (without extension).
        fmt: Export format — "csv", "json", or "votable".

    Returns:
        Path to the written file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = [r.to_dict() for r in results]
    df = pd.DataFrame(records)

    if fmt == "csv":
        path = out / f"{filename}.csv"
        df.to_csv(path, index=False)
    elif fmt == "json":
        path = out / f"{filename}.json"
        with open(path, "w") as f:
            json.dump(records, f, indent=2, default=str)
    elif fmt == "votable":
        path = out / f"{filename}.xml"
        from astropy.table import Table
        table = Table.from_pandas(df)
        table.write(str(path), format="votable", overwrite=True)
    else:
        raise ValueError(f"Unsupported export format: {fmt}")

    logger.info("Saved %d results to %s", len(results), path)
    return path


def load_catalog_csv(filepath: str) -> pd.DataFrame:
    """Load a catalog CSV file into a DataFrame."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Catalog file not found: {filepath}")
    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows from %s", len(df), filepath)
    return df


def export_report(
    results: list[AnomalyResult],
    df: pd.DataFrame,
    output_dir: str,
    title: str = "Stellar Anomaly Detection Report",
) -> Path:
    """Generate an HTML analysis report with embedded statistics."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from ..analysis.ensemble import EnsembleScorer
    scorer = EnsembleScorer()
    stats = scorer.summary_stats(results)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_parts = [
        "<!DOCTYPE html><html><head>",
        f"<title>{title}</title>",
        "<style>",
        "body { font-family: 'Segoe UI', sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }",
        "h1 { color: #1a237e; } h2 { color: #283593; }",
        "table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background: #e8eaf6; } tr:nth-child(even) { background: #f5f5f5; }",
        ".stat { display: inline-block; background: #e3f2fd; padding: 10px 20px; margin: 5px; border-radius: 5px; }",
        ".stat-value { font-size: 24px; font-weight: bold; color: #1565c0; }",
        "</style></head><body>",
        f"<h1>{title}</h1>",
        f"<p>Generated: {timestamp}</p>",
        "<h2>Summary</h2>",
        "<div>",
        f'<div class="stat"><div class="stat-value">{len(df)}</div>Stars Analyzed</div>',
        f'<div class="stat"><div class="stat-value">{stats["total"]}</div>Anomalies Found</div>',
        f'<div class="stat"><div class="stat-value">{stats.get("unique_stars", 0)}</div>Unique Stars</div>',
        f'<div class="stat"><div class="stat-value">{stats.get("high_priority_count", 0)}</div>High Priority</div>',
        "</div>",
    ]

    # Top anomalies table
    if results:
        top = sorted(results, key=lambda r: r.follow_up_priority, reverse=True)[:20]
        html_parts.append("<h2>Top Anomalies</h2><table>")
        html_parts.append(
            "<tr><th>Star ID</th><th>Type</th><th>Confidence</th>"
            "<th>Priority</th><th>Method</th><th>Description</th></tr>"
        )
        for r in top:
            html_parts.append(
                f"<tr><td>{r.star_id}</td><td>{r.anomaly_type.anomaly_name}</td>"
                f"<td>{r.confidence:.2f}</td><td>{r.follow_up_priority}</td>"
                f"<td>{r.detection_method}</td><td>{r.description[:80]}</td></tr>"
            )
        html_parts.append("</table>")

    # Type breakdown
    if stats.get("by_type"):
        html_parts.append("<h2>Anomaly Types</h2><table>")
        html_parts.append("<tr><th>Type</th><th>Count</th></tr>")
        for t, c in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
            html_parts.append(f"<tr><td>{t}</td><td>{c}</td></tr>")
        html_parts.append("</table>")

    html_parts.append("</body></html>")

    path = out / "analysis_report.html"
    path.write_text("\n".join(html_parts))
    logger.info("Report saved to %s", path)
    return path
