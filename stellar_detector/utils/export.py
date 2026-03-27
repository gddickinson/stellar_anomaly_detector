"""Advanced export pipeline: VOTable, FITS, LaTeX tables, publication-ready plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ..core.models import AnomalyResult

logger = logging.getLogger(__name__)


def export_votable(df: pd.DataFrame, filepath: str):
    """Export catalog data as a VOTable (Virtual Observatory standard)."""
    from astropy.table import Table

    table = Table.from_pandas(df)
    table.write(filepath, format="votable", overwrite=True)
    logger.info("VOTable exported: %s (%d rows)", filepath, len(df))


def export_fits(df: pd.DataFrame, filepath: str):
    """Export catalog data as a FITS binary table."""
    from astropy.table import Table

    table = Table.from_pandas(df)
    table.write(filepath, format="fits", overwrite=True)
    logger.info("FITS exported: %s (%d rows)", filepath, len(df))


def export_latex_table(
    results: list[AnomalyResult],
    filepath: str,
    max_rows: int = 30,
    caption: str = "Anomaly Detection Results",
):
    """Export top anomaly results as a LaTeX table for publications."""
    if not results:
        return

    sorted_results = sorted(results, key=lambda r: r.follow_up_priority, reverse=True)
    rows = sorted_results[:max_rows]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        r"\begin{tabular}{llrrll}",
        r"\hline",
        r"Star ID & Type & Conf. & Priority & Method & Description \\",
        r"\hline",
    ]

    for r in rows:
        star = _latex_escape(str(r.star_id)[:15])
        atype = _latex_escape(r.anomaly_type.anomaly_name.replace("_", " ")[:18])
        desc = _latex_escape(r.description[:40])
        method = _latex_escape(r.detection_method[:15])
        lines.append(
            f"{star} & {atype} & {r.confidence:.2f} & {r.follow_up_priority} "
            f"& {method} & {desc} \\\\"
        )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    Path(filepath).write_text("\n".join(lines))
    logger.info("LaTeX table exported: %s (%d rows)", filepath, len(rows))


def export_publication_plot(fig: Figure, filepath: str, dpi: int = 300):
    """Save a matplotlib figure in publication-ready format (SVG or PDF)."""
    path = Path(filepath)
    fmt = path.suffix.lstrip(".")
    if fmt not in ("svg", "pdf", "png", "eps"):
        fmt = "pdf"
        path = path.with_suffix(".pdf")

    fig.savefig(str(path), format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    logger.info("Publication plot exported: %s (format=%s, dpi=%d)", path, fmt, dpi)


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters."""
    for char in ["&", "%", "$", "#", "_", "{", "}"]:
        text = text.replace(char, f"\\{char}")
    return text
