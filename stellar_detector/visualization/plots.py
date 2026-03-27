"""Matplotlib-based visualizations for stellar analysis results.

All plot functions return a matplotlib Figure so they can be displayed
in the GUI or saved to files.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ..core.models import AnomalyResult


def plot_hr_diagram(
    df: pd.DataFrame,
    anomalies: list[AnomalyResult] | None = None,
    title: str = "Hertzsprung-Russell Diagram",
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """Plot an HR diagram (Color-Magnitude Diagram) with anomalies highlighted."""
    fig, ax = plt.subplots(figsize=figsize)

    if "bp_rp" not in df.columns or "abs_mag" not in df.columns:
        ax.text(0.5, 0.5, "Missing bp_rp or abs_mag columns", transform=ax.transAxes,
                ha="center", va="center")
        return fig

    # Background: all stars
    clean = df.dropna(subset=["bp_rp", "abs_mag"])
    ax.scatter(
        clean["bp_rp"], clean["abs_mag"],
        s=1, c="gray", alpha=0.3, label=f"All stars (n={len(clean)})",
    )

    # Overlay anomalies
    if anomalies:
        anomaly_ids = {r.star_id for r in anomalies}
        id_col = "source_id" if "source_id" in df.columns else df.index.name or "index"
        if id_col == "index":
            anom_mask = df.index.astype(str).isin(anomaly_ids)
        else:
            anom_mask = df[id_col].astype(str).isin(anomaly_ids)

        anom_df = df[anom_mask].dropna(subset=["bp_rp", "abs_mag"])
        ax.scatter(
            anom_df["bp_rp"], anom_df["abs_mag"],
            s=20, c="red", alpha=0.8, edgecolors="darkred", linewidths=0.5,
            label=f"Anomalies (n={len(anom_df)})", zorder=5,
        )

    ax.set_xlabel("BP - RP (mag)")
    ax.set_ylabel("Absolute G Magnitude")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_sky_map(
    df: pd.DataFrame,
    anomalies: list[AnomalyResult] | None = None,
    projection: str = "mollweide",
    title: str = "Sky Distribution",
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Plot stars on a sky projection (Mollweide or Aitoff)."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=projection)

    if "ra" not in df.columns or "dec" not in df.columns:
        return fig

    # Convert RA/Dec to radians for projection (RA: 0-360 -> -pi to pi)
    ra_rad = np.deg2rad(df["ra"].values - 180)
    dec_rad = np.deg2rad(df["dec"].values)

    ax.scatter(ra_rad, dec_rad, s=0.5, c="steelblue", alpha=0.3)

    if anomalies:
        anomaly_ids = {r.star_id for r in anomalies}
        id_col = "source_id" if "source_id" in df.columns else None
        if id_col:
            mask = df[id_col].astype(str).isin(anomaly_ids)
            anom = df[mask]
            ra_anom = np.deg2rad(anom["ra"].values - 180)
            dec_anom = np.deg2rad(anom["dec"].values)
            ax.scatter(ra_anom, dec_anom, s=15, c="red", alpha=0.8, zorder=5)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_anomaly_dashboard(
    df: pd.DataFrame,
    anomalies: list[AnomalyResult],
    figsize: tuple[float, float] = (16, 12),
) -> Figure:
    """Multi-panel dashboard summarizing analysis results."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Panel 1: HR diagram
    ax = axes[0, 0]
    if "bp_rp" in df.columns and "abs_mag" in df.columns:
        clean = df.dropna(subset=["bp_rp", "abs_mag"])
        ax.scatter(clean["bp_rp"], clean["abs_mag"], s=1, c="gray", alpha=0.3)
        anomaly_ids = {r.star_id for r in anomalies}
        if "source_id" in df.columns:
            mask = df["source_id"].astype(str).isin(anomaly_ids)
            anom = df[mask].dropna(subset=["bp_rp", "abs_mag"])
            ax.scatter(anom["bp_rp"], anom["abs_mag"], s=10, c="red", alpha=0.7)
        ax.invert_yaxis()
    ax.set_title("HR Diagram")
    ax.set_xlabel("BP-RP")
    ax.set_ylabel("Abs Mag")

    # Panel 2: Anomaly type distribution
    ax = axes[0, 1]
    type_counts = Counter(r.anomaly_type.anomaly_name for r in anomalies)
    if type_counts:
        names = list(type_counts.keys())
        counts = list(type_counts.values())
        ax.barh(range(len(names)), counts, color="steelblue")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n.replace("_", " ").title()[:20] for n in names], fontsize=7)
    ax.set_title("Anomaly Types")
    ax.set_xlabel("Count")

    # Panel 3: Confidence distribution
    ax = axes[0, 2]
    if anomalies:
        confidences = [r.confidence for r in anomalies]
        ax.hist(confidences, bins=20, color="steelblue", edgecolor="white")
    ax.set_title("Confidence Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")

    # Panel 4: Proper motion
    ax = axes[1, 0]
    if "pmra" in df.columns and "pmdec" in df.columns:
        clean = df.dropna(subset=["pmra", "pmdec"])
        ax.scatter(clean["pmra"], clean["pmdec"], s=1, c="gray", alpha=0.3)
        if "source_id" in df.columns:
            mask = df["source_id"].astype(str).isin(anomaly_ids)
            anom = df[mask].dropna(subset=["pmra", "pmdec"])
            ax.scatter(anom["pmra"], anom["pmdec"], s=10, c="red", alpha=0.7)
    ax.set_title("Proper Motion")
    ax.set_xlabel("pmRA (mas/yr)")
    ax.set_ylabel("pmDec (mas/yr)")

    # Panel 5: Distance distribution
    ax = axes[1, 1]
    if "distance_pc" in df.columns:
        dist = df["distance_pc"].dropna()
        ax.hist(dist.clip(0, dist.quantile(0.99)), bins=50, color="steelblue", edgecolor="white")
    ax.set_title("Distance Distribution")
    ax.set_xlabel("Distance (pc)")

    # Panel 6: Priority distribution
    ax = axes[1, 2]
    if anomalies:
        priorities = [r.follow_up_priority for r in anomalies]
        ax.hist(priorities, bins=range(1, 12), color="coral", edgecolor="white")
    ax.set_title("Follow-up Priority")
    ax.set_xlabel("Priority (1-10)")
    ax.set_ylabel("Count")

    fig.suptitle(
        f"Stellar Anomaly Detection Dashboard — {len(anomalies)} anomalies in {len(df)} stars",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_anomaly_distribution(
    anomalies: list[AnomalyResult],
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Bar chart of anomaly counts by type with significance coloring."""
    fig, ax = plt.subplots(figsize=figsize)

    if not anomalies:
        ax.text(0.5, 0.5, "No anomalies to display", transform=ax.transAxes,
                ha="center", va="center")
        return fig

    type_data: dict[str, list[float]] = {}
    for r in anomalies:
        name = r.anomaly_type.anomaly_name.replace("_", " ").title()
        type_data.setdefault(name, []).append(r.significance_score)

    names = sorted(type_data.keys(), key=lambda n: len(type_data[n]), reverse=True)
    counts = [len(type_data[n]) for n in names]
    mean_sig = [np.mean(type_data[n]) for n in names]

    colors = plt.cm.YlOrRd(np.array(mean_sig) / max(mean_sig) if max(mean_sig) > 0 else [0.5])
    ax.barh(range(len(names)), counts, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Number of Detections")
    ax.set_title("Anomaly Type Distribution")

    fig.tight_layout()
    return fig
