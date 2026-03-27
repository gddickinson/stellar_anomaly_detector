"""Dimensionality reduction for feature space visualization.

Provides t-SNE and UMAP embeddings of variability/photometric feature spaces,
plus SHAP-based feature importance for anomaly explanation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..core.models import AnomalyResult, DetectionConfig

logger = logging.getLogger(__name__)


def compute_tsne_embedding(
    df: pd.DataFrame,
    feature_columns: list[str],
    perplexity: float = 70.0,
    n_components: int = 2,
) -> np.ndarray | None:
    """Compute t-SNE embedding of feature space.

    Uses openTSNE if available (faster, better defaults), falls back to sklearn.
    Returns array of shape (n_samples, n_components) or None on failure.
    """
    clean = df[feature_columns].dropna()
    if len(clean) < 30:
        logger.warning("Too few samples (%d) for t-SNE", len(clean))
        return None

    X = StandardScaler().fit_transform(clean.values)
    perp = min(perplexity, (len(X) - 1) / 3)

    try:
        from openTSNE import TSNE
        tsne = TSNE(n_components=n_components, perplexity=perp, n_jobs=-1, random_state=42)
        embedding = tsne.fit(X)
        logger.info("openTSNE embedding computed: %s", embedding.shape)
        return np.array(embedding)
    except ImportError:
        pass

    from sklearn.manifold import TSNE as SkTSNE
    tsne = SkTSNE(n_components=n_components, perplexity=perp, random_state=42, n_iter=1000)
    embedding = tsne.fit_transform(X)
    logger.info("sklearn t-SNE embedding computed: %s", embedding.shape)
    return embedding


def compute_umap_embedding(
    df: pd.DataFrame,
    feature_columns: list[str],
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    n_components: int = 2,
) -> np.ndarray | None:
    """Compute UMAP embedding of feature space."""
    try:
        import umap
    except ImportError:
        logger.info("umap-learn not installed — skipping UMAP")
        return None

    clean = df[feature_columns].dropna()
    if len(clean) < 30:
        return None

    X = StandardScaler().fit_transform(clean.values)
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, len(X) - 1),
        min_dist=min_dist,
        n_components=n_components,
        random_state=42,
    )
    embedding = reducer.fit_transform(X)
    logger.info("UMAP embedding computed: %s", embedding.shape)
    return embedding


def explain_anomalies_shap(
    df: pd.DataFrame,
    feature_columns: list[str],
    results: list[AnomalyResult],
    config: DetectionConfig | None = None,
) -> dict[str, dict[str, float]]:
    """Compute SHAP feature contributions for each anomalous star.

    Returns dict mapping star_id -> {feature_name: shap_value}.
    Uses an Isolation Forest as the model to explain.
    """
    try:
        import shap
    except ImportError:
        logger.info("shap not installed — skipping explanations")
        return {}

    config = config or DetectionConfig()
    clean = df[feature_columns + ["source_id"]].dropna(subset=feature_columns)
    if len(clean) < 30:
        return {}

    X = clean[feature_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(
        n_estimators=config.isolation_n_estimators,
        contamination=config.isolation_contamination,
        random_state=42,
    )
    iso.fit(X_scaled)

    # Sample background for SHAP
    background = X_scaled[np.random.choice(len(X_scaled), min(100, len(X_scaled)), replace=False)]
    explainer = shap.KernelExplainer(iso.score_samples, background)

    anomaly_ids = {r.star_id for r in results}
    explanations = {}

    for idx, row in clean.iterrows():
        star_id = str(row["source_id"])
        if star_id not in anomaly_ids:
            continue

        x_single = X_scaled[clean.index.get_loc(idx)].reshape(1, -1)
        shap_values = explainer.shap_values(x_single, nsamples=100)
        contributions = dict(zip(feature_columns, shap_values[0]))
        explanations[star_id] = contributions

    logger.info("SHAP explanations computed for %d stars", len(explanations))
    return explanations


def xgboost_chemical_classifier(
    df: pd.DataFrame,
    label_column: str = "chemical_class",
    feature_columns: list[str] | None = None,
) -> tuple[object, dict]:
    """Train an XGBoost classifier for chemical peculiar star classification.

    Returns (trained model, metrics dict).
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("xgboost not installed")
        return None, {}

    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    if label_column not in df.columns:
        logger.info("No label column '%s' for XGBoost training", label_column)
        return None, {}

    if feature_columns is None:
        feature_columns = ["teff_gspphot", "logg_gspphot", "mh_gspphot"]
        feature_columns = [c for c in feature_columns if c in df.columns]

    clean = df.dropna(subset=feature_columns + [label_column])
    if len(clean) < 20:
        return None, {}

    X = clean[feature_columns].values
    le = LabelEncoder()
    y = le.fit_transform(clean[label_column])

    model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, use_label_encoder=False, eval_metric="mlogloss",
    )
    scores = cross_val_score(model, X, y, cv=min(5, len(clean) // 5), scoring="accuracy")
    model.fit(X, y)

    metrics = {
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "n_classes": len(le.classes_),
        "n_samples": len(clean),
    }
    logger.info("XGBoost classifier: accuracy=%.3f +/- %.3f", metrics["accuracy_mean"], metrics["accuracy_std"])
    return model, metrics
