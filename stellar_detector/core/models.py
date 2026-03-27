"""Core data models for the stellar anomaly detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AnomalyType(Enum):
    """Anomaly types with scientific significance ratings (1-10)."""

    HR_OUTLIER = ("hr_diagram_outlier", 8)
    METALLICITY = ("unusual_metallicity", 7)
    LIFETIME = ("lifetime_anomaly", 9)
    KINEMATICS = ("unusual_kinematics", 6)
    PHOTOMETRY = ("photometric_anomaly", 5)
    VARIABILITY = ("variability_anomaly", 7)
    ISOLATION = ("spatial_isolation", 6)
    CLUSTERING = ("unusual_clustering", 8)
    LUMINOSITY = ("luminosity_anomaly", 7)
    COLOR = ("color_anomaly", 5)
    ROTATION = ("rotation_anomaly", 6)
    MAGNETIC = ("magnetic_anomaly", 8)
    CHEMICAL = ("chemical_anomaly", 7)
    GEOMETRIC = ("geometric_pattern", 9)
    TEMPORAL = ("temporal_pattern", 8)
    DYSON_SPHERE = ("dyson_sphere_candidate", 10)
    STELLAR_ENGINE = ("stellar_engine_candidate", 9)
    MEGASTRUCTURE = ("megastructure_candidate", 9)
    BINARY_ANOMALY = ("binary_system_anomaly", 6)
    GALACTIC_ORBIT = ("galactic_orbit_anomaly", 7)
    CROSS_CATALOG = ("cross_catalog_anomaly", 9)
    INFRARED_EXCESS = ("infrared_excess", 8)
    ASTROMETRIC = ("astrometric_anomaly", 7)

    def __init__(self, anomaly_name: str, significance: int):
        self.anomaly_name = anomaly_name
        self.significance = significance


class CatalogSource(Enum):
    """Supported astronomical catalog sources."""

    GAIA_DR3 = "gaia_dr3"
    HIPPARCOS = "hipparcos"
    TYCHO2 = "tycho2"
    TWOMASS = "2mass"
    ALLWISE = "allwise"
    BRIGHT_STAR = "bright_star"
    VARIABLE_STAR = "variable_star"
    SYNTHETIC = "synthetic"


@dataclass
class AnomalyResult:
    """Result from a single anomaly detection on a single star."""

    star_id: str
    anomaly_type: AnomalyType
    confidence: float
    significance_score: float
    parameters: dict[str, Any]
    description: str
    follow_up_priority: int
    detection_method: str
    statistical_tests: dict[str, float]
    catalog_source: str = ""
    cross_validation_score: float | None = None
    literature_matches: list[str] = field(default_factory=list)
    observational_recommendations: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    feature_contributions: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for export."""
        return {
            "star_id": self.star_id,
            "anomaly_type": self.anomaly_type.anomaly_name,
            "confidence": self.confidence,
            "significance_score": self.significance_score,
            "follow_up_priority": self.follow_up_priority,
            "detection_method": self.detection_method,
            "description": self.description,
            "catalog_source": self.catalog_source,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DetectionConfig:
    """Configuration for all anomaly detection parameters."""

    # --- Statistical thresholds ---
    outlier_threshold: float = 4.0
    robust_outlier_threshold: float = 3.5
    mad_threshold: float = 4.0

    # --- Machine learning ---
    isolation_n_estimators: int = 200
    isolation_contamination: float = 0.02
    isolation_max_samples: int = 256
    lof_n_neighbors: int = 30
    lof_contamination: float = 0.03
    ocsvm_nu: float = 0.01
    ensemble_threshold: float = 0.7

    # --- Clustering ---
    dbscan_eps: float = 0.1
    dbscan_min_samples: int = 5
    hdbscan_min_cluster_size: int = 5
    gmm_n_components: int = 10

    # --- Spatial / geometric ---
    spatial_grid_size: int = 100
    spatial_significance: float = 3.0
    geometric_regularity: float = 0.8

    # --- Variability / temporal ---
    variability_threshold: float = 0.1
    period_range_min: float = 0.1
    period_range_max: float = 1000.0
    lomb_scargle_fit_mean: bool = True
    stetson_j_threshold: float = 1.0

    # --- Cross-matching ---
    cross_match_radius_arcsec: float = 5.0
    min_catalogs_for_anomaly: int = 2
    cross_match_confidence_sigma: float = 5.0

    # --- Data quality (Gaia DR3) ---
    min_parallax_over_error: float = 5.0
    max_ruwe: float = 1.4
    max_phot_bp_rp_excess: float = 1.5
    min_phot_bp_rp_excess: float = 0.8

    # --- Technosignature (Project Hephaistos) ---
    dyson_temp_min_k: float = 100.0
    dyson_temp_max_k: float = 700.0
    dyson_covering_min: float = 0.1
    dyson_covering_max: float = 0.9
    dyson_sed_rmse_max: float = 0.2
    dyson_snr_min: float = 3.5

    # --- Autoencoder (spectral) ---
    autoencoder_latent_dim: int = 10
    autoencoder_anomaly_percentile: float = 99.0
    autoencoder_n_ensemble: int = 10

    # --- Processing ---
    min_data_points: int = 10
    max_missing_fraction: float = 0.5
    use_robust_statistics: bool = True
    bootstrap_iterations: int = 1000
    n_jobs: int = -1

    # --- Output ---
    save_intermediate: bool = True
    generate_plots: bool = True
    export_format: str = "csv"
