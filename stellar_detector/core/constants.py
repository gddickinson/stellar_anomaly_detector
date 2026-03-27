"""Physical constants, quality thresholds, and catalog metadata."""

from __future__ import annotations


# --- Physical constants ---
PHYSICAL_CONSTANTS = {
    "solar_luminosity_watts": 3.828e26,
    "solar_mass_kg": 1.989e30,
    "solar_temperature_k": 5778,
    "solar_absolute_mag_v": 4.83,
    "solar_absolute_mag_g": 4.67,
    "parsec_meters": 3.0857e16,
    "speed_of_light_m_s": 2.998e8,
    "stefan_boltzmann": 5.670e-8,
    "hubble_time_gyr": 13.8,
}

# --- Quality thresholds for Gaia DR3 ---
QUALITY_THRESHOLDS = {
    "min_parallax_over_error": 5.0,
    "max_ruwe": 1.4,
    "max_astrometric_excess_noise_sig": 2.0,
    "phot_bp_rp_excess_range": (0.8, 1.5),
    "max_g_variability": 2.0,
    "min_classprob_star": 0.9,
    "min_snr_w3": 3.5,
    "min_snr_w4": 3.5,
}

# --- Catalog metadata ---
CATALOG_METADATA = {
    "gaia_dr3": {
        "name": "Gaia DR3",
        "vizier_id": "I/355/gaiadr3",
        "tap_url": "https://gea.esac.esa.int/tap-server/tap",
        "epoch": 2016.0,
        "description": "ESA Gaia Data Release 3 — ~1.8 billion sources",
        "key_columns": [
            "source_id", "ra", "dec", "parallax", "parallax_error",
            "pmra", "pmdec", "phot_g_mean_mag", "bp_rp",
            "teff_gspphot", "logg_gspphot", "mh_gspphot",
            "ruwe", "astrometric_excess_noise", "astrometric_excess_noise_sig",
        ],
    },
    "hipparcos": {
        "name": "Hipparcos-2",
        "vizier_id": "I/311/hip2",
        "epoch": 1991.25,
        "description": "Hipparcos re-reduction — ~118k stars",
        "key_columns": [
            "HIP", "RArad", "DErad", "Plx", "e_Plx",
            "pmRA", "pmDE", "Hpmag", "B-V",
        ],
    },
    "tycho2": {
        "name": "Tycho-2",
        "vizier_id": "I/259/tyc2",
        "epoch": 2000.0,
        "description": "Tycho-2 catalog — ~2.5 million stars",
        "key_columns": [
            "TYC", "RAmdeg", "DEmdeg", "pmRA", "pmDE",
            "BTmag", "VTmag",
        ],
    },
    "2mass": {
        "name": "2MASS Point Source",
        "vizier_id": "II/246/out",
        "epoch": 2000.0,
        "description": "Two Micron All Sky Survey — ~470 million sources",
        "key_columns": [
            "RAJ2000", "DEJ2000", "Jmag", "Hmag", "Kmag",
            "e_Jmag", "e_Hmag", "e_Kmag", "Qflg",
        ],
    },
    "allwise": {
        "name": "AllWISE",
        "vizier_id": "II/328/allwise",
        "epoch": 2010.5,
        "description": "AllWISE Source Catalog — ~747 million sources",
        "key_columns": [
            "RAJ2000", "DEJ2000", "W1mag", "W2mag", "W3mag", "W4mag",
            "e_W1mag", "e_W2mag", "e_W3mag", "e_W4mag",
            "snr1", "snr2", "snr3", "snr4",
        ],
    },
}

# --- Display names for anomaly types ---
ANOMALY_DISPLAY_NAMES = {
    "hr_diagram_outlier": "HR Diagram Outlier",
    "unusual_metallicity": "Unusual Metallicity",
    "lifetime_anomaly": "Lifetime Anomaly",
    "unusual_kinematics": "Unusual Kinematics",
    "photometric_anomaly": "Photometric Anomaly",
    "variability_anomaly": "Variability Anomaly",
    "spatial_isolation": "Spatial Isolation",
    "unusual_clustering": "Unusual Clustering",
    "luminosity_anomaly": "Luminosity Anomaly",
    "color_anomaly": "Color Anomaly",
    "rotation_anomaly": "Rotation Anomaly",
    "magnetic_anomaly": "Magnetic Anomaly",
    "chemical_anomaly": "Chemical Anomaly",
    "geometric_pattern": "Geometric Pattern",
    "temporal_pattern": "Temporal Pattern",
    "dyson_sphere_candidate": "Dyson Sphere Candidate",
    "stellar_engine_candidate": "Stellar Engine Candidate",
    "megastructure_candidate": "Megastructure Candidate",
    "binary_system_anomaly": "Binary System Anomaly",
    "galactic_orbit_anomaly": "Galactic Orbit Anomaly",
    "cross_catalog_anomaly": "Cross-Catalog Anomaly",
    "infrared_excess": "Infrared Excess",
    "astrometric_anomaly": "Astrometric Anomaly",
}
