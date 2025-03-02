# Stellar Anomaly Detector

This project downloads stellar data from various astronomical catalogs and analyzes it to detect potential anomalies that might indicate artificial modification of stars by advanced civilizations.

## Types of Anomalies Detected

1. **HR Diagram Outliers**: Stars that don't fit on the standard Hertzsprung-Russell diagram, potentially indicating non-standard evolution.

2. **Unusual Metallicity Patterns**: Stars with unexpectedly high metallicity, especially for their age, which could suggest artificial enrichment.

3. **Lifetime Anomalies**: Stars that appear to be living longer than expected based on their mass, potentially indicating artificial life extension.

## Installation

### Requirements
- Python 3.7+
- Required packages: numpy, pandas, matplotlib, astropy, astroquery, scipy

### Setup

1. Create a virtual environment (recommended):
```
python -m venv stellar_env
source stellar_env/bin/activate  # On Windows: stellar_env\Scripts\activate
```

2. Install required packages:
```
pip install -r requirements.txt
```

## Usage

Run the main script with your preferred data source:
```
python stellar_anomaly_detector.py --source sample
```

Available data sources:
- `gaia`: Fetch data from Gaia DR3 catalog (requires internet connection)
- `hipparcos`: Fetch data from Hipparcos-2 catalog via VizieR
- `tycho2`: Fetch data from Tycho-2 catalog via VizieR
- `bright_star`: Use Yale Bright Star Catalog
- `sample`: Use built-in synthetic sample data (works offline)
- `cached`: Load previously downloaded data

Examples:
```
# Use sample data (works offline)
python stellar_anomaly_detector.py --source sample

# Use Gaia data (needs internet connection)
python stellar_anomaly_detector.py --source gaia --stars 5000

# Use previously downloaded data
python stellar_anomaly_detector.py --source cached --cached ./stellar_data/gaia_data.csv
```

The script will:
1. Fetch or load data from the specified source
2. Analyze the data for various types of anomalies
3. Generate visualizations in the `stellar_data` directory
4. Save a CSV file of potential anomalies to `stellar_data/stellar_anomalies.csv`

## Interpreting Results

The anomalies detected by this program are statistical outliers that warrant further investigation. They are not definitive evidence of artificial stellar modification, but rather interesting targets for follow-up study.

For each anomalous star, consider:

1. Could measurement errors explain the anomaly?
2. Are there natural astrophysical explanations?
3. Is the star part of a binary or multiple system that affects measurements?
4. Does the star show other unusual characteristics?

Stars with multiple types of anomalies are particularly interesting candidates for further investigation.

## Limitations

- The analysis is limited by the accuracy and completeness of the stellar catalogs
- Stellar age and mass estimates have significant uncertainties
- Some natural but rare stellar phenomena might be flagged as anomalies
- The number of stars analyzed is a tiny fraction of the Milky Way's population

## Future Improvements

- Incorporate data from multiple catalogs and surveys
- Implement more sophisticated anomaly detection algorithms
- Add detection of unusual stellar variability patterns
- Include analysis of stars' galactic orbits
- Add machine learning approaches to identify complex anomaly patterns
