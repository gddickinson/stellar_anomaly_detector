import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
import os
import requests
import io
import warnings
warnings.filterwarnings('ignore')

class StellarAnomalyDetector:
    """
    A class to download and analyze stellar data to detect potential anomalies
    that might indicate artificial modification of stars.
    """

    def __init__(self, output_dir='./stellar_data'):
        """Initialize the detector with a directory for output data."""
        self.output_dir = output_dir
        self.data = None
        self.anomalies = None
        self.data_source = None

        # Configure Vizier to return larger tables
        Vizier.ROW_LIMIT = -1

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        print("Stellar Anomaly Detector initialized.")

    def fetch_gaia_data(self, n_stars=1000, random_sample=True):
        """
        Fetch stellar data from Gaia DR3

        Parameters:
        -----------
        n_stars : int
            Number of stars to download
        random_sample : bool
            If True, get a random sample; otherwise get the brightest stars
        """
        print(f"Fetching Gaia data for {n_stars} stars...")
        self.data_source = "gaia"

        try:
            # Start with a simpler query to test the connection
            print("Testing Gaia connection with a simple query...")
            test_query = """
            SELECT TOP 5
                source_id, ra, dec
            FROM gaiadr3.gaia_source
            """
            test_job = Gaia.launch_job_async(test_query)
            test_results = test_job.get_results()
            print("Gaia connection test successful!")

            # Set up the main query
            if random_sample:
                # Get a random sample by using a random offset
                import random
                max_offset = 100000  # Reduced from 1,000,000 to make query more manageable
                offset = random.randint(0, max_offset)
                print(f"Using random offset: {offset}")

                # More focused query with fewer columns for better reliability
                query = f"""
                SELECT TOP {n_stars}
                    source_id, ra, dec, parallax,
                    phot_g_mean_mag, bp_rp,
                    teff_gspphot
                FROM gaiadr3.gaia_source
                WHERE parallax > 0
                  AND teff_gspphot > 0
                  AND parallax_over_error > 5  -- Ensure more reliable data
                OFFSET {offset}
                """
            else:
                # Get the brightest stars
                print("Querying brightest stars...")
                query = f"""
                SELECT TOP {n_stars}
                    source_id, ra, dec, parallax,
                    phot_g_mean_mag, bp_rp,
                    teff_gspphot
                FROM gaiadr3.gaia_source
                WHERE parallax > 0
                  AND teff_gspphot > 0
                  AND parallax_over_error > 5  -- Ensure more reliable data
                ORDER BY phot_g_mean_mag ASC
                """

            print("Executing Gaia query...")
            print("Query: ", query)

            # Execute the query with a timeout
            #job = Gaia.launch_job_async(query, timeout=300)  # 5 minute timeout
            job = Gaia.launch_job_async(query)  # Without timeout parameter
            results = job.get_results()

            # Convert to pandas DataFrame
            print("Converting results to DataFrame...")
            self.data = results.to_pandas()

            print(f"Successfully downloaded data for {len(self.data)} stars from Gaia DR3.")

            # Calculate absolute magnitude
            self.data['distance'] = 1000 / self.data['parallax']  # in parsecs
            self.data['abs_g_mag'] = self.data['phot_g_mean_mag'] - 5 * np.log10(self.data['distance']) + 5

            # Save the data to CSV for future use
            self.data.to_csv(f"{self.output_dir}/gaia_data.csv", index=False)

            # Print a sample of the data for verification
            print("Sample of retrieved data:")
            print(self.data.head())

            return self.data

        except Exception as e:
            print(f"Error fetching Gaia data: {e}")
            print("Error details:", str(e))
            import traceback
            traceback.print_exc()
            print("You can try an alternative data source or use sample data.")
            return None

    def fetch_hipparcos_data(self, n_stars=1000):
        """
        Fetch stellar data from Hipparcos-2 catalog via VizieR

        Parameters:
        -----------
        n_stars : int
            Number of stars to download
        """
        print(f"Fetching Hipparcos data for {n_stars} stars...")
        self.data_source = "hipparcos"

        try:
            # Set up Vizier query for Hipparcos-2 catalog
            catalog = "I/311/hip2"
            v = Vizier(columns=["**"])
            v.ROW_LIMIT = n_stars

            # Query bright stars
            result = v.query_constraints(catalog=catalog,
                                       Plx=">0",
                                       Hpmag="<10")  # Bright stars with positive parallax

            if len(result) > 0 and len(result[0]) > 0:
                # Convert to pandas DataFrame
                self.data = result[0].to_pandas()

                print("Original Hipparcos columns:", list(self.data.columns))

                # Rename columns to be more consistent with Gaia
                column_mapping = {
                    'HIP': 'source_id',
                    'RArad': 'ra',
                    'DErad': 'dec',
                    'Plx': 'parallax',
                    'pmRA': 'pmra',
                    'pmDE': 'pmdec',
                    'Hpmag': 'phot_hp_mean_mag',  # Hipparcos magnitude
                    'B-V': 'bp_rp',  # Color equivalent
                }

                # Apply renaming for columns that exist
                for old_col, new_col in column_mapping.items():
                    if old_col in self.data.columns:
                        self.data = self.data.rename(columns={old_col: new_col})

                # Calculate distance in parsecs
                if 'parallax' in self.data.columns and self.data['parallax'].notnull().any():
                    self.data['distance'] = 1000 / self.data['parallax']

                    # Calculate absolute magnitude using Hipparcos magnitude
                    if 'phot_hp_mean_mag' in self.data.columns:
                        self.data['abs_hp_mag'] = self.data['phot_hp_mean_mag'] - 5 * np.log10(self.data['distance']) + 5

                # Estimate temperature from B-V color (rough estimation)
                if 'bp_rp' in self.data.columns:
                    self.data['teff_est'] = 4600 * (1 / (0.92 * self.data['bp_rp'] + 1.7) + 1/0.92)

                print("After renaming:", list(self.data.columns))

                # Save the data to CSV for future use
                self.data.to_csv(f"{self.output_dir}/hipparcos_data.csv", index=False)

                print(f"Successfully downloaded data for {len(self.data)} stars from Hipparcos catalog.")
                return self.data
            else:
                print("No data returned from Hipparcos catalog.")
                return None

        except Exception as e:
            print(f"Error fetching Hipparcos data: {e}")
            print("Trying to use sample data instead...")
            return self.load_sample_data()

    # def fetch_tycho2_data(self, n_stars=1000):
    #     """
    #     Fetch stellar data from Tycho-2 catalog via VizieR

    #     Parameters:
    #     -----------
    #     n_stars : int
    #         Number of stars to download
    #     """
    #     print(f"Fetching Tycho-2 data for {n_stars} stars...")
    #     self.data_source = "tycho2"

    #     try:
    #         # Set up Vizier query for Tycho-2 catalog
    #         catalog = "I/259/tyc2"
    #         v = Vizier(columns=["**"])
    #         v.ROW_LIMIT = n_stars

    #         # Query bright stars
    #         result = v.query_constraints(catalog=catalog,
    #                                    VTmag="<10")  # Bright stars

    #         if len(result) > 0 and len(result[0]) > 0:
    #             # Convert to pandas DataFrame
    #             self.data = result[0].to_pandas()

    #             # Rename columns to be more consistent with Gaia
    #             column_mapping = {
    #                 'TYC1': 'tyc1',
    #                 'TYC2': 'tyc2',
    #                 'TYC3': 'tyc3',
    #                 'RAmdeg': 'ra',
    #                 'DEmdeg': 'dec',
    #                 'pmRA': 'pmra',
    #                 'pmDE': 'pmdec',
    #                 'BTmag': 'phot_b_mean_mag',
    #                 'VTmag': 'phot_v_mean_mag',
    #                 'B-V': 'bp_rp'
    #             }

    #             # Apply renaming for columns that exist
    #             for old_col, new_col in column_mapping.items():
    #                 if old_col in self.data.columns:
    #                     self.data = self.data.rename(columns={old_col: new_col})

    #             # After renaming columns:

    #             # Create B-V color from B and V magnitudes
    #             if 'phot_b_mean_mag' in self.data.columns and 'phot_v_mean_mag' in self.data.columns:
    #                 self.data['bp_rp'] = self.data['phot_b_mean_mag'] - self.data['phot_v_mean_mag']

    #             # Look for HIP catalog matches (many Tycho stars have corresponding Hipparcos entries)
    #             # If there's a HIP column, we might be able to get parallax from there

    #             # Estimate temperature from B-V color
    #             if 'bp_rp' in self.data.columns:
    #                 self.data['teff_est'] = 4600 * (1 / (0.92 * self.data['bp_rp'] + 1.7) + 1/0.92)


    #             # Create a source_id from Tycho IDs
    #             if 'tyc1' in self.data.columns and 'tyc2' in self.data.columns and 'tyc3' in self.data.columns:
    #                 self.data['source_id'] = self.data['tyc1'].astype(str) + '-' + \
    #                                         self.data['tyc2'].astype(str) + '-' + \
    #                                         self.data['tyc3'].astype(str)

    #             # Save the data to CSV for future use
    #             self.data.to_csv(f"{self.output_dir}/tycho2_data.csv", index=False)

    #             print(f"Successfully downloaded data for {len(self.data)} stars from Tycho-2 catalog.")
    #             return self.data
    #         else:
    #             print("No data returned from Tycho-2 catalog.")
    #             return None

    #     except Exception as e:
    #         print(f"Error fetching Tycho-2 data: {e}")
    #         return None


    def fetch_tycho2_data(self, n_stars=1000):
        """
        Fetch stellar data from Tycho-2 catalog via VizieR

        Parameters:
        -----------
        n_stars : int
            Number of stars to download
        """
        print(f"Fetching Tycho-2 data for {n_stars} stars...")
        self.data_source = "tycho2"

        try:
            # Set up Vizier query for Tycho-2 catalog
            catalog = "I/259/tyc2"
            v = Vizier(columns=["**"])
            v.ROW_LIMIT = n_stars

            # Query bright stars
            result = v.query_constraints(catalog=catalog,
                                       VTmag="<10")  # Bright stars

            if len(result) > 0 and len(result[0]) > 0:
                # Convert to pandas DataFrame
                self.data = result[0].to_pandas()

                # Print original columns
                print("Original Tycho-2 columns:", list(self.data.columns))

                # Rename columns to be more consistent with Gaia
                column_mapping = {
                    'TYC1': 'tyc1',
                    'TYC2': 'tyc2',
                    'TYC3': 'tyc3',
                    'RAmdeg': 'ra',
                    'DEmdeg': 'dec',
                    'pmRA': 'pmra',
                    'pmDE': 'pmdec',
                    'BTmag': 'phot_b_mean_mag',
                    'VTmag': 'phot_v_mean_mag',
                }

                # Apply renaming for columns that exist
                for old_col, new_col in column_mapping.items():
                    if old_col in self.data.columns:
                        self.data = self.data.rename(columns={old_col: new_col})

                # Create a source_id from Tycho IDs
                if 'tyc1' in self.data.columns and 'tyc2' in self.data.columns and 'tyc3' in self.data.columns:
                    self.data['source_id'] = self.data['tyc1'].astype(str) + '-' + \
                                            self.data['tyc2'].astype(str) + '-' + \
                                            self.data['tyc3'].astype(str)

                # Create B-V color from B and V magnitudes
                if 'phot_b_mean_mag' in self.data.columns and 'phot_v_mean_mag' in self.data.columns:
                    self.data['bp_rp'] = self.data['phot_b_mean_mag'] - self.data['phot_v_mean_mag']

                # Estimate temperature from B-V color
                if 'bp_rp' in self.data.columns:
                    # Filter out unrealistic B-V values
                    valid_color = (self.data['bp_rp'] > -0.5) & (self.data['bp_rp'] < 3.0)
                    self.data.loc[valid_color, 'teff_est'] = 4600 * (1 / (0.92 * self.data.loc[valid_color, 'bp_rp'] + 1.7) + 1/0.92)

                # Now add absolute magnitude calculations using Hipparcos cross-reference
                # First check if we have HIP IDs for cross-matching
                # Add absolute magnitude calculations using direct Hipparcos access
                if 'HIP' in self.data.columns:
                    # Find stars with valid Hipparcos IDs
                    hip_stars = self.data[self.data['HIP'].notna() & (self.data['HIP'] > 0)]
                    if len(hip_stars) > 0:
                        print(f"Found {len(hip_stars)} stars with Hipparcos IDs, fetching parallax data...")

                        try:
                            # Directly query the Hipparcos catalog
                            hip_catalog = Vizier(columns=["HIP", "Plx"])
                            hip_catalog.ROW_LIMIT = -1  # No row limit

                            # Extract unique non-null HIP IDs
                            unique_hip_ids = hip_stars['HIP'].dropna().astype(int).unique().tolist()

                            # Query in smaller batches (VizieR sometimes fails with too many constraints)
                            plx_data = pd.DataFrame()
                            batch_size = 100
                            for i in range(0, len(unique_hip_ids), batch_size):
                                batch_ids = unique_hip_ids[i:i+batch_size]
                                hip_query = " | ".join([f"HIP={hip_id}" for hip_id in batch_ids])

                                result = hip_catalog.query_constraints(catalog="I/311/hip2",
                                                                      column_filters={"HIP": batch_ids})

                                if result and len(result) > 0:
                                    batch_data = result[0].to_pandas()
                                    plx_data = pd.concat([plx_data, batch_data])

                            if len(plx_data) > 0:
                                print(f"Retrieved parallax data for {len(plx_data)} Hipparcos stars")

                                # Convert HIP to int for proper merging
                                plx_data['HIP'] = plx_data['HIP'].astype(int)

                                # Merge with our dataset on HIP column
                                self.data = pd.merge(self.data, plx_data, on='HIP', how='left')

                                # Now calculate distance and absolute magnitude
                                mask = (self.data['Plx'] > 0) & self.data['Plx'].notna()
                                if mask.any():
                                    self.data.loc[mask, 'distance'] = 1000 / self.data.loc[mask, 'Plx']

                                    if 'phot_v_mean_mag' in self.data.columns:
                                        self.data.loc[mask, 'abs_v_mag'] = (
                                            self.data.loc[mask, 'phot_v_mean_mag'] -
                                            5 * np.log10(self.data.loc[mask, 'distance']) + 5
                                        )

                                    print(f"Added distance and absolute magnitude for {mask.sum()} stars")
                        except Exception as e:
                            print(f"Error retrieving parallax data: {e}")
                            print("Will continue without absolute magnitudes")


                # Print final columns
                print("Final Tycho-2 columns:", list(self.data.columns))

                # Save the data to CSV for future use
                self.data.to_csv(f"{self.output_dir}/tycho2_data.csv", index=False)

                print(f"Successfully downloaded data for {len(self.data)} stars from Tycho-2 catalog.")
                return self.data
            else:
                print("No data returned from Tycho-2 catalog.")
                return None

        except Exception as e:
            print(f"Error fetching Tycho-2 data: {e}")
            return None

    def load_bright_star_catalog(self):
        """
        Load Yale Bright Star Catalog (local download or remote)
        """
        print("Loading Yale Bright Star Catalog...")
        self.data_source = "bright_star"

        try:
            # Try to download the catalog from VizieR
            catalog = "V/50/catalog"
            v = Vizier(columns=["**"])
            result = v.query_constraints(catalog=catalog)

            if len(result) > 0 and len(result[0]) > 0:
                # Convert to pandas DataFrame
                self.data = result[0].to_pandas()

                # Rename columns to be more consistent
                column_mapping = {
                    'HR': 'source_id',
                    'RAh': 'ra_h',
                    'RAm': 'ra_m',
                    'RAs': 'ra_s',
                    'DEd': 'dec_d',
                    'DEm': 'dec_m',
                    'DEs': 'dec_s',
                    'Vmag': 'phot_v_mean_mag',
                    'B-V': 'bp_rp',
                    'SpType': 'spectral_type'
                }

                # Apply renaming for columns that exist
                for old_col, new_col in column_mapping.items():
                    if old_col in self.data.columns:
                        self.data = self.data.rename(columns={old_col: new_col})

                # Calculate RA and Dec in degrees
                if 'ra_h' in self.data.columns and 'ra_m' in self.data.columns and 'ra_s' in self.data.columns:
                    self.data['ra'] = 15 * (self.data['ra_h'] +
                                           self.data['ra_m']/60 +
                                           self.data['ra_s']/3600)

                if 'dec_d' in self.data.columns and 'dec_m' in self.data.columns and 'dec_s' in self.data.columns:
                    dec_sign = self.data['dec_d'].apply(lambda x: -1 if x < 0 else 1)
                    self.data['dec'] = self.data['dec_d'] + dec_sign * (
                                       self.data['dec_m']/60 +
                                       self.data['dec_s']/3600)

                # Save the data to CSV for future use
                self.data.to_csv(f"{self.output_dir}/bright_star_catalog.csv", index=False)

                print(f"Successfully loaded {len(self.data)} stars from Yale Bright Star Catalog.")
                return self.data
            else:
                print("No data returned from Yale Bright Star Catalog.")
                return None

        except Exception as e:
            print(f"Error loading Yale Bright Star Catalog: {e}")
            print("Trying to load sample data instead...")
            return self.load_sample_data()

    def load_sample_data(self):
        """Load sample stellar data included with the package"""
        print("Loading sample stellar data...")
        self.data_source = "sample"

        try:
            # Check if there's a cached sample dataset
            sample_file = f"{self.output_dir}/sample_stars.csv"

            # If we already have a sample dataset, load it
            if os.path.exists(sample_file):
                self.data = pd.read_csv(sample_file)
                print(f"Loaded sample data with {len(self.data)} stars.")
                return self.data

            # Otherwise, create a synthetic dataset
            n_stars = 1000
            np.random.seed(42)  # For reproducibility

            # Generate random star parameters
            source_ids = np.arange(1, n_stars + 1)
            ra = np.random.uniform(0, 360, n_stars)
            dec = np.random.uniform(-90, 90, n_stars)

            # Generate realistic masses (weighted towards lower masses)
            mass = np.random.lognormal(mean=0.0, sigma=0.5, size=n_stars) * 0.5 + 0.1

            # Temperature depends on mass
            teff = 5800 * (mass ** 0.5) * (1 + 0.1 * np.random.normal(size=n_stars))

            # Luminosity depends on mass and age
            # L ∝ M^3.5 for main sequence
            age = np.random.uniform(0.1, 10, n_stars)  # Age in Gyr
            age_factor = np.where(age > 1, 1 + 0.1 * np.log(age), 1)  # Slight increase with age
            luminosity = (mass ** 3.5) * age_factor

            # Radius ∝ M^0.8 for main sequence
            radius = (mass ** 0.8) * (1 + 0.05 * np.random.normal(size=n_stars))

            # Metallicity depends on age (older stars are less metal-rich)
            metal = -0.3 * (age / 10) + 0.1 * np.random.normal(size=n_stars)

            # Create synthetic dataframe
            self.data = pd.DataFrame({
                'source_id': source_ids,
                'ra': ra,
                'dec': dec,
                'mass_flame': mass,
                'age_flame': age,
                'teff_gspphot': teff,
                'lum_flame': luminosity,
                'radius_flame': radius,
                'mh_gspphot': metal
            })

            # Add some anomalies for testing
            # 1. A few long-lived stars
            anomaly_indices = np.random.choice(n_stars, 10, replace=False)
            self.data.loc[anomaly_indices, 'age_flame'] = self.data.loc[anomaly_indices, 'age_flame'] * 2

            # 2. A few metal-rich old stars
            anomaly_indices = np.random.choice(n_stars, 5, replace=False)
            self.data.loc[anomaly_indices, 'mh_gspphot'] = 0.5  # Very metal rich
            self.data.loc[anomaly_indices, 'age_flame'] = 8 + np.random.uniform(0, 2, 5)  # Old stars

            # 3. A few HR diagram outliers
            anomaly_indices = np.random.choice(n_stars, 8, replace=False)
            self.data.loc[anomaly_indices, 'teff_gspphot'] = 12000  # Hot
            self.data.loc[anomaly_indices, 'lum_flame'] = 0.01  # But dim

            # Save the synthetic data
            self.data.to_csv(sample_file, index=False)

            print(f"Created synthetic sample data with {n_stars} stars (including anomalies for testing).")
            return self.data

        except Exception as e:
            print(f"Error creating sample data: {e}")
            return None

    def load_cached_data(self, filename=None):
        """Load previously downloaded stellar data from a CSV file"""
        if filename is None:
            # Try to load any available data set
            for file in ['gaia_data.csv', 'hipparcos_data.csv', 'tycho2_data.csv',
                         'bright_star_catalog.csv', 'sample_stars.csv']:
                full_path = os.path.join(self.output_dir, file)
                if os.path.exists(full_path):
                    filename = full_path
                    self.data_source = file.split('_')[0].replace('.csv', '')
                    break

            if filename is None:
                print("No cached data files found. Please download data first.")
                return None

        print(f"Loading data from {filename}...")
        try:
            self.data = pd.read_csv(filename)
            print(f"Successfully loaded {len(self.data)} stars from {filename}.")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def fetch_variable_stars(self):
        """Fetch data on known variable stars from VizieR."""
        print("Fetching variable star data...")

        try:
            # Use the General Catalogue of Variable Stars
            catalog = "B/gcvs/gcvs_cat"
            v = Vizier(columns=["**"])
            result = v.get_catalogs(catalog)

            if len(result) > 0:
                variables = result[0].to_pandas()
                print(f"Downloaded {len(variables)} variable stars.")
                return variables
            else:
                print("No variable star data found.")
                return None

        except Exception as e:
            print(f"Error fetching variable star data: {e}")
            return None

    def detect_hr_diagram_outliers(self, plot=True):
        """
        Detect stars that are outliers on the HR diagram,
        which might indicate non-standard stellar evolution.
        """
        if self.data is None:
            print("No data loaded. Please fetch data first.")
            return None

        print("Looking for HR diagram outliers...")

        # Define temperature and color/magnitude columns based on data source
        if self.data_source == 'gaia':
            temp_col = 'teff_gspphot'
            color_col = 'bp_rp'
            abs_mag_col = 'abs_g_mag'
        elif self.data_source == 'hipparcos':
            temp_col = 'teff_est' if 'teff_est' in self.data.columns else None
            color_col = 'bp_rp'  # B-V in Hipparcos
            abs_mag_col = 'abs_hp_mag'
        elif self.data_source == 'sample':
            temp_col = 'teff_gspphot'
            color_col = None
            abs_mag_col = None

            # For sample data, we can use luminosity instead
            if 'lum_flame' in self.data.columns:
                self.data['abs_mag_est'] = -2.5 * np.log10(self.data['lum_flame']) + 4.74
                abs_mag_col = 'abs_mag_est'
        else:
            # Try to find appropriate columns for temperature
            temp_candidates = ['teff_gspphot', 'teff_est', 'teff', 'Teff']
            for col in temp_candidates:
                if col in self.data.columns:
                    temp_col = col
                    break
            else:
                temp_col = None

            # Try to find appropriate columns for color
            color_candidates = ['bp_rp', 'B-V', 'color']
            for col in color_candidates:
                if col in self.data.columns:
                    color_col = col
                    break
            else:
                color_col = None

            # Try to find appropriate columns for absolute magnitude
            mag_candidates = ['abs_g_mag', 'abs_v_mag', 'abs_mag']
            for col in mag_candidates:
                if col in self.data.columns:
                    abs_mag_col = col
                    break
            else:
                abs_mag_col = None

        # Check which parameters we can use for the HR diagram
        usable_params = []

        if temp_col is not None and temp_col in self.data.columns:
            usable_params.append(temp_col)

        if color_col is not None and color_col in self.data.columns:
            usable_params.append(color_col)

        if abs_mag_col is not None and abs_mag_col in self.data.columns:
            usable_params.append(abs_mag_col)

        if len(usable_params) < 2:
            print(f"Not enough parameters for HR diagram. Need at least 2 of: temperature, color, absolute magnitude.")
            # Try to use luminosity if available
            if 'lum_flame' in self.data.columns:
                print("Using luminosity instead of absolute magnitude.")
                self.data['abs_mag_est'] = -2.5 * np.log10(self.data['lum_flame']) + 4.74
                abs_mag_col = 'abs_mag_est'
                usable_params.append(abs_mag_col)

        if len(usable_params) < 2:
            print("Cannot create HR diagram with available columns.")
            return None

        # Decide which columns to use for HR diagram
        if temp_col is not None and abs_mag_col is not None:
            # Use temperature and absolute magnitude
            x_col = temp_col
            y_col = abs_mag_col
            x_label = 'Temperature (K)'
            y_label = 'Absolute Magnitude'
            x_reverse = True

        elif color_col is not None and abs_mag_col is not None:
            # Use color and absolute magnitude
            x_col = color_col
            y_col = abs_mag_col
            x_label = f'Color ({color_col})'
            y_label = 'Absolute Magnitude'
            x_reverse = False

        elif temp_col is not None and color_col is not None:
            # Use temperature and color
            x_col = temp_col
            y_col = color_col
            x_label = 'Temperature (K)'
            y_label = f'Color ({color_col})'
            x_reverse = True

        else:
            # Fallback to whatever we have
            x_col = usable_params[0]
            y_col = usable_params[1]
            x_label = x_col
            y_label = y_col
            x_reverse = False

        # Filter out rows with missing values
        filtered_data = self.data.dropna(subset=[x_col, y_col])

        if len(filtered_data) == 0:
            print("No valid data points after filtering.")
            return None

        # Calculate z-scores for both parameters
        from scipy import stats
        filtered_data['x_zscore'] = stats.zscore(filtered_data[x_col])
        filtered_data['y_zscore'] = stats.zscore(filtered_data[y_col])

        # Define outliers as stars with combined z-scores > 3
        filtered_data['combined_zscore'] = np.sqrt(filtered_data['x_zscore']**2 + filtered_data['y_zscore']**2)
        outliers = filtered_data[filtered_data['combined_zscore'] > 3]

        print(f"Found {len(outliers)} potential HR diagram outliers out of {len(filtered_data)} stars.")

        if plot and len(filtered_data) > 0:
            plt.figure(figsize=(10, 8))

            # Plot normal stars
            plt.scatter(filtered_data[x_col], filtered_data[y_col],
                       alpha=0.5, s=5, color='blue', label='Normal stars')

            # Plot outliers
            if len(outliers) > 0:
                plt.scatter(outliers[x_col], outliers[y_col],
                           alpha=0.8, s=20, color='red', label='Potential anomalies')

            if y_col.startswith('abs_'):
                plt.gca().invert_yaxis()  # Invert y-axis for absolute magnitudes (astronomical convention)

            if x_reverse:
                plt.gca().invert_xaxis()  # Invert x-axis for temperature

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title('HR Diagram with Potential Anomalies')
            plt.legend()
            plt.savefig(f"{self.output_dir}/hr_diagram_outliers.png")
            plt.show()

        return outliers

    def detect_unusual_metallicity(self):
        """
        Detect stars with unusual metallicity patterns that may
        indicate artificial enrichment or stellar engineering.
        """
        if self.data is None:
            print("No data loaded. Please fetch data first.")
            return None

        print("Looking for unusual metallicity patterns...")

        # Define the metallicity column based on data source
        if self.data_source == 'gaia':
            metal_col = 'mh_gspphot'
            if metal_col not in self.data.columns:
                print(f"Metallicity column '{metal_col}' not found in Gaia data. Skipping metallicity analysis.")
                return None
        elif self.data_source == 'sample':
            metal_col = 'mh_gspphot'
        else:
            # Try to find any metallicity column
            metal_candidates = ['mh_gspphot', 'mh', 'feh', 'metallicity', '[Fe/H]', 'Fe_H']
            for col in metal_candidates:
                if col in self.data.columns:
                    metal_col = col
                    break
            else:
                print("No metallicity data found in this dataset.")
                return None

        # Filter out rows with missing metallicity
        filtered_data = self.data.dropna(subset=[metal_col])

        if len(filtered_data) == 0:
            print("No metallicity data available after filtering.")
            return None

        # Calculate metallicity statistics
        mean_metal = filtered_data[metal_col].mean()
        std_metal = filtered_data[metal_col].std()

        # Find metal-rich outliers (3 sigma above mean)
        # These could indicate artificial enrichment
        metal_rich = filtered_data[filtered_data[metal_col] > mean_metal + 3*std_metal]

        # Find stars that are unusually metal-rich for their age
        age_metal_anomalies = pd.DataFrame()
        age_col = None

        # Try to find an age column
        age_candidates = ['age_flame', 'age', 'Age']
        for col in age_candidates:
            if col in filtered_data.columns:
                age_col = col
                break

        if age_col is not None:
            # Older stars should have lower metallicity
            filtered_data = filtered_data.dropna(subset=[age_col])
            if len(filtered_data) > 0:
                filtered_data['expected_metal'] = -0.3 * filtered_data[age_col] / 10 + 0.1
                filtered_data['metal_anomaly'] = filtered_data[metal_col] - filtered_data['expected_metal']

                # Find stars with metallicity much higher than expected for their age
                age_metal_anomalies = filtered_data[filtered_data['metal_anomaly'] > 0.5]

                print(f"Found {len(metal_rich)} unusually metal-rich stars")
                print(f"Found {len(age_metal_anomalies)} stars with metallicity too high for their age")

                # Plot metallicity vs age
                plt.figure(figsize=(10, 6))
                plt.scatter(filtered_data[age_col], filtered_data[metal_col], alpha=0.5, s=5)

                if len(age_metal_anomalies) > 0:
                    plt.scatter(age_metal_anomalies[age_col], age_metal_anomalies[metal_col],
                               color='red', alpha=0.8, s=20)

                plt.xlabel('Age (Gyr)')
                plt.ylabel(f'Metallicity [{metal_col}]')
                plt.title('Metallicity vs Age with Potential Anomalies')
                plt.savefig(f"{self.output_dir}/metallicity_anomalies.png")
        else:
            print(f"Found {len(metal_rich)} unusually metal-rich stars (no age data available)")

        # Combine the two types of metallicity anomalies
        if len(age_metal_anomalies) > 0:
            return pd.concat([metal_rich, age_metal_anomalies]).drop_duplicates()
        else:
            return metal_rich

    def detect_lifetime_anomalies(self):
        """
        Detect stars that appear older than their expected lifetime
        based on their mass, which could indicate artificial life extension.
        """
        if self.data is None:
            print("No data loaded. Please fetch data first.")
            return None

        print("Looking for stellar lifetime anomalies...")

        # Define mass and age columns based on data source
        if self.data_source == 'gaia':
            mass_col = 'mass_flame'
            age_col = 'age_flame'
            if mass_col not in self.data.columns or age_col not in self.data.columns:
                print(f"Required mass or age columns not found in Gaia data. Skipping lifetime anomaly detection.")
                return None
        elif self.data_source == 'sample':
            mass_col = 'mass_flame'
            age_col = 'age_flame'
        else:
            # Try to find appropriate columns
            mass_candidates = ['mass_flame', 'mass', 'Mass', 'M']
            age_candidates = ['age_flame', 'age', 'Age']

            # Find mass column
            for col in mass_candidates:
                if col in self.data.columns:
                    mass_col = col
                    break
            else:
                print("No mass data found in this dataset.")
                return None

            # Find age column
            for col in age_candidates:
                if col in self.data.columns:
                    age_col = col
                    break
            else:
                print("No age data found in this dataset.")
                return None

        # Check if the columns exist
        if mass_col not in self.data.columns or age_col not in self.data.columns:
            print(f"Required columns {mass_col} or {age_col} not found in data.")
            return None

        # Filter out rows with missing values
        filtered_data = self.data.dropna(subset=[mass_col, age_col])

        if len(filtered_data) == 0:
            print("No mass and age data available after filtering.")
            return None

        # Calculate expected lifetime based on mass
        # Main sequence lifetime ~ M^-2.5 (in solar units)
        # Sun's lifetime is about 10 Gyr
        filtered_data['expected_lifetime'] = 10 * (filtered_data[mass_col] ** -2.5)

        # Calculate lifetime ratio (actual age / expected lifetime)
        filtered_data['lifetime_ratio'] = filtered_data[age_col] / filtered_data['expected_lifetime']

        # Find stars living longer than expected
        long_lived = filtered_data[filtered_data['lifetime_ratio'] > 0.9]

        print(f"Found {len(long_lived)} potential long-lived stars out of {len(filtered_data)}.")

        # Plot mass vs age with expected lifetime curve
        plt.figure(figsize=(10, 6))
        plt.scatter(filtered_data[mass_col], filtered_data[age_col], alpha=0.5, s=5)

        if len(long_lived) > 0:
            plt.scatter(long_lived[mass_col], long_lived[age_col],
                       color='red', alpha=0.8, s=20, label='Potential long-lived anomalies')

        # Plot expected lifetime curve
        mass_range = np.linspace(0.5, 3, 100)
        expected_age = 10 * (mass_range ** -2.5)
        plt.plot(mass_range, expected_age, 'k--', label='Expected maximum age')

        plt.xlabel(f'Mass ({mass_col})')
        plt.ylabel(f'Age ({age_col})')
        plt.title('Stellar Age vs Mass with Potential Anomalies')
        plt.legend()
        plt.savefig(f"{self.output_dir}/lifetime_anomalies.png")

        return long_lived


    def compile_anomalies(self):
        """Compile all detected anomalies into a single report."""
        print("Compiling anomaly report...")

        # Run all detection algorithms
        hr_outliers = self.detect_hr_diagram_outliers()
        metallicity_outliers = self.detect_unusual_metallicity()
        lifetime_outliers = self.detect_lifetime_anomalies()

        # Collect all anomalies
        all_anomalies = []

        if hr_outliers is not None and len(hr_outliers) > 0:
            hr_outliers['anomaly_type'] = 'HR diagram outlier'
            all_anomalies.append(hr_outliers)
        else:
            print("No HR diagram outliers detected or analysis skipped.")

        if metallicity_outliers is not None and len(metallicity_outliers) > 0:
            metallicity_outliers['anomaly_type'] = 'Unusual metallicity'
            all_anomalies.append(metallicity_outliers)
        else:
            print("No metallicity outliers detected or analysis skipped.")

        if lifetime_outliers is not None and len(lifetime_outliers) > 0:
            lifetime_outliers['anomaly_type'] = 'Unusual lifetime'
            all_anomalies.append(lifetime_outliers)
        else:
            print("No lifetime outliers detected or analysis skipped.")

        # Combine all anomalies
        if all_anomalies:
            self.anomalies = pd.concat(all_anomalies)

            # Count stars with multiple anomalies
            anomaly_counts = self.anomalies.groupby('source_id').size()
            multi_anomaly_stars = anomaly_counts[anomaly_counts > 1].index

            print(f"Found a total of {len(self.anomalies)} anomalies in {len(self.anomalies['source_id'].unique())} stars.")
            print(f"Found {len(multi_anomaly_stars)} stars with multiple anomaly types.")

            # Save the anomalies to a CSV file
            self.anomalies.to_csv(f"{self.output_dir}/stellar_anomalies_{self.data_source}.csv", index=False)


            return self.anomalies
        else:
            print("No anomalies detected.")
            return None

    def cross_match_anomalies(self, max_separation=5.0):
        """
        Cross-match anomalies between different catalogs based on sky coordinates.

        Parameters:
        -----------
        max_separation : float
            Maximum separation in arcseconds to consider stars as matching

        Returns:
        --------
        pandas.DataFrame
            Table of matched anomalies across catalogs
        """
        print("\nCross-matching anomalies between catalogs...")

        # Get list of anomaly files in the output directory
        anomaly_files = []
        for file in os.listdir(self.output_dir):
            if file.startswith("stellar_anomalies") and file.endswith(".csv"):
                anomaly_files.append(file)

        if len(anomaly_files) < 2:
            print("Need at least two anomaly files for cross-matching. Found: {}".format(anomaly_files))
            return None

        print(f"Found {len(anomaly_files)} anomaly files: {anomaly_files}")

        # Load anomalies from each file
        catalog_anomalies = {}
        for file in anomaly_files:
            catalog_name = file.replace("stellar_anomalies_", "").replace(".csv", "")
            if not catalog_name:
                catalog_name = "unknown"

            file_path = os.path.join(self.output_dir, file)
            try:
                df = pd.read_csv(file_path)
                if 'ra' in df.columns and 'dec' in df.columns:
                    catalog_anomalies[catalog_name] = df
                    print(f"Loaded {len(df)} anomalies from {catalog_name}")
                else:
                    print(f"Skipping {file}: missing coordinates")
            except Exception as e:
                print(f"Error loading {file}: {e}")

        if len(catalog_anomalies) < 2:
            print("Need at least two valid anomaly catalogs for cross-matching.")
            return None

        # Use astropy's SkyCoord for accurate coordinate matching
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from astropy.table import Table

        # Create a dictionary to store SkyCoord objects for each catalog
        sky_coords = {}
        for name, df in catalog_anomalies.items():
            sky_coords[name] = SkyCoord(ra=df['ra'].values * u.degree,
                                        dec=df['dec'].values * u.degree)

        # Create a table to store match results
        matches = []

        # Compare each pair of catalogs
        catalogs = list(catalog_anomalies.keys())
        for i in range(len(catalogs)):
            for j in range(i+1, len(catalogs)):
                cat1, cat2 = catalogs[i], catalogs[j]

                # Find matches between catalogs
                idx1, idx2, sep, _ = sky_coords[cat1].search_around_sky(
                    sky_coords[cat2], max_separation * u.arcsec)

                print(f"Found {len(idx1)} matches between {cat1} and {cat2}")

                # Store matches
                for k in range(len(idx1)):
                    star1 = catalog_anomalies[cat1].iloc[idx1[k]]
                    star2 = catalog_anomalies[cat2].iloc[idx2[k]]

                    # Get common properties
                    props = {
                        'ra': (star1['ra'] + star2['ra']) / 2,
                        'dec': (star1['dec'] + star2['dec']) / 2,
                        'separation_arcsec': sep[k].to(u.arcsec).value,
                        'catalogs': f"{cat1},{cat2}",
                    }

                    # Add anomaly types
                    if 'anomaly_type' in star1:
                        props[f'{cat1}_anomaly'] = star1['anomaly_type']
                    if 'anomaly_type' in star2:
                        props[f'{cat2}_anomaly'] = star2['anomaly_type']

                    # Add catalog-specific IDs
                    for col in star1.index:
                        if 'source_id' in col or 'HIP' in col or 'TYC' in col:
                            props[f'{cat1}_{col}'] = star1[col]
                    for col in star2.index:
                        if 'source_id' in col or 'HIP' in col or 'TYC' in col:
                            props[f'{cat2}_{col}'] = star2[col]

                    matches.append(props)

        if not matches:
            print("No cross-catalog matches found.")
            return None

        # Convert to DataFrame
        matches_df = pd.DataFrame(matches)

        # Save results
        output_file = os.path.join(self.output_dir, "cross_matched_anomalies.csv")
        matches_df.to_csv(output_file, index=False)
        print(f"Saved {len(matches_df)} cross-matched anomalies to {output_file}")

        # Create a sky plot of matched anomalies
        try:
            plt.figure(figsize=(12, 8))
            plt.scatter(matches_df['ra'], matches_df['dec'], c=matches_df['separation_arcsec'],
                       cmap='viridis', s=50, alpha=0.8)
            plt.colorbar(label='Separation (arcsec)')
            plt.xlabel('Right Ascension (degrees)')
            plt.ylabel('Declination (degrees)')
            plt.title('Cross-Matched Anomalous Stars')
            plt.grid(alpha=0.3)

            # Save the plot
            plot_file = os.path.join(self.output_dir, "cross_matched_sky_plot.png")
            plt.savefig(plot_file)
            print(f"Saved sky plot to {plot_file}")
        except Exception as e:
            print(f"Error creating sky plot: {e}")

        return matches_df


    def analyze_multiple_catalogs(self, catalogs=['gaia', 'hipparcos', 'tycho2'], n_stars=2000):
        """
        Run analysis on multiple catalogs and cross-match the results.

        Parameters:
        -----------
        catalogs : list
            List of catalog names to analyze
        n_stars : int
            Number of stars to analyze from each catalog

        Returns:
        --------
        pandas.DataFrame
            Cross-matched anomalies across catalogs
        """
        print(f"\nAnalyzing multiple catalogs: {catalogs}")

        for source in catalogs:
            print(f"\nAnalyzing {source} data...")

            # Create a new detector for each catalog to ensure clean data
            # (We'll reuse self for the final cross-matching)
            if source != self.data_source:
                detector = StellarAnomalyDetector(output_dir=self.output_dir)

                # Load or fetch data based on the catalog
                data = None
                if source == 'gaia':
                    data = detector.fetch_gaia_data(n_stars=n_stars)
                elif source == 'hipparcos':
                    data = detector.fetch_hipparcos_data(n_stars=n_stars)
                elif source == 'tycho2':
                    data = detector.fetch_tycho2_data(n_stars=n_stars)
                elif source == 'bright_star':
                    data = detector.load_bright_star_catalog()
                elif source == 'sample':
                    data = detector.load_sample_data()

                # Only proceed if data was successfully loaded
                if data is not None:
                    # Find anomalies and save with catalog-specific filename
                    anomalies = detector.compile_anomalies()
                    if anomalies is not None:
                        anomalies.to_csv(f"{self.output_dir}/stellar_anomalies_{source}.csv", index=False)
            else:
                # If we're already analyzing this source, just rename the anomalies file
                if self.anomalies is not None:
                    self.anomalies.to_csv(f"{self.output_dir}/stellar_anomalies_{source}.csv", index=False)

        # Cross-match anomalies between catalogs
        return self.cross_match_anomalies()

# Main execution
if __name__ == "__main__":
    import os
    import argparse

    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Detect stellar anomalies that might indicate artificial modification')
    parser.add_argument('--source', type=str, choices=['gaia', 'hipparcos', 'tycho2', 'bright_star', 'sample', 'cached', 'multi'],
                        default='sample', help='Data source to use (use "multi" for multi-catalog analysis)')
    parser.add_argument('--stars', type=int, default=2000, help='Number of stars to analyze')
    parser.add_argument('--cached', type=str, help='Path to cached data file')
    parser.add_argument('--output', type=str, default='./stellar_data',
                        help='Directory to store results')
    parser.add_argument('--catalogs', type=str, default='gaia,hipparcos,tycho2',
                        help='Comma-separated list of catalogs to use with --source=multi')

    args = parser.parse_args()

    # Initialize the detector
    detector = StellarAnomalyDetector(output_dir=args.output)

    # Special case for multi-catalog analysis
    if args.source == 'multi':
        catalogs = args.catalogs.split(',')
        detector.analyze_multiple_catalogs(catalogs=catalogs, n_stars=args.stars)
    else:
        # Fetch data from the specified source
        data = None
        if args.source == 'gaia':
            data = detector.fetch_gaia_data(n_stars=args.stars)
        elif args.source == 'hipparcos':
            data = detector.fetch_hipparcos_data(n_stars=args.stars)
        elif args.source == 'tycho2':
            data = detector.fetch_tycho2_data(n_stars=args.stars)
        elif args.source == 'bright_star':
            data = detector.load_bright_star_catalog()
        elif args.source == 'sample':
            data = detector.load_sample_data()
        elif args.source == 'cached':
            data = detector.load_cached_data(filename=args.cached)

        if data is not None:
            # Run the anomaly detection pipeline
            anomalies = detector.compile_anomalies()

            if anomalies is not None:
                print("\nTop potential anomalous stars:")

                # Determine which columns to show based on data source
                if detector.data_source in ['gaia', 'sample']:
                    columns = ['source_id', 'ra', 'dec', 'teff_gspphot', 'mass_flame', 'age_flame', 'anomaly_type']
                elif detector.data_source == 'hipparcos':
                    columns = ['source_id', 'ra', 'dec', 'phot_v_mean_mag', 'bp_rp', 'anomaly_type']
                else:
                    # For other data sources, just show what's available
                    columns = ['source_id', 'anomaly_type']
                    for col in ['ra', 'dec', 'teff_gspphot', 'mass_flame', 'age_flame', 'phot_v_mean_mag']:
                        if col in anomalies.columns:
                            columns.insert(-1, col)

                # Filter columns that exist in the anomalies dataframe
                columns = [col for col in columns if col in anomalies.columns]

                # Print top anomalies
                print(anomalies[columns].head(10))

    print("\nAnalysis complete. Check the '{}' directory for results.".format(args.output))
