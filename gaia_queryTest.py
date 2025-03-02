from astroquery.gaia import Gaia

try:
    # Simple test query - just get 5 rows
    test_query = """
    SELECT TOP 5
        source_id, ra, dec
    FROM gaiadr3.gaia_source
    """
    job = Gaia.launch_job_async(test_query)
    results = job.get_results()

    print("Gaia is online! Test query returned:")
    print(results)
except Exception as e:
    print("Gaia service error:", e)
