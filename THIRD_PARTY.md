# Third-party data

## Gaia DR3 demo field (`stellar_detector/web/demo/gaia_dr3_kic8462852_field.csv`)

- **What:** all 4,112 Gaia DR3 sources within 0.15° of RA 301.5644°, Dec +44.4568° (the position of KIC 8462852,
  "Boyajian's star", Gaia DR3 2081900940499099136) with `parallax > 0` and a G magnitude. Used as the demo dataset of
  the web interface and the static site, and by `tests/test_web.py`.
- **How:** fetched with the repo's own code,
  `DataFetcher().fetch(CatalogSource.GAIA_DR3, n_stars=20000, ra_center=301.5644, dec_center=44.4568, radius_deg=0.15, use_cache=False)`,
  i.e. the "full async" ADQL query in `stellar_detector/data/fetcher.py` against `gaiadr3.gaia_source` at the ESA Gaia
  archive (TAP). `TOP 20000` was not reached, so the cone is complete for those conditions. Columns as returned
  (source_id, ra, dec, parallax, parallax_error, pmra, pmdec, phot_g_mean_mag, bp_rp, teff_gspphot, logg_gspphot,
  mh_gspphot, ruwe, astrometric_excess_noise, astrometric_excess_noise_sig, phot_bp_rp_excess_factor, radial_velocity,
  phot_g_mean_flux_over_error) plus `catalog_source`. Not modified.
- **When:** 2026-11-11.
- **Source and licence:** ESA/Gaia/DPAC, Gaia Data Release 3. Gaia data are released under the Creative Commons
  Attribution-ShareAlike 3.0 IGO licence (CC BY-SA 3.0 IGO); see https://www.cosmos.esa.int/web/gaia-users/license.
  This CSV is therefore shared under CC BY-SA 3.0 IGO, not under the repository's MIT licence.
- **Acknowledgement:** This work has made use of data from the European Space Agency (ESA) mission Gaia
  (https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and Analysis Consortium (DPAC,
  https://www.cosmos.esa.int/web/gaia/dpac/consortium). Funding for the DPAC has been provided by national
  institutions, in particular the institutions participating in the Gaia Multilateral Agreement.
- **Citation:** Gaia Collaboration, Vallenari et al. (2023), A&A 674, A1 (Gaia DR3 summary).

No WISE or 2MASS photometry is included: CDS VizieR was unavailable when the demo was built.

## Front-end markdown renderer

`stellar_detector/web/static/md.js` is copied from George's particle_trajectory_analyzer web interface (same author).
