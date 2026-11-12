// Method (what each module tests, from the code), About (running, credits, data licence) and the README.
import { api, app, esc, route } from './core.js';
import { f } from './fmt.js';
import { markdown, wireAnchors } from './md.js';
import { getInfo, getSummary } from './state.js';

route('method', async () => {
  const [info, sum] = await Promise.all([getInfo(), getSummary('demo')]);
  const c = sum.config, e = info.ensemble;
  app().innerHTML = `<p class="kicker">Method</p><h1 class="glow">How the pipeline decides</h1>
    <p class="lead">The same steps as <code>python -m stellar_detector.cli</code>: fetch, derive quantities, quality cuts, six
      detection modules, then the ensemble scorer. Values below are the defaults (<code>DetectionConfig</code>).</p>
    <h2>1. Derived quantities and quality cuts</h2>
    <p>Distance = 1/parallax; absolute G = G − 5 log₁₀(d) + 5 (no extinction correction); total proper motion; tangential velocity
      4.74 × μ × d; luminosity from absolute G. Stars are kept if parallax/error ≥ ${f(c.min_parallax_over_error)}, RUWE ≤ ${f(c.max_ruwe)}
      and the BP/RP flux excess factor is between ${f(c.min_phot_bp_rp_excess)} and ${f(c.max_phot_bp_rp_excess)} (missing values pass).
      In the demo field this keeps ${sum.n_stars} of ${sum.n_input.toLocaleString()} sources, mostly by removing faint distant stars with imprecise parallaxes.</p>
    <h2>2. Detection modules</h2>
    <p class="small mute">"z" below is a robust z-score: (value − median) / MAD, with the raw median absolute deviation (not scaled
      by 1.4826). The default threshold of ${f(c.mad_threshold)} MAD corresponds to about ${f(c.mad_threshold / 1.4826, 2)} σ for a Gaussian distribution.</p>
    <div class="grid wide">
      <div class="card"><b>HR diagram</b><span>Four tests on colour (BP−RP) and absolute G: distance from a piecewise-linear main-sequence model
        (|z| > ${f(c.mad_threshold)}), DBSCAN noise points, the lowest 1% of KDE density, and the lowest 1% of GMM likelihood.
        A star is reported only if at least two of the four agree.</span></div>
      <div class="card"><b>Stellar lifetime</b><span>Estimates mass from absolute G (L ∝ M³·⁵) and how far the star sits between
        linear ZAMS and TAMS lines in Teff, and flags it above 90% of the band. Giants and subgiants sit beyond the TAMS line and are
        flagged as a matter of course; this is evolution, not an impossible age.</span></div>
      <div class="card"><b>Kinematics</b><span>High total proper motion (z > ${f(c.mad_threshold)}), RUWE above ${f(c.max_ruwe)},
        astrometric excess-noise significance ≥ 5, and tangential velocity > 200 km/s with z > ${f(c.mad_threshold)}. Because the
        quality cut already removes RUWE > ${f(c.max_ruwe)}, the RUWE test cannot fire in the standard pipeline.</span></div>
      <div class="card"><b>Spectral</b><span>On Gaia GSP-Phot Teff, log g and [M/H]: metallicity z-score, Isolation Forest on
        the three, and LOF in the Teff–[M/H] plane (needs 50+ stars). Reported if two of the three agree.</span></div>
      <div class="card"><b>Technosignature</b><span>Needs WISE W3/W4 photometry. A Project Hephaistos-style grid of star +
        warm-shell blackbody SEDs (shell 100–700 K, covering factor 0.1–0.9), W1−W2 / W2−W3 colour excess, and fast stars with
        W1−W2 excess ("stellar engine"). Infrared excess has many natural causes: circumstellar dust, debris discs, young or
        evolved stars, and galaxies blended into the large WISE beam.</span></div>
      <div class="card"><b>ML pipeline</b><span>Isolation Forest, Local Outlier Factor and One-Class SVM (and coniferest's adaptive
        forest if installed) on all numeric columns except IDs and sky position. Scores are min-max normalised per method, averaged
        with weights, and flagged above ${f(c.ensemble_threshold)}.</span></div>
    </div>
    <h2>3. Ensemble</h2>
    <p>Detections are grouped by star and anomaly types are mapped to independent families (photometric, kinematic, astrometric,
      chemical, evolutionary, technosignature, infrared, ...). A star is selected as</p>
    <ul><li><b>Tier 1</b> if its detections span ≥ ${e.tier1_min_families} families;</li>
      <li><b>Tier 2</b> if one anomaly type is found by ≥ ${e.tier2_min_methods} different methods;</li>
      <li><b>Tier 3</b> if its strongest single detection has significance ≥ ${e.single_min_significance}, or is a high-importance type
        (${e.high_importance_types.map(esc).join(', ')}) with confidence ≥ ${e.high_importance_min_confidence}.</li></ul>
    <p>Candidates are ranked by follow-up priority, then significance.</p>
    <div class="callout"><b>Caveat on Tier 3.</b> Significance is not on one scale: it is |z| for the z-score tests but
      10 × age fraction for the lifetime test (always ≥ 9 when flagged) and 10 × ensemble score for the ML pipeline (always ≥ 7 with the
      defaults). Every lifetime or ML detection therefore passes the Tier 3 cut, and in the demo field Tier 3 is mostly evolved stars
      from the lifetime test. The <a href="#/scores">Scores</a> page lets you see what changes with other rules.</div>
    <h2>4. Reading the results</h2>
    <p>The modules compare each star with the others in the same field and with simple analytic models. A flagged star is unusual by
      those measures. In a field of a few hundred stars, a handful will be: the nearest dwarfs move fastest, giants lie above the main
      sequence, binaries spoil astrometric fits. Treat the ranked list as a follow-up queue to check against the literature (SIMBAD links
      are on each star's page), never as a detection of technology.</p>`;
});

route('about', async () => {
  const info = await getInfo(), d = info.demo;
  app().innerHTML = `<p class="kicker">About</p><h1 class="glow">Stellar Anomaly Detector</h1>
    <p class="lead">A modular Python package that fetches stellar data from astronomical catalogues and analyses it to find
      stars that are unusual, as candidates for follow-up in a search for technosignatures. Version ${esc(info.version)}, MIT licence.</p>
    <h2>Run the desktop app and command line</h2>
    <pre>pip install -e .
python -m stellar_detector.cli --source synthetic --stars 2000   # quick test, no network
python -m stellar_detector.cli --source gaia --stars 5000        # Gaia DR3 (network)
python -m stellar_detector.cli --gui                             # PySide6 GUI</pre>
    <h2>Run the web app</h2>
    <pre>pip install -r requirements.txt -r requirements-web.txt
python run_web.py            # http://127.0.0.1:8233</pre>
    <p>The web app runs the same pipeline: fetch any catalogue and sky region, upload a CSV, change detection settings, and browse the
      results. The static site (GitHub Pages) shows the demo field precomputed at build time:
      <code>python scripts/build_static.py</code>.</p>
    <h2>Data and credits</h2>
    <p>Demo field: ${esc(d.title)}, ${d.n_input.toLocaleString()} sources retrieved ${esc(d.retrieved)} with the repo's
      <code>DataFetcher</code>. This work has made use of data from the European Space Agency (ESA) mission Gaia
      (<a href="https://www.cosmos.esa.int/gaia" target="_blank" rel="noopener">cosmos.esa.int/gaia</a>), processed by the Gaia Data
      Processing and Analysis Consortium (DPAC). Gaia data: ${esc(d.credit)}. See THIRD_PARTY.md in the repository.</p>
    <p>Other catalogues the fetcher supports: Hipparcos-2, Tycho-2, 2MASS and AllWISE via CDS VizieR. Built with AI assistance
      from Claude (Anthropic), as noted in the README.</p>
    <p><a href="#/readme">Read the README</a> · <a href="https://github.com/gddickinson/stellar_anomaly_detector" target="_blank" rel="noopener">Source on GitHub</a></p>`;
});

route('readme', async () => {
  const doc = await api('/docs/readme');
  app().innerHTML = `<div class="md">${markdown(doc.markdown)}</div>`;
  wireAnchors(app());
});
