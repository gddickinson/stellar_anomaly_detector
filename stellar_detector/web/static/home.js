// Home: hero, real counts from the demo field, the featured star, and cards to the pages.
import { app, esc, isStatic, route } from './core.js';
import { MODULE_NAMES } from './fmt.js';
import { getInfo, getStar, getSummary } from './state.js';

route('home', async () => {
  const info = await getInfo();
  const d = info.demo, sum = await getSummary('demo');
  const feat = await getStar('demo', d.featured.id).catch(() => null);
  const mods = sum.modules.map(m => `${MODULE_NAMES[m.id] || m.label}: ${m.n}`).join(' · ');
  let featured;
  if (!feat) featured = `${esc(d.featured.name)} did not pass the quality cuts in this run.`;
  else if (!feat.detections.length) featured = `${esc(d.featured.name)} is in the field, passes the quality cuts, and <b>none of the six modules flags it</b>.
    Its famous irregular dimming is in Kepler light curves, which this pipeline does not use: Gaia DR3 catalogue values alone
    (position, parallax, motion, colour, brightness) show nothing unusual by these tests.
    ${feat.properties.teff_gspphot === null ? 'It has no GSP-Phot temperature or metallicity in DR3, so the lifetime and spectral tests cannot evaluate it.' : ''}`;
  else featured = `${esc(d.featured.name)} is flagged by ${feat.detections.length} detection(s)${feat.candidate ? ` and ranked #${feat.candidate.rank} by the ensemble` : ', but not selected by the ensemble'}.`;

  app().innerHTML = `
  <section class="hero">
    <p class="kicker">Gaia DR3 · statistical outliers · technosignature search</p>
    <h1 class="glow">Stellar Anomaly Detector</h1>
    <p class="lead">Fetch stars from astronomical catalogues (Gaia DR3, Hipparcos, Tycho-2, 2MASS, AllWISE), apply quality cuts,
      and run six detection modules (HR diagram, stellar lifetime, kinematics, spectral, technosignature, machine learning).
      An ensemble scorer then keeps only stars that are unusual in several independent ways, or extremely so in one.</p>
    <a class="btn" href="#/plots">Explore the HR diagram</a><a class="btn alt" href="#/candidates">Ranked candidates</a><a class="btn alt" href="#/method">How it works</a>
  </section>
  <div class="callout"><b>What "anomaly" means here.</b> Every flag is a statistical outlier relative to the other stars in the
    field, or a departure from simple stellar models. Nearly all have ordinary explanations: nearby fast-moving stars, giants,
    binaries, noisy astrometry, or limits of the models themselves. The repo searches for technosignatures such as Dyson
    spheres, but an outlier is a reason to look closer, not evidence of technology.</div>
  <div class="tiles">
    <div class="tile"><b>${d.n_input.toLocaleString()}</b><span>Gaia DR3 sources in the demo cone</span></div>
    <div class="tile"><b>${d.n_stars.toLocaleString()}</b><span>pass the quality cuts</span></div>
    <div class="tile"><b>${d.n_detections}</b><span>raw detections on ${sum.n_flagged_stars} stars</span></div>
    <div class="tile"><b>${d.n_candidates}</b><span>stars selected by the ensemble</span></div>
    <div class="tile"><b>${sum.tiers['1'] || 0}</b><span>Tier 1: unusual in 3+ independent ways</span></div>
  </div>
  <p class="small mute">Raw detections per module: ${esc(mods)}.</p>
  ${isStatic() ? `<div class="callout a">This is the static showcase: the demo field was analysed by the repo's Python pipeline when
    the site was built, and the Scores page can re-apply the ensemble rules in your browser. Fetching other sky regions,
    other catalogues, uploads and changed detection settings need the local app (see About).</div>` : ''}
  <h2>The demo field</h2>
  <div class="panel"><p><b>${esc(d.title)}</b>: every Gaia DR3 source within ${d.radius_deg}° of RA ${d.ra}°, Dec +${d.dec}°
    with a positive parallax, fetched with the repo's own <code>DataFetcher</code> on ${esc(d.retrieved)}.</p>
    <p>${featured} <a href="#/star/${encodeURIComponent(d.featured.id)}">Open its page</a>.</p>
    <p class="small mute">The field has no WISE photometry (the AllWISE fetch needs CDS VizieR, which was unavailable when the
    demo was built), so the Dyson-sphere and infrared-excess tests have nothing to work on here and report no candidates.</p></div>
  <h2>Pages</h2>
  <div class="grid">
    <a class="card" href="#/run"><b>Run</b><span>Fetch a catalogue for any sky region, upload a CSV, or re-run the demo with other thresholds (local app).</span></a>
    <a class="card" href="#/plots"><b>Plots</b><span>HR diagram with the main-sequence model, proper motions, Teff–[M/H] and the sky, coloured by ensemble tier or module. Click a star.</span></a>
    <a class="card" href="#/candidates"><b>Candidates</b><span>The ensemble's ranked list with tier, families, methods and descriptions; download as CSV.</span></a>
    <a class="card" href="#/scores"><b>Scores</b><span>Raw detections per module and method, their score distributions, and an ensemble-rule explorer.</span></a>
    <a class="card" href="#/method"><b>Method</b><span>What each module tests, the quality cuts, the ensemble tiers, and the caveats.</span></a>
    <a class="card" href="#/readme"><b>README</b><span>The project README: features, installation, command line and Python API.</span></a>
  </div>`;
});
