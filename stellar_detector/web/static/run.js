// Run: fetch a catalogue for a sky region (repo's DataFetcher, needs the network), upload a catalogue CSV, or re-run the
// demo field with other detection settings. Local app only; the static site explains how the demo was made.
import { api, app, esc, go, isStatic, needsServer, route } from './core.js';
import { getInfo, setRun } from './state.js';

const SRC = {
  demo: 'Demo field (committed Gaia DR3 CSV, no download)', gaia: 'Gaia DR3 (ESA archive, TAP)', hipparcos: 'Hipparcos-2 (VizieR)',
  tycho2: 'Tycho-2 (VizieR)', '2mass': '2MASS (VizieR)', allwise: 'AllWISE (VizieR)', synthetic: 'Synthetic (generated, with injected anomalies)',
};

function settings(info) {
  return `<fieldset><legend>Detection settings</legend><div class="row">${info.config_fields.map(c => `<div>
    <label for="cf-${c.name}">${esc(c.label)}</label><input id="cf-${c.name}" data-cf="${c.name}" type="number" min="${c.min}" max="${c.max}" step="${c.step}" value="${c.default}">
    <p class="hint">${esc(c.help)}</p></div>`).join('')}</div></fieldset>`;
}
const readConfig = root => Object.fromEntries([...root.querySelectorAll('[data-cf]')].map(i => [i.dataset.cf, i.value]));

route('run', async () => {
  const info = await getInfo(), d = info.demo;
  const demoText = `<div class="panel"><h3>How the demo field was made</h3><p>${esc(d.title)}: ${d.n_input.toLocaleString()} sources,
    ${d.n_stars} after the quality cuts. Query: <code>${esc(d.query)}</code>. Retrieved ${esc(d.retrieved)}. Data: ${esc(d.credit)}.</p>
    <p class="small">Equivalent command line (fetches live data, so the result can differ slightly):</p>
    <pre>python -m stellar_detector.cli --source gaia --stars 20000 --ra ${d.ra} --dec ${d.dec} --radius ${d.radius_deg}</pre></div>`;
  if (isStatic()) {
    app().innerHTML = `<h1>Run an analysis</h1>${needsServer('Fetching catalogues, uploading a CSV and changing the detection settings')}
      <p>The static site shows the demo field analysed with the default settings. The ensemble rules can be changed in your
      browser on the <a href="#/scores">Scores</a> page.</p>${demoText}`;
    return;
  }
  app().innerHTML = `<h1>Run an analysis</h1>
    <p class="lead">The full pipeline in Python on this computer: fetch, quality cuts, six modules, ensemble. Catalogue queries
      go to the ESA Gaia archive or CDS VizieR and can take a minute; repeated regions are served from the fetcher's cache.</p>
    <form class="panel params" id="f"><fieldset><legend>Data</legend>
      <label for="src">Source</label><select id="src"><option value="demo">${SRC.demo}</option>${info.sources.map(s => `<option value="${s}">${esc(SRC[s] || s)}</option>`).join('')}</select>
      <div class="row" id="region"><div><label for="n">Stars (max)</label><input id="n" type="number" min="10" max="50000" value="2000"></div>
        <div><label for="ra">RA centre (deg)</label><input id="ra" type="number" min="0" max="360" step="any" value="${d.ra}"></div>
        <div><label for="dec">Dec centre (deg)</label><input id="dec" type="number" min="-90" max="90" step="any" value="${d.dec}"></div>
        <div><label for="rad">Radius (deg)</label><input id="rad" type="number" min="0.001" max="10" step="any" value="0.2"></div></div>
      <label><input type="checkbox" id="cache" checked> Use the fetcher's cache</label>
      <p class="hint">Gaia returns the first N rows matching the cone (no ordering), so a small N is not a random sample.</p></fieldset>
      ${settings(info)}<button id="go">Run</button><span id="msg" class="small mute" role="status"></span></form>
    <form class="panel params" id="u"><fieldset><legend>Or upload a catalogue CSV</legend>
      <label for="file">CSV with Gaia DR3 column names (source_id, ra, dec, parallax, parallax_error, pmra, pmdec, phot_g_mean_mag, bp_rp, ...), as <code>cli --file</code> reads</label>
      <input id="file" type="file" accept=".csv,text/csv"><p class="hint">Uses the detection settings above.</p></fieldset>
      <button id="up">Analyse file</button><span id="umsg" class="small mute" role="status"></span></form>${demoText}`;
  const root = app(), f = root.querySelector('#f');
  const region = () => root.querySelector('#region').classList.toggle('hide', root.querySelector('#src').value === 'demo');
  root.querySelector('#src').addEventListener('change', region); region();
  const done = r => { setRun(r.run_id); go('#/candidates'); };
  f.addEventListener('submit', async e => {
    e.preventDefault();
    const btn = root.querySelector('#go'), msg = root.querySelector('#msg');
    btn.disabled = true; msg.textContent = ' Running...';
    try {
      done(await api('/run', { body: { source: root.querySelector('#src').value, n_stars: root.querySelector('#n').value,
        ra: root.querySelector('#ra').value, dec: root.querySelector('#dec').value, radius: root.querySelector('#rad').value,
        use_cache: root.querySelector('#cache').checked, config: readConfig(root) } }));
    } catch (err) { msg.innerHTML = ` <span class="err">${esc(err.message)}</span>`; btn.disabled = false; }
  });
  root.querySelector('#u').addEventListener('submit', async e => {
    e.preventDefault();
    const file = root.querySelector('#file').files[0], msg = root.querySelector('#umsg');
    if (!file) { msg.textContent = ' Choose a file first.'; return; }
    const fd = new FormData(); fd.append('file', file); fd.append('config', JSON.stringify(readConfig(root)));
    msg.textContent = ' Analysing...';
    try { done(await api('/run/upload', { body: fd })); } catch (err) { msg.innerHTML = ` <span class="err">${esc(err.message)}</span>`; }
  });
});
