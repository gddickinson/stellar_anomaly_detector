// Scores: raw detections per module and method, score distributions, and an explorer that re-applies the ensemble
// rules (ensemble_engine.js, a tested port of EnsembleScorer) to the raw detections in the browser.
import { app, esc, go, route, slider } from './core.js';
import { aggregate, DEFAULTS } from './ensemble_engine.js';
import { MODULE_COLORS, MODULE_NAMES, TIER_COLORS, dot, f, nice, tierDot } from './fmt.js';
import { histogram } from './plot.js';
import { getCandidates, getDetections, getInfo, getSummary, runBar, runId, wireRunBar } from './state.js';

const SCALE = {
  hr: 'max over the agreeing methods: |z| of the main-sequence test, or fixed 6.0 (DBSCAN), 5.5 (KDE), 5.0 (GMM)',
  lifetime: '10 × age fraction (fraction of the ZAMS→TAMS band crossed, clipped at 1.5); flagged above 0.9, so always ≥ 9',
  kinematics: '|z| in MAD units (proper motion, velocity), RUWE × 3, or excess-noise significance',
  spectral: '|z| (metallicity), 10 × |Isolation Forest score|, or 5 × |LOF score|',
  technosignature: '50 × RMSE improvement (Dyson fit), |z| (WISE colours), v_tan / 50 (stellar engine)',
  ml: '10 × weighted ensemble score; flagged above the ensemble threshold, so always ≥ 10 × threshold',
};

route('scores', async (args, query) => {
  const id = runId(query);
  const [info, sum, dets, cands] = await Promise.all([getInfo(), getSummary(id), getDetections(id), getCandidates(id)]);
  const byMod = m => dets.filter(d => d.module === m.id);
  app().innerHTML = `${runBar(sum)}<h1>Scores</h1>
    <p class="lead">${dets.length} raw detections from six modules on ${sum.n_flagged_stars} of ${sum.n_stars.toLocaleString()} stars,
      reduced by the ensemble to ${cands.length} candidates.</p>
    <div class="scroll"><table class="tbl"><thead><tr><th>Module</th><th>Detections</th><th>Methods (count)</th><th>Significance scale</th></tr></thead><tbody>
    ${sum.modules.map(m => `<tr><td class="nw">${dot(MODULE_NAMES[m.id] || m.label, MODULE_COLORS[m.id])}</td><td>${m.n}</td>
      <td>${m.error ? `<span class="err">failed: ${esc(m.error)}</span>` : Object.entries(m.methods).map(([k, n]) => `<span class="tag">${esc(k)} ${n}</span>`).join('') || '<span class="mute">none</span>'}</td>
      <td class="desc small">${esc(SCALE[m.id] || '')}</td></tr>`).join('')}</tbody></table></div>
    <h2>Significance by module</h2>
    <div class="charts">${sum.modules.filter(m => m.n).map(m => `<div class="panel"><h3>${esc(MODULE_NAMES[m.id])}</h3>
      ${histogram(byMod(m).map(d => d.significance_score), { xlabel: 'significance score', bins: 20, marks: [{ x: info.ensemble.single_min_significance, label: 'Tier 3 cut' }] })}</div>`).join('')}</div>
    <p class="small mute">The dashed line is the ensemble's single-method cut (significance ≥ ${info.ensemble.single_min_significance}).
      Because the scales differ, the same cut is strict for some modules and automatically passed by others.</p>
    <h2>Ensemble explorer</h2>
    <p>Re-apply the ensemble rules to these raw detections with other settings, in your browser. With the defaults this reproduces
      the Python result exactly (${cands.length} candidates); the code is a port of <code>EnsembleScorer</code> tested against it.</p>
    <div class="side"><div class="panel params" id="ctl"><fieldset><legend>Modules</legend><div class="checks">
      ${sum.modules.map(m => `<label><input type="checkbox" data-mod="${m.id}" checked> ${esc(MODULE_NAMES[m.id] || m.label)}</label>`).join('')}</div></fieldset>
      <fieldset id="rules"><legend>Rules</legend></fieldset><button class="alt" id="reset">Reset</button></div>
      <div id="out"></div></div>`;
  const root = app();
  wireRunBar(root);
  const o = { ...DEFAULTS, high_importance_types: info.ensemble.high_importance_types };
  const rules = root.querySelector('#rules');
  const sliders = [
    slider(rules, { id: 't1', label: 'Tier 1: minimum independent families', min: 2, max: 6, value: o.tier1_min_families }, v => { o.tier1_min_families = v; run(); }),
    slider(rules, { id: 't2', label: 'Tier 2: minimum methods for one type', min: 2, max: 4, value: o.tier2_min_methods }, v => { o.tier2_min_methods = v; run(); }),
    slider(rules, { id: 't3', label: 'Tier 3: minimum single-method significance', min: 2, max: 30, step: 0.5, value: o.single_min_significance }, v => { o.single_min_significance = v; run(); }),
    slider(rules, { id: 'hi', label: 'High-importance types: minimum confidence', min: 0.5, max: 1, step: 0.05, value: o.high_importance_min_confidence, fmt: v => v.toFixed(2) }, v => { o.high_importance_min_confidence = v; run(); }),
  ];
  function run() {
    o.modules = new Set([...root.querySelectorAll('[data-mod]')].filter(c => c.checked).map(c => c.dataset.mod));
    const res = aggregate(dets, o), tiers = [1, 2, 3].map(t => res.filter(r => r.tier === t).length);
    const types = {};
    for (const r of res) types[r.anomaly_type] = (types[r.anomaly_type] || 0) + 1;
    root.querySelector('#out').innerHTML = `<div class="tiles">
      <div class="tile"><b>${res.length}</b><span>candidates (Python default: ${cands.length})</span></div>
      ${tiers.map((n, i) => `<div class="tile" style="border-left-color:${TIER_COLORS[i + 1]}"><b>${n}</b><span>Tier ${i + 1}</span></div>`).join('')}</div>
      <p class="small">${Object.entries(types).sort((a, b) => b[1] - a[1]).map(([t, n]) => `<span class="tag">${esc(nice(t))} ${n}</span>`).join('')}</p>
      <div class="scroll"><table class="tbl"><thead><tr><th>#</th><th>Star</th><th>Tier</th><th>Priority</th><th>Significance</th><th>Description</th></tr></thead><tbody>
      ${res.slice(0, 15).map(r => `<tr class="click" tabindex="0" data-id="${esc(r.star_id)}"><td>${r.rank}</td><td class="nw">${esc(r.star_id)}</td><td>${tierDot(r.tier)}</td><td>${r.follow_up_priority}</td><td>${f(r.significance_score, 3)}</td><td class="desc">${esc(r.description)}</td></tr>`).join('')}
      </tbody></table></div>${res.length > 15 ? `<p class="small mute">First 15 of ${res.length}.</p>` : ''}`;
    root.querySelectorAll('#out tr[data-id]').forEach(tr => {
      const open = () => go(`#/star/${encodeURIComponent(tr.dataset.id)}`);
      tr.addEventListener('click', open); tr.addEventListener('keydown', e => { if (e.key === 'Enter') open(); });
    });
  }
  root.querySelectorAll('[data-mod]').forEach(c => c.addEventListener('change', run));
  root.querySelector('#reset').addEventListener('click', () => {
    Object.assign(o, DEFAULTS, { high_importance_types: info.ensemble.high_importance_types });
    [o.tier1_min_families, o.tier2_min_methods, o.single_min_significance, o.high_importance_min_confidence].forEach((v, i) => {
      sliders[i].value = v; sliders[i].dispatchEvent(new Event('input'));
    });
    root.querySelectorAll('[data-mod]').forEach(c => { c.checked = true; });
    run();
  });
  run();
});
