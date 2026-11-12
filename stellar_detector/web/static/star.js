// One star: ensemble result, every raw detection with its parameters, catalogue properties, and its place on the HR
// diagram and in the field's distributions.
import { app, esc, go, route } from './core.js';
import { MODULE_COLORS, MODULE_NAMES, TIER_NAMES, colLabel, dot, f, nice, tierDot } from './fmt.js';
import { drawOrder, starColors } from './plots.js';
import { scatter } from './plot.js';
import { getCandidates, getDetections, getStar, getStars, getSummary, runBar, runId, starIndex, wireRunBar } from './state.js';

const PCT_COLS = ['pm_total', 'v_tan_km_s', 'abs_mag', 'bp_rp', 'mh_gspphot', 'ruwe', 'astrometric_excess_noise_sig'];

function percentile(vals, v, analysed) {
  const s = vals.filter((x, i) => x !== null && (!analysed || analysed[i]));
  return s.length && v !== null ? (100 * s.filter(x => x < v).length) / s.length : null;
}
const kv = obj => `<dl class="kv">${Object.entries(obj).map(([k, v]) => `<dt>${esc(nice(k))}</dt><dd>${esc(typeof v === 'number' ? f(v) : Array.isArray(v) ? v.join(', ') : JSON.stringify(v))}</dd>`).join('')}</dl>`;

route('star', async (args, query) => {
  const id = runId(query), sid = args[0];
  const [sum, star, stars, cands, dets] = await Promise.all([
    getSummary(id), getStar(id, sid), getStars(id), getCandidates(id), getDetections(id)]);
  const p = star.properties, c = star.candidate, idx = starIndex(stars).get(sid);
  const simbad = `https://simbad.cds.unistra.fr/simbad/sim-coo?Coord=${p.ra}+${p.dec}&Radius=5&Radius.unit=arcsec`;
  const status = c ? `${tierDot(c.tier)} ranked <b>#${c.rank}</b> of ${star.n_candidates} candidates`
    : star.detections.length ? `${dot(TIER_NAMES.raw, '#9b8cd9')}: ${star.detections.length} raw detection(s), below the ensemble's bar`
      : 'Not flagged by any module.';
  app().innerHTML = `${runBar(sum)}<p class="kicker">Star</p><h1 class="glow" style="overflow-wrap:anywhere">${esc(sid)}</h1>
    <p>${status}</p>
    ${p.analysed === false ? `<p class="static-note">This star failed the RUWE quality cut (RUWE ${f(p.ruwe, 3)} &gt; ${f(sum.config.max_ruwe)}), so only the RUWE test was run on it.</p>` : ''}
    ${c ? `<div class="panel"><h3>Ensemble result</h3><p>${esc(c.description)}</p>
      <p class="small mute">Type ${esc(nice(c.anomaly_type))} · priority ${c.follow_up_priority} · significance ${f(c.significance_score, 3)} · confidence ${f(c.confidence, 2)} · method <code>${esc(c.detection_method)}</code></p>
      ${c.observational_recommendations.length ? `<p class="small">Suggested follow-up (from the module): ${c.observational_recommendations.map(esc).join('; ')}</p>` : ''}</div>` : ''}
    <div class="callout">A flag means this star is a statistical outlier in this field or departs from the simple models used.
      Check the ordinary explanations first: a nearby or fast-moving star, a giant or subgiant, an unresolved binary, noisy astrometry,
      a wrong parallax, extinction, or the limits of the models.</div>
    <div class="two"><div><h2>Raw detections</h2>
      ${star.detections.length ? star.detections.map(d => `<div class="panel"><p>${dot(MODULE_NAMES[d.module], MODULE_COLORS[d.module])} <b>${esc(nice(d.anomaly_type))}</b>
        <span class="small mute">· ${esc(d.detection_method)}</span></p><p>${esc(d.description)}</p>
        <p class="small mute">significance ${f(d.significance_score, 3)} · confidence ${f(d.confidence, 2)} · priority ${d.follow_up_priority} · family ${esc(d.family)}</p>
        ${Object.keys(d.parameters).length ? `<details><summary class="small">Parameters</summary>${kv(d.parameters)}</details>` : ''}</div>`).join('')
        : '<p class="mute">None of the six modules flagged this star.</p>'}
      <h2>Within the field</h2><p class="small mute">Percentile among the ${sum.n_stars.toLocaleString()} stars of this run (share of stars with a smaller value).</p>
      <dl class="kv">${PCT_COLS.filter(k => stars[k]).map(k => { const q = percentile(stars[k], p[k], stars.analysed); return `<dt>${esc(colLabel(k))}</dt><dd>${f(p[k])} ${q === null ? '' : `· ${q.toFixed(1)}th percentile`}</dd>`; }).join('')}</dl></div>
    <div><h2>On the HR diagram</h2><div id="hr"></div>
      <h2>Catalogue values</h2><dl class="kv">${Object.entries(p).filter(([, v]) => v !== null).map(([k, v]) => `<dt>${esc(colLabel(k))}</dt><dd>${esc(typeof v === 'number' ? f(v, 6) : String(v))}</dd>`).join('')}</dl>
      <p class="small"><a href="${simbad}" target="_blank" rel="noopener">Look up these coordinates in SIMBAD</a></p></div></div>
    <p><a href="#/candidates">← All candidates</a></p>`;
  const root = app();
  wireRunBar(root);
  if (stars.bp_rp && stars.abs_mag && p.bp_rp !== null && p.abs_mag !== null) {
    const colors = starColors(stars, cands, dets, 'tier'), hr = sum.hr;
    scatter(root.querySelector('#hr'), {
      x: stars.bp_rp, y: stars.abs_mag, colors, order: drawOrder(colors), invertY: true, highlight: idx, aspect: 0.85,
      xlabel: colLabel('bp_rp'), ylabel: colLabel('abs_mag'), lines: [{ x: hr.bp_rp, y: hr.ms_mag, color: '#e8f1ff' }],
      label: i => esc(stars.source_id[i]), onPick: i => go(`#/star/${encodeURIComponent(stars.source_id[i])}`),
    });
  } else root.querySelector('#hr').innerHTML = '<p class="mute">No colour or absolute magnitude for this star.</p>';
});
