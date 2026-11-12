// Plots: HR diagram (with the HR module's main-sequence model and threshold band), proper motions, Teff–[M/H] and the
// sky, each star coloured by ensemble tier or by the first module that flagged it. Click a point to open the star.
import { app, esc, go, route } from './core.js';
import { MODULE_COLORS, MODULE_NAMES, TIER_COLORS, TIER_NAMES, colLabel, dot, f } from './fmt.js';
import { scatter } from './plot.js';
import { getCandidates, getDetections, getStars, getSummary, runBar, runId, wireRunBar } from './state.js';

const VIEWS = {
  hr: { title: 'HR diagram', x: 'bp_rp', y: 'abs_mag', invertY: true },
  pm: { title: 'Proper motion', x: 'pmra', y: 'pmdec', clip: 0.002 },
  chem: { title: 'Teff vs [M/H]', x: 'teff_gspphot', y: 'mh_gspphot', invertX: true },
  vel: { title: 'Distance vs tangential velocity', x: 'distance_pc', y: 'v_tan_km_s', clip: 0.002 },
  sky: { title: 'Sky', x: 'ra', y: 'dec', invertX: true },
};

export function starColors(stars, cands, dets, mode) {
  const ids = stars.source_id || [];
  const tier = new Map(cands.map(c => [c.star_id, c.tier]));
  const firstMod = new Map();
  for (const d of dets) if (!firstMod.has(d.star_id)) firstMod.set(d.star_id, d.module);
  return ids.map(id => {
    if (mode === 'module') return firstMod.has(id) ? MODULE_COLORS[firstMod.get(id)] : TIER_COLORS.none;
    if (tier.has(id)) return TIER_COLORS[tier.get(id)];
    return firstMod.has(id) ? TIER_COLORS.raw : TIER_COLORS.none;
  });
}

// Draw order: unflagged first, then flagged, then candidates by tier (Tier 1 on top).
export function drawOrder(colors) {
  const rank = c => [TIER_COLORS.none, TIER_COLORS.raw, TIER_COLORS[3], TIER_COLORS[2], TIER_COLORS[1]].indexOf(c);
  return colors.map((c, i) => i).sort((a, b) => rank(colors[a]) - rank(colors[b]));
}

route('plots', async (args, query) => {
  const id = runId(query), view = VIEWS[query.view] ? query.view : 'hr', mode = query.color === 'module' ? 'module' : 'tier';
  const [sum, stars, cands, dets] = await Promise.all([getSummary(id), getStars(id), getCandidates(id), getDetections(id)]);
  const v = VIEWS[view], hr = sum.hr;
  const has = c => (stars[c] || []).some(x => x !== null);
  const tabs = Object.entries(VIEWS).filter(([, w]) => has(w.x) && has(w.y))
    .map(([k, w]) => `<button class="${k === view ? 'on' : ''}" data-view="${k}">${esc(w.title)}</button>`).join('');
  const legend = mode === 'module'
    ? Object.entries(MODULE_COLORS).map(([m, c]) => dot(MODULE_NAMES[m], c)).join('') + dot('not flagged', TIER_COLORS.none)
    : [1, 2, 3, 'raw', 'none'].map(t => dot(TIER_NAMES[t], TIER_COLORS[t])).join('');
  app().innerHTML = `${runBar(sum)}<h1>Plots</h1>
    <p class="lead">${stars.source_id.length.toLocaleString()} stars after quality cuts. Hover for the star, click to open it.</p>
    <div class="row" style="align-items:center"><div class="seg" role="group" aria-label="Plot">${tabs}</div>
      <div><label for="cmode">Colour by</label><select id="cmode"><option value="tier">ensemble tier</option><option value="module" ${mode === 'module' ? 'selected' : ''}>first module that flagged the star</option></select></div></div>
    <div class="legend">${legend}</div>
    <div id="plot"></div>
    ${view === 'hr' && hr.center !== null ? `<p class="small mute">Solid line: the piecewise main-sequence model the HR module uses
      (<code>main_sequence_mag_from_color</code>). Dashed: model + median offset ± ${f(hr.threshold)} × MAD (${f(hr.scale, 3)} mag), the
      robust z-score limit of the main-sequence test. A star outside the band is also checked by DBSCAN, KDE and GMM; the HR module
      reports it only if at least two of the four agree. Absolute G uses 1/parallax with no extinction correction, so in a low-latitude field such as the demo (b ≈ +7°) interstellar dust moves stars to the right of and below the model line.</p>` : ''}
    ${view === 'sky' ? '<p class="small mute">RA increases to the left, as on the sky.</p>' : ''}`;
  const root = app();
  wireRunBar(root);
  const colors = starColors(stars, cands, dets, mode);
  const lines = [];
  if (view === 'hr') {
    lines.push({ x: hr.bp_rp, y: hr.ms_mag, color: '#e8f1ff' });
    if (hr.center !== null) for (const s of [-1, 1]) lines.push({ x: hr.bp_rp, y: hr.ms_mag.map(m => m + hr.center + s * hr.threshold * hr.scale), color: '#ffd23f', dash: [6, 5] });
  }
  const tier = new Map(cands.map(c => [c.star_id, c]));
  const ids = stars.source_id;
  scatter(root.querySelector('#plot'), {
    x: stars[v.x], y: stars[v.y], colors, order: drawOrder(colors), xlabel: colLabel(v.x), ylabel: colLabel(v.y),
    invertX: v.invertX, invertY: v.invertY, clip: v.clip, lines, aspect: view === 'hr' ? 0.8 : 0.66,
    label: i => `${esc(ids[i])}<br>${esc(colLabel(v.x))} ${f(stars[v.x][i])} · ${esc(colLabel(v.y))} ${f(stars[v.y][i])}${tier.has(ids[i]) ? `<br>#${tier.get(ids[i]).rank}, Tier ${tier.get(ids[i]).tier}` : ''}`,
    onPick: i => go(`#/star/${encodeURIComponent(ids[i])}`),
  });
  root.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => go(`#/plots?view=${b.dataset.view}&color=${mode}`)));
  root.querySelector('#cmode').addEventListener('change', e => go(`#/plots?view=${view}&color=${e.target.value}`));
});
