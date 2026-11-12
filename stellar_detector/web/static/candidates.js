// Candidates: the EnsembleScorer's ranked selection, filterable by tier and type, sortable, exportable as CSV.
import { app, download, esc, go, route } from './core.js';
import { TIER_COLORS, TIER_NAMES, dot, f, nice, tierDot, toCsv } from './fmt.js';
import { getCandidates, getSummary, runBar, runId, wireRunBar } from './state.js';

const COLS = [
  ['rank', '#'], ['star_id', 'Star'], ['tier', 'Tier'], ['anomaly_type', 'Type'], ['follow_up_priority', 'Priority'],
  ['significance_score', 'Significance'], ['confidence', 'Confidence'], ['description', 'Description'],
];
const PAGE = 40;

route('candidates', async (args, query) => {
  const id = runId(query);
  const [sum, cands] = await Promise.all([getSummary(id), getCandidates(id)]);
  const types = [...new Set(cands.map(c => c.anomaly_type))].sort();
  const st = { tier: '', type: '', sort: 'rank', dir: 1, page: 0 };
  app().innerHTML = `${runBar(sum)}<h1>Ranked candidates</h1>
    <p class="lead">${cands.length} of ${sum.n_stars.toLocaleString()} stars selected by the ensemble scorer, ranked by follow-up
      priority, then significance (as <code>EnsembleScorer.aggregate</code> orders them). ${sum.n_flagged_stars} stars had at least one raw detection.</p>
    <div class="legend">${[1, 2, 3].map(t => dot(TIER_NAMES[t] + ` (${sum.tiers[t] || 0})`, TIER_COLORS[t])).join('')}</div>
    <p class="small mute">Significance is not on one scale across modules: for z-score tests it is the robust |z| (in σ), for the lifetime
      test it is 10 × the fraction of the main-sequence band crossed, for the ML pipeline 10 × the ensemble score. See Method.</p>
    <div class="row"><div><label for="ftier">Tier</label><select id="ftier"><option value="">all</option>${[1, 2, 3].map(t => `<option value="${t}">Tier ${t}</option>`).join('')}</select></div>
      <div><label for="ftype">Type</label><select id="ftype"><option value="">all</option>${types.map(t => `<option value="${esc(t)}">${esc(nice(t))}</option>`).join('')}</select></div>
      <div style="flex:0"><button id="csv" class="alt">Download CSV</button></div></div>
    <div id="tbl"></div>`;
  const root = app();
  wireRunBar(root);
  const rows = () => cands.filter(c => (!st.tier || String(c.tier) === st.tier) && (!st.type || c.anomaly_type === st.type))
    .sort((a, b) => { const x = a[st.sort], y = b[st.sort]; return (typeof x === 'number' ? x - y : String(x).localeCompare(String(y))) * st.dir; });
  function draw() {
    const r = rows(), pages = Math.max(1, Math.ceil(r.length / PAGE));
    st.page = Math.min(st.page, pages - 1);
    const slice = r.slice(st.page * PAGE, (st.page + 1) * PAGE);
    root.querySelector('#tbl').innerHTML = `<div class="scroll"><table class="tbl"><thead><tr>${COLS.map(([k, l]) => `<th><button data-sort="${k}">${l}${st.sort === k ? (st.dir > 0 ? ' ▲' : ' ▼') : ''}</button></th>`).join('')}</tr></thead>
      <tbody>${slice.map(c => `<tr class="click" tabindex="0" data-id="${esc(c.star_id)}"><td>${c.rank}</td><td class="nw">${esc(c.star_id)}</td><td>${tierDot(c.tier)}</td>
        <td class="nw">${esc(nice(c.anomaly_type))}</td><td>${c.follow_up_priority}</td><td>${f(c.significance_score, 3)}</td><td>${f(c.confidence, 2)}</td><td class="desc">${esc(c.description)}</td></tr>`).join('')}</tbody></table></div>
      <div class="pager"><button class="alt" data-p="-1" ${st.page ? '' : 'disabled'}>Previous</button><span class="small mute">${r.length} rows · page ${st.page + 1} of ${pages}</span><button class="alt" data-p="1" ${st.page < pages - 1 ? '' : 'disabled'}>Next</button></div>`;
    root.querySelectorAll('[data-sort]').forEach(b => b.addEventListener('click', () => {
      st.dir = st.sort === b.dataset.sort ? -st.dir : (b.dataset.sort === 'rank' ? 1 : -1); st.sort = b.dataset.sort; draw();
    }));
    root.querySelectorAll('[data-p]').forEach(b => b.addEventListener('click', () => { st.page += +b.dataset.p; draw(); }));
    root.querySelectorAll('tr[data-id]').forEach(tr => {
      const open = () => go(`#/star/${encodeURIComponent(tr.dataset.id)}`);
      tr.addEventListener('click', open);
      tr.addEventListener('keydown', e => { if (e.key === 'Enter') open(); });
    });
  }
  root.querySelector('#ftier').addEventListener('change', e => { st.tier = e.target.value; st.page = 0; draw(); });
  root.querySelector('#ftype').addEventListener('change', e => { st.type = e.target.value; st.page = 0; draw(); });
  root.querySelector('#csv').addEventListener('click', () => download(`candidates_${id}.csv`,
    toCsv(rows(), ['rank', 'star_id', 'tier', 'anomaly_type', 'follow_up_priority', 'significance_score', 'confidence', 'detection_method', 'description']), 'text/csv'));
  draw();
});
