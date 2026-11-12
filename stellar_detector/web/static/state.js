// Which run the pages show ('demo' unless a local run was started) and cached loaders for its data.
import { api, esc, isStatic } from './core.js';

let current = 'demo';
try { current = sessionStorage.getItem('sad-run') || 'demo'; } catch (e) { /* storage blocked */ }
if (isStatic()) current = 'demo';

export const runId = query => (query && query.run) || current;
export function setRun(id) {
  current = id;
  try { sessionStorage.setItem('sad-run', id); } catch (e) { /* ignore */ }
}

export const getInfo = () => api('/info');
export const getSummary = id => api(`/runs/${id}/summary`);
export const getStars = id => api(`/runs/${id}/stars`);
export const getDetections = id => api(`/runs/${id}/detections`);
export const getCandidates = id => api(`/runs/${id}/candidates`);
export const getStar = (id, sid) => api(`/runs/${id}/star/${encodeURIComponent(sid)}`);

// Small bar naming the run on show, with a way back to the demo field.
export function runBar(sum) {
  const id = sum.source.id;
  return `<div class="runbar"><span>Data:</span><b>${esc(sum.source.title)}</b>
    <span>${sum.n_stars.toLocaleString()} stars after quality cuts</span>
    ${id !== 'demo' ? '<button class="alt" data-demo>Back to the demo field</button>' : ''}</div>`;
}
// The button reloads the current page with ?run=demo (a new hash, so the router re-renders).
export function wireRunBar(root) {
  const b = root.querySelector('[data-demo]');
  if (b) b.addEventListener('click', () => { setRun('demo'); location.hash = location.hash.split('?')[0] + '?run=demo'; });
}

// Index stars by source_id -> row number.
export function starIndex(stars) {
  const m = new Map();
  (stars.source_id || []).forEach((s, i) => m.set(s, i));
  return m;
}
