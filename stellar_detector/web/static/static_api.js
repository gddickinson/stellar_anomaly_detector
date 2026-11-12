// Static-site data layer: answers api() calls from the JSON exported by scripts/build_static.py (the demo field,
// analysed by the Python pipeline at build time). New runs and uploads need the local Python server.
const files = new Map();
function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export const NEEDS_SERVER = 'needs the full app running locally (Python server): see About, "Run the web app"';

export async function staticApi(path, body) {
  const clean = path.split('?')[0].replace(/^\/+/, '');
  if (body || clean.startsWith('run')) {
    if (clean.startsWith('runs/demo/')) return file(clean);
    throw new Error('Fetching catalogues and analysing new data ' + NEEDS_SERVER + '.');
  }
  return file(clean);
}
