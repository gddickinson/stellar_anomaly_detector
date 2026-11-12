// Entry point: lazy page modules, the mode label, a faint twinkling star field, then the router.
import { $, app, isStatic, lazy, route, start } from './core.js';

const PAGES = {
  'home.js': ['home'], 'run.js': ['run'], 'plots.js': ['plots'], 'candidates.js': ['candidates'],
  'scores.js': ['scores'], 'star.js': ['star'], 'docs.js': ['method', 'about', 'readme'],
};
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));
route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: the demo field was analysed by the Python pipeline at build time; ensemble rules re-applied in your browser'
  : 'local app: analysis runs in Python on this computer';

function background(canvas) {
  if (!canvas || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const g = canvas.getContext('2d');
  let stars = [];
  const fit = () => {
    canvas.width = innerWidth; canvas.height = innerHeight;
    stars = Array.from({ length: Math.round(innerWidth * innerHeight / 9000) }, () => ({
      x: Math.random() * innerWidth, y: Math.random() * innerHeight, r: Math.random() * 1.2 + 0.2,
      p: Math.random() * 6.28, s: 0.3 + Math.random(), c: Math.random() < 0.12 ? '#ffc857' : Math.random() < 0.1 ? '#6fb6ff' : '#dfe8ff' }));
  };
  fit(); addEventListener('resize', fit);
  let last = 0;
  const tick = t => {
    requestAnimationFrame(tick);
    if (document.hidden || t - last < 80) return;
    last = t;
    g.clearRect(0, 0, canvas.width, canvas.height);
    for (const s of stars) {
      g.globalAlpha = 0.35 + 0.35 * Math.sin(s.p + t / 1000 * s.s);
      g.fillStyle = s.c; g.beginPath(); g.arc(s.x, s.y, s.r, 0, 6.28); g.fill();
    }
  };
  requestAnimationFrame(tick);
}
background($('#bg'));
start();
