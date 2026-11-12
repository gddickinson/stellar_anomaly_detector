// Client-side plots from the run JSON: a canvas scatter (hover + click to open a star) and SVG histograms.
import { esc } from './core.js';
import { f } from './fmt.js';

const finite = a => a.filter(v => v !== null && Number.isFinite(v));
export function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const i = (sorted.length - 1) * q, lo = Math.floor(i);
  return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (i - lo);
}
export function ticks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toPrecision(12));
  return out;
}
function range(vals, clip) {
  const s = finite(vals).sort((a, b) => a - b);
  if (!s.length) return [0, 1];
  let lo = clip ? quantile(s, clip) : s[0], hi = clip ? quantile(s, 1 - clip) : s[s.length - 1];
  if (hi <= lo) { lo -= 0.5; hi += 0.5; }
  const pad = (hi - lo) * 0.04;
  return [lo - pad, hi + pad];
}

// opts: {x, y, colors, order, xlabel, ylabel, invertX, invertY, clip, lines: [{x: [], y: [], color, dash}],
//        highlight, label(i), onPick(i)}. Returns {redraw(colors, highlight)}.
export function scatter(host, o) {
  host.innerHTML = `<div class="plotwrap"><canvas role="img" aria-label="${esc(o.ylabel)} against ${esc(o.xlabel)}; click a point to open that star"></canvas><div class="tip hide"></div></div><p class="small mute note"></p>`;
  const cv = host.querySelector('canvas'), tip = host.querySelector('.tip'), note = host.querySelector('.note');
  const W = 900, H = Math.round(W * (o.aspect || 0.66)), M = { l: 58, r: 14, t: 12, b: 44 };
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext('2d'); g.scale(dpr, dpr);
  const [x0, x1] = range(o.x, o.clip), [y0, y1] = range(o.y, o.clip);
  const X = v => (o.invertX ? W - M.r - (v - x0) / (x1 - x0) * (W - M.l - M.r) : M.l + (v - x0) / (x1 - x0) * (W - M.l - M.r));
  const Y = v => (o.invertY ? M.t + (v - y0) / (y1 - y0) * (H - M.t - M.b) : H - M.b - (v - y0) / (y1 - y0) * (H - M.t - M.b));
  const inside = i => o.x[i] !== null && o.y[i] !== null && o.x[i] >= x0 && o.x[i] <= x1 && o.y[i] >= y0 && o.y[i] <= y1;
  let colors = o.colors, highlight = o.highlight ?? null;
  const order = o.order || o.x.map((_, i) => i);

  function draw() {
    g.fillStyle = '#03040a'; g.fillRect(0, 0, W, H);
    g.strokeStyle = '#212c4c'; g.lineWidth = 1; g.fillStyle = '#94a3c6'; g.font = '12px ui-monospace, Menlo, monospace';
    g.beginPath(); g.moveTo(M.l, M.t); g.lineTo(M.l, H - M.b); g.lineTo(W - M.r, H - M.b); g.stroke();
    g.textAlign = 'center';
    for (const t of ticks(x0, x1, 7)) g.fillText(f(t, 3), X(t), H - M.b + 16);
    g.fillText(o.xlabel, (M.l + W - M.r) / 2, H - 8);
    g.textAlign = 'right';
    for (const t of ticks(y0, y1, 6)) g.fillText(f(t, 3), M.l - 6, Y(t) + 4);
    g.save(); g.translate(14, (M.t + H - M.b) / 2); g.rotate(-Math.PI / 2); g.textAlign = 'center'; g.fillText(o.ylabel, 0, 0); g.restore();
    g.save(); g.beginPath(); g.rect(M.l, M.t, W - M.l - M.r, H - M.t - M.b); g.clip();
    for (const ln of o.lines || []) {
      g.strokeStyle = ln.color; g.setLineDash(ln.dash || []); g.lineWidth = 1.4; g.beginPath();
      ln.x.forEach((xv, k) => { const px = X(xv), py = Y(ln.y[k]); k ? g.lineTo(px, py) : g.moveTo(px, py); });
      g.stroke(); g.setLineDash([]);
    }
    for (const i of order) {
      if (!inside(i)) continue;
      g.fillStyle = colors[i]; g.globalAlpha = colors[i] === '#3a4566' ? 0.75 : 0.95;
      g.beginPath(); g.arc(X(o.x[i]), Y(o.y[i]), colors[i] === '#3a4566' ? 2 : 3.2, 0, 2 * Math.PI); g.fill();
    }
    g.globalAlpha = 1;
    if (highlight !== null && inside(highlight)) {
      g.strokeStyle = '#ffffff'; g.lineWidth = 2; g.beginPath(); g.arc(X(o.x[highlight]), Y(o.y[highlight]), 9, 0, 2 * Math.PI); g.stroke();
    }
    g.restore();
    const n = o.x.filter((_, i) => o.x[i] !== null && o.y[i] !== null).length, shown = o.x.filter((_, i) => inside(i)).length;
    note.textContent = `${shown.toLocaleString()} of ${n.toLocaleString()} stars with both values plotted` + (shown < n ? ' (axis range clipped to the central 99.6%)' : '') + '.';
  }
  function nearest(ev) {
    const r = cv.getBoundingClientRect(), mx = (ev.clientX - r.left) * W / r.width, my = (ev.clientY - r.top) * H / r.height;
    let best = null, bd = 100;
    for (let i = 0; i < o.x.length; i++) {
      if (!inside(i)) continue;
      const d = (X(o.x[i]) - mx) ** 2 + (Y(o.y[i]) - my) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    return { i: best, px: ev.clientX - r.left, py: ev.clientY - r.top };
  }
  cv.addEventListener('mousemove', ev => {
    const { i, px, py } = nearest(ev);
    if (i === null) { tip.classList.add('hide'); return; }
    tip.innerHTML = o.label ? o.label(i) : String(i);
    tip.classList.remove('hide');
    tip.style.left = Math.min(px + 12, cv.clientWidth - tip.offsetWidth - 4) + 'px'; tip.style.top = (py + 12) + 'px';
  });
  cv.addEventListener('mouseleave', () => tip.classList.add('hide'));
  cv.addEventListener('click', ev => { const { i } = nearest(ev); if (i !== null && o.onPick) o.onPick(i); });
  draw();
  return { redraw(c, h) { if (c) colors = c; if (h !== undefined) highlight = h; draw(); } };
}

// SVG histogram of values (log10 optional). marks: [{x, label}] as dashed lines.
export function histogram(values, { xlabel, bins = 30, marks = [], log = false } = {}) {
  const W = 360, H = 220, M = { l: 40, r: 12, t: 12, b: 38 };
  let v = finite(values);
  if (log) v = v.filter(x => x > 0).map(Math.log10);
  v.sort((a, b) => a - b);
  if (!v.length) return '<p class="mute small">No values.</p>';
  let lo = v[0], hi = v[v.length - 1];
  if (hi <= lo) { lo -= 0.5; hi += 0.5; }
  const counts = new Array(bins).fill(0), w = (hi - lo) / bins;
  for (const x of v) counts[Math.min(bins - 1, Math.floor((x - lo) / w))]++;
  const cmax = Math.max(...counts);
  const X = x => M.l + (x - lo) / (hi - lo) * (W - M.l - M.r), Y = c => H - M.b - c / cmax * (H - M.t - M.b);
  let s = `<line class="axis" x1="${M.l}" y1="${H - M.b}" x2="${W - M.r}" y2="${H - M.b}"/><line class="axis" x1="${M.l}" y1="${M.t}" x2="${M.l}" y2="${H - M.b}"/>`;
  for (const t of ticks(lo, hi, 5)) s += `<text x="${X(t)}" y="${H - M.b + 14}" text-anchor="middle">${log ? f(10 ** t, 2) : f(t, 3)}</text>`;
  for (const t of ticks(0, cmax, 4)) s += `<text x="${M.l - 5}" y="${Y(t) + 4}" text-anchor="end">${t}</text>`;
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 4}" text-anchor="middle">${esc(xlabel)}${log ? ' (log scale)' : ''}</text>`;
  counts.forEach((c, i) => { if (c) s += `<rect class="bar" x="${X(lo + i * w) + 0.5}" y="${Y(c)}" width="${Math.max(0.5, X(lo + w) - X(lo) - 1)}" height="${Y(0) - Y(c)}"><title>${f(log ? 10 ** (lo + i * w) : lo + i * w, 3)}–${f(log ? 10 ** (lo + (i + 1) * w) : lo + (i + 1) * w, 3)}: ${c}</title></rect>`; });
  for (const m of marks) {
    const mx = log ? Math.log10(m.x) : m.x;
    if (mx < lo || mx > hi) continue;
    s += `<line x1="${X(mx)}" x2="${X(mx)}" y1="${M.t}" y2="${H - M.b}" stroke="#ffd23f" stroke-dasharray="4 3"/><text x="${X(mx) + 4}" y="${M.t + 10}" style="fill:#ffd23f">${esc(m.label)}</text>`;
  }
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Histogram of ${esc(xlabel)}">${s}</svg>`;
}
