// Formatting, labels and colours shared by the pages.
import { esc } from './core.js';

export const TIER_COLORS = { 1: '#ff5f8f', 2: '#ffc857', 3: '#6fb6ff', raw: '#9b8cd9', none: '#3a4566' };
export const TIER_NAMES = {
  1: 'Tier 1: anomalous in 3+ independent families', 2: 'Tier 2: one type confirmed by 2+ methods',
  3: 'Tier 3: one strong single-method detection', raw: 'flagged by a module, not selected', none: 'not flagged',
};
export const MODULE_COLORS = {
  hr: '#ffc857', lifetime: '#ff8a5c', kinematics: '#6fb6ff', spectral: '#3dff8f', technosignature: '#ff5f8f', ml: '#b48cff',
};
export const MODULE_NAMES = {
  hr: 'HR diagram', lifetime: 'Stellar lifetime', kinematics: 'Kinematics', spectral: 'Spectral',
  technosignature: 'Technosignature', ml: 'ML pipeline',
};

export const nice = s => String(s ?? '').replace(/_/g, ' ');
export const dot = (label, color) => `<span class="tier" style="--c:${color}">${esc(label)}</span>`;
export const tierDot = t => dot(t === 'raw' || t === 'none' ? TIER_NAMES[t] : `Tier ${t}`, TIER_COLORS[t]);

// Number formatting: null/NaN -> '–', otherwise `digits` significant digits (integers as is).
export function f(v, digits = 4) {
  if (v === null || v === undefined || v === '' || Number.isNaN(+v)) return '–';
  const x = +v;
  if (!Number.isFinite(x)) return x > 0 ? '∞' : '-∞';
  if (Number.isInteger(x) && Math.abs(x) < 1e7) return String(x);
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e7)) return x.toExponential(digits - 1);
  return String(Number(x.toPrecision(digits)));
}

// Column labels and units for star properties.
export const COLS = {
  source_id: ['Source ID', ''], ra: ['RA', 'deg'], dec: ['Dec', 'deg'], parallax: ['Parallax', 'mas'],
  parallax_error: ['Parallax error', 'mas'], distance_pc: ['Distance (1/parallax)', 'pc'], pmra: ['pmRA', 'mas/yr'],
  pmdec: ['pmDec', 'mas/yr'], pm_total: ['Total proper motion', 'mas/yr'], v_tan_km_s: ['Tangential velocity', 'km/s'],
  phot_g_mean_mag: ['G', 'mag'], bp_rp: ['BP−RP', 'mag'], abs_mag: ['Absolute G', 'mag'],
  luminosity_solar: ['Luminosity (from M_G)', 'L☉'], teff_gspphot: ['Teff (GSP-Phot)', 'K'],
  logg_gspphot: ['log g (GSP-Phot)', 'dex'], mh_gspphot: ['[M/H] (GSP-Phot)', 'dex'], ruwe: ['RUWE', ''],
  astrometric_excess_noise: ['Astrometric excess noise', 'mas'],
  astrometric_excess_noise_sig: ['Excess noise significance', ''], phot_bp_rp_excess_factor: ['BP/RP excess factor', ''],
  radial_velocity: ['Radial velocity', 'km/s'], quality_score: ['Quality score', '0–1'],
  Jmag: ['J', 'mag'], Hmag: ['H', 'mag'], Kmag: ['K', 'mag'], W1mag: ['W1', 'mag'], W2mag: ['W2', 'mag'],
  W3mag: ['W3', 'mag'], W4mag: ['W4', 'mag'], J_H: ['J−H', 'mag'], H_K: ['H−K', 'mag'], W1_W2: ['W1−W2', 'mag'],
  W2_W3: ['W2−W3', 'mag'], is_injected_anomaly: ['Injected anomaly (synthetic)', ''],
};
export const colLabel = c => (COLS[c] ? COLS[c][0] + (COLS[c][1] ? ` (${COLS[c][1]})` : '') : c);

export function toCsv(rows, cols) {
  const q = v => { const s = v === null || v === undefined ? '' : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  return [cols.join(','), ...rows.map(r => cols.map(c => q(r[c])).join(','))].join('\n') + '\n';
}
