// Port of analysis/ensemble.py (EnsembleScorer.aggregate / _evaluate_star) to JS, so the static site can re-apply the
// ensemble rules to the precomputed raw detections with other settings. Tested against the Python code on the demo
// detections (ensemble_engine.test.mjs, fixture from scripts/make_ensemble_fixture.py).
//
// detections: [{star_id, anomaly_type, family, detection_method, confidence, significance_score, follow_up_priority,
//               description, module}]
// opts: {modules: Set|null, tier1_min_families, tier2_min_methods, single_min_significance,
//        high_importance_types: [...], high_importance_min_confidence}

export const DEFAULTS = {
  tier1_min_families: 3, tier2_min_methods: 2, single_min_significance: 6.0,
  high_importance_types: ['dyson_sphere_candidate', 'infrared_excess', 'megastructure_candidate', 'stellar_engine_candidate'],
  high_importance_min_confidence: 0.7,
};

// Python max(key=...) keeps the first maximal element.
const best = ds => ds.reduce((b, d) => (d.significance_score > b.significance_score ? d : b));
const pySorted = xs => [...new Set(xs)].sort();

function groupBy(xs, key) {
  const m = new Map();
  for (const x of xs) { const k = key(x); if (!m.has(k)) m.set(k, []); m.get(k).push(x); }
  return m;
}

function tier1(sid, dets, families) {
  const b = best(dets), fam = pySorted(families), n = fam.length, methods = pySorted(dets.map(d => d.detection_method));
  return {
    star_id: sid, tier: 1, anomaly_type: 'cross_catalog_anomaly', family: 'multi',
    confidence: Math.min(1, b.confidence + 0.1 * (n - 2)),
    significance_score: b.significance_score * (1 + 0.15 * (n - 2)),
    follow_up_priority: Math.min(10, b.follow_up_priority + n - 2),
    detection_method: `tier1_ensemble(${methods.slice(0, 3).join(', ')}...)`,
    description: `Multi-dimensional anomaly across ${n} independent families (${fam.join(', ')})`,
    families: fam,
  };
}

function tier2(sid, type, dets, methods) {
  const b = best(dets), m = pySorted(methods), n = m.length;
  return {
    star_id: sid, tier: 2, anomaly_type: type, family: b.family,
    confidence: Math.min(1, b.confidence + 0.05 * (n - 1)),
    significance_score: b.significance_score * (1 + 0.1 * (n - 1)),
    follow_up_priority: Math.min(10, b.follow_up_priority + n - 1),
    detection_method: `tier2_ensemble(${m.join(', ')})`,
    description: `${b.description} [confirmed by ${n} methods: ${m.join(', ')}]`,
  };
}

// EnsembleScorer._evaluate_star: returns a candidate or null.
export function evaluateStar(sid, dets, o = DEFAULTS) {
  const families = new Set(dets.map(d => d.family));
  if (families.size >= o.tier1_min_families) return tier1(sid, dets, families);
  for (const [type, td] of groupBy(dets, d => d.anomaly_type)) {
    const methods = new Set(td.map(d => d.detection_method));
    if (methods.size >= o.tier2_min_methods) return tier2(sid, type, td, methods);
  }
  const b = best(dets);
  const keep = (o.high_importance_types.includes(b.anomaly_type) && b.confidence >= o.high_importance_min_confidence)
    || b.significance_score >= o.single_min_significance;
  return keep ? { ...b, tier: 3, families: undefined } : null;
}

// EnsembleScorer.aggregate with tiers, ranked by (priority, significance) descending (stable, like Python's sort).
export function aggregate(detections, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const use = o.modules ? detections.filter(d => o.modules.has(d.module)) : detections;
  const out = [];
  for (const [sid, dets] of groupBy(use, d => d.star_id)) {
    const r = evaluateStar(sid, dets, o);
    if (r) out.push(r);
  }
  out.sort((a, b) => (b.follow_up_priority - a.follow_up_priority) || (b.significance_score - a.significance_score));
  out.forEach((r, i) => { r.rank = i + 1; });
  return out;
}
