// node stellar_detector/web/static/ensemble_engine.test.mjs
// The JS ensemble must reproduce the Python EnsembleScorer on the demo detections (tests/fixtures/ensemble_fixture.json).
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { aggregate } from './ensemble_engine.js';

const fx = JSON.parse(readFileSync(new URL('../../../tests/fixtures/ensemble_fixture.json', import.meta.url)));
const close = (a, b, what) => assert.ok(Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b)), `${what}: ${a} vs ${b}`);

for (const c of fx.cases) {
  const o = { ...c.opts };
  if (o.modules) o.modules = new Set(o.modules);
  const got = aggregate(fx.detections, o);
  assert.equal(got.length, c.expected.length, `${c.name}: number of candidates`);
  c.expected.forEach((e, i) => {
    const g = got[i], at = `${c.name} #${i + 1} ${e.star_id}`;
    for (const k of ['star_id', 'tier', 'anomaly_type', 'follow_up_priority', 'detection_method', 'description']) assert.equal(g[k], e[k], `${at} ${k}`);
    close(g.confidence, e.confidence, `${at} confidence`);
    close(g.significance_score, e.significance_score, `${at} significance`);
  });
  console.log(`ok ${c.name}: ${got.length} candidates`);
}
