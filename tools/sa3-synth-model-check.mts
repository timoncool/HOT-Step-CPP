// Live check of /sa3-synth's synth_model resolution against a running engine.
// Needs sa3-dit-BF16.gguf and sa3-dit-Q8_0.gguf in the models root and an idle GPU.
// Run from server/: node --import tsx ../tools/sa3-synth-model-check.mts
import { tokenizeForSa3 } from '../server/src/services/sa3Tokenizer.js';

const ENGINE = process.env.ACE_URL ?? 'http://127.0.0.1:8085';
const { ids, nTokens } = await tokenizeForSa3('warm analog synth pad, slow, ambient');

// [label, synth_model (null = omitted), expected status, expected X-SA3-DiT]
const cases: Array<[string, string | null, number, string | null]> = [
  ['quant', 'sa3-dit-Q8_0.gguf', 200, 'sa3-dit-Q8_0.gguf'],
  ['omitted', null, 200, 'sa3-dit-BF16.gguf'],
  ['empty', '', 200, 'sa3-dit-BF16.gguf'],
  ['not installed', 'sa3-dit-Q2_K.gguf', 200, 'sa3-dit-BF16.gguf'],
  ['not sa3', 'acestep-v15-turbo.gguf', 200, 'sa3-dit-BF16.gguf'],
  ['quant wrong case', 'SA3-DIT-Q8_0.GGUF', 200, 'sa3-dit-BF16.gguf'],
  ['base', 'sa3-dit-base-BF16.gguf', 400, null],
  ['base uppercase', 'sa3-dit-BASE-BF16.gguf', 400, null],
  ['base mixed case', 'SA3-DIT-Base-BF16.GGUF', 400, null],
  ['traversal', '..\\sa3-dit-BF16.gguf', 400, null],
];

let failed = 0;
for (const [label, sm, wantStatus, wantDit] of cases) {
  const p = new URLSearchParams({ tokens: ids.join(','), n_tokens: String(nTokens), duration: '5', steps: '8', seed: '42' });
  if (sm !== null) p.set('synth_model', sm);
  const r = await fetch(`${ENGINE}/sa3-synth?${p}`, { method: 'POST' });
  const body = Buffer.from(await r.arrayBuffer());
  const dit = r.headers.get('x-sa3-dit');
  const ok = r.status === wantStatus && dit === wantDit;
  if (!ok) failed++;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${label.padEnd(16)} ${r.status} X-SA3-DiT=${dit}` +
    (r.ok ? ` bytes=${body.length}` : ` ${body.toString().slice(0, 100)}`));
}
console.log(failed ? `sa3-synth-model-check: ${failed} FAIL` : 'sa3-synth-model-check: PASS');
process.exit(failed ? 1 : 0);
