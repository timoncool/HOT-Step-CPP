// What this guards: semantic_top_k=0 (unlimited) with temperature > 1.0 is
// clamped to 1.0 with a warning; top_k is truncated like the engine (0.5 -> 0);
// an unset temperature falls back to the checkpoint's own value.
// Run: npx tsx src/services/backends/yue2/semanticGuard.test.ts
import { clampYue2SemanticSampling } from './semanticGuard.js';

let fails = 0;
function check(name: string, ov: Record<string, number>, ckpt: number | undefined, wantTemp: number | undefined, wantWarn: boolean) {
  const notes: string[] = [];
  clampYue2SemanticSampling(ov, notes, ckpt);
  const warned = notes.some(n => n.startsWith('WARNING'));
  const ok = ov.semantic_temperature === wantTemp && warned === wantWarn;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}: temp=${ov.semantic_temperature} (want ${wantTemp}) warned=${warned} (want ${wantWarn})`);
  if (!ok) fails++;
}
check('topk0 temp1.5 clamps', { semantic_top_k: 0, semantic_temperature: 1.5 }, undefined, 1.0, true);
check('topk0 temp0.9 untouched', { semantic_top_k: 0, semantic_temperature: 0.9 }, undefined, 0.9, false);
check('topk50 temp1.5 untouched', { semantic_top_k: 50, semantic_temperature: 1.5 }, undefined, 1.5, false);
check('topk0.5 truncates to 0, clamps', { semantic_top_k: 0.5, semantic_temperature: 1.4 }, undefined, 1.0, true);
check('topk0 unset temp, ckpt 1.3 clamps', { semantic_top_k: 0 }, 1.3, 1.0, true);
check('topk0 unset temp, ckpt 0.8 untouched', { semantic_top_k: 0 }, 0.8, undefined, false);
check('topk unset, temp1.5 untouched', { semantic_temperature: 1.5 }, 1.3, 1.5, false);
if (fails) { console.log(`${fails} FAILED`); process.exit(1); }
console.log('all passed');
