// TEETH check for the SA3 seed-breeding default blend (ear test b33/b34 picked 0.5).
// Run: npx tsx scripts/check-sa3-seed-blend.ts   (OLD=1 runs the pre-change formula and must fail)
import { resolveSa3SeedBlend, SA3_DEFAULT_SEED_BLEND } from '../src/services/backends/sa3/generate.js';

const oldFormula = (v: unknown) => Math.min(1, Math.max(0, Number(v) || 0));
const f = process.env.OLD === '1' ? oldFormula : resolveSa3SeedBlend;

const cases: Array<[unknown, number]> = [
  [undefined, 0.5], [null, 0.5], ['', 0.5], [0, 0], [0.25, 0.25], ['0.75', 0.75], [2, 1], [-1, 0],
];
let fail = 0;
for (const [input, want] of cases) {
  const got = f(input);
  const ok = Math.abs(got - want) < 1e-9;
  if (!ok) fail++;
  console.log(`${ok ? 'PASS' : 'FAIL'} input=${JSON.stringify(input)} got=${got} want=${want}`);
}
console.log(`SA3_DEFAULT_SEED_BLEND=${SA3_DEFAULT_SEED_BLEND} mode=${process.env.OLD === '1' ? 'old' : 'new'} failures=${fail}`);
process.exit(fail ? 1 : 0);
