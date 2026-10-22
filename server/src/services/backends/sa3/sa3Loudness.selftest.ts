// sa3Loudness.selftest.ts -- unit test for SA3 LUFS normalization
//
//   cd server && npx tsx src/services/backends/sa3/sa3Loudness.selftest.ts
//
// TEETH CHECK: prints actual numbers. A -26 dBFS RMS input must land within
// 1 dB of the target and the peak must stay at or below -1 dBFS.
// The test is first run against a no-op stub to show it fails, then against
// the real normalizeLufs implementation to show it passes.

import os from 'os';
import path from 'path';
import fs from 'fs';
import {
  parseWav, encodeWav, measureLevels, integratedLufs, linToDb, dbToLin,
  type FloatWav,
} from '../../generation/audioLevel.js';
import { normalizeLufs } from '../../generation/lufsNormalize.js';

let failures = 0;
function check(name: string, ok: boolean, detail = ''): void {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  -- ' + detail : ''}`);
  if (!ok) failures++;
}

const SR = 44100; // SA3 native sample rate

/** Make a 5-second stereo 1 kHz sine at a given amplitude. */
function makeTone(ampLin: number): FloatWav {
  const n = SR * 5;
  const l = new Float32Array(n), r = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = ampLin * Math.sin((2 * Math.PI * 1000 * i) / SR);
    l[i] = v; r[i] = v;
  }
  return { channels: [l, r], sampleRate: SR, numChannels: 2, sourceFormat: 'f32' };
}

function writeToTmp(wav: FloatWav): string {
  const p = path.join(os.tmpdir(), `sa3_lufs_test_${Date.now()}.wav`);
  fs.writeFileSync(p, encodeWav(wav, 'f32'));
  return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// Input: a tone at -26 dBFS RMS (amplitude = dbToLin(-26)).
// Target: -14 LUFS.
// Expected: output LUFS within 1 dB of -14, peak <= -1 dBFS (lin <= 0.891).
// ─────────────────────────────────────────────────────────────────────────────

const INPUT_AMP    = dbToLin(-26);   // ~0.0501 linear
const TARGET_LUFS  = -14;
const CEIL_DBFS    = -1;
const CEIL_LIN     = dbToLin(CEIL_DBFS); // ~0.891

const inputWav  = makeTone(INPUT_AMP);
const inputLufs = integratedLufs(inputWav.channels, SR);
const inputPeak = measureLevels(inputWav, { skipTruePeak: true }).samplePeakDb;
console.log(`\nInput: LUFS=${inputLufs.toFixed(2)}, peak=${inputPeak.toFixed(2)} dBFS\n`);

// ── ROUND 1: no-op stub (should FAIL the assertions) ─────────────────────────
console.log('--- Round 1: no-op stub (expected FAILs below) ---');

function noOpNormalize(wavPath: string, _target: number) {
  // Does nothing: the file is unchanged.
  void wavPath;
}

{
  const tmpPath = writeToTmp(makeTone(INPUT_AMP));
  noOpNormalize(tmpPath, TARGET_LUFS);
  const buf = fs.readFileSync(tmpPath);
  const out = parseWav(buf);
  const outLufs = integratedLufs(out.channels, SR);
  const outPeak = measureLevels(out, { skipTruePeak: true }).samplePeak;
  const outPeakDb = linToDb(outPeak);
  console.log(`  stub output: LUFS=${outLufs.toFixed(2)}, peak=${outPeakDb.toFixed(2)} dBFS, peak_lin=${outPeak.toFixed(4)}`);
  const withinTarget = Math.abs(outLufs - TARGET_LUFS) <= 1.0;
  const peakOk = outPeak <= CEIL_LIN;
  check('stub: LUFS within 1 dB of target', withinTarget,
    `got ${outLufs.toFixed(2)}, target ${TARGET_LUFS}`);
  check('stub: peak <= -1 dBFS', peakOk,
    `got ${outPeakDb.toFixed(2)} dBFS`);
  fs.unlinkSync(tmpPath);
}

const stubFailures = failures;
console.log(`\n  (stub failures so far: ${stubFailures} -- both expected)\n`);

// ── ROUND 2: real normalizeLufs (should PASS) ─────────────────────────────────
console.log('--- Round 2: real normalizeLufs (all should PASS) ---');
failures = 0; // reset for clean final count

{
  const tmpPath = writeToTmp(makeTone(INPUT_AMP));
  const result = normalizeLufs(tmpPath, TARGET_LUFS, CEIL_DBFS, { outFormat: 'f32' });
  const buf = fs.readFileSync(tmpPath);
  const out = parseWav(buf);
  const outLufs = integratedLufs(out.channels, SR);
  const outPeak = measureLevels(out, { skipTruePeak: true }).samplePeak;
  const outPeakDb = linToDb(outPeak);

  console.log(`  normalizeLufs result:`);
  console.log(`    measured=${result.measuredLufs.toFixed(2)} LUFS`);
  console.log(`    target=${result.targetLufs} LUFS`);
  console.log(`    gain=${result.appliedGainDb.toFixed(2)} dB`);
  console.log(`    peak_before=${linToDb(result.peakBefore).toFixed(2)} dBFS`);
  console.log(`    peak_after=${linToDb(result.peakAfter).toFixed(2)} dBFS`);
  console.log(`    limiter=${result.limiterActive}`);
  console.log(`  verified output: LUFS=${outLufs.toFixed(2)}, peak=${outPeakDb.toFixed(2)} dBFS, peak_lin=${outPeak.toFixed(4)}`);

  check('measured LUFS reported correctly',
    Math.abs(result.measuredLufs - inputLufs) < 0.2,
    `reported=${result.measuredLufs.toFixed(2)}, actual_input=${inputLufs.toFixed(2)}`);
  check('output LUFS within 1 dB of target',
    Math.abs(outLufs - TARGET_LUFS) <= 1.0,
    `got ${outLufs.toFixed(2)}, target ${TARGET_LUFS}`);
  check('output peak <= -1 dBFS',
    outPeak <= CEIL_LIN + 1e-4,
    `got ${outPeakDb.toFixed(2)} dBFS (limit ${CEIL_DBFS} dBFS)`);
  check('meaningful gain applied (>= 10 dB)',
    result.appliedGainDb >= 10,
    `gain=${result.appliedGainDb.toFixed(2)} dB (input was ${inputLufs.toFixed(1)} LUFS, target ${TARGET_LUFS})`);

  fs.unlinkSync(tmpPath);
}

console.log(`\nRound 2 failures: ${failures} (expected 0)`);
if (failures > 0) process.exit(1);

console.log('\nAll real tests PASS.');
