/**
 * Self-test for the Compare studio analyzer (server/src/services/compare).
 * Run from server/:  npx tsx scripts/test-analyzer.ts
 *
 * Synthesizes signals in memory, plus one end-to-end WAV through ffmpeg decode.
 * Ends with a teeth check: loudness tests must FAIL with K-weighting bypassed
 * and the true-peak test must FAIL with oversampling bypassed.
 */
import fs from 'fs';
import os from 'os';
import path from 'path';
import { analyze, analyzeSamples, diffTraceSamples, ANALYZER_VERSION, type AnalyzeOptions } from '../src/services/compare/analyzer.js';

const FS = 48000;
const dbToAmp = (db: number) => Math.pow(10, db / 20);

function sine(sec: number, freq: number, peakDb: number, phase = 0): Float32Array {
    const n = Math.round(sec * FS), a = dbToAmp(peakDb), x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = a * Math.sin((2 * Math.PI * freq * i) / FS + phase);
    return x;
}
let seed = 12345;
function rand(): number { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; }
function noise(sec: number, amp = 0.25): Float32Array {
    const x = new Float32Array(Math.round(sec * FS));
    for (let i = 0; i < x.length; i++) x[i] = amp * (2 * rand() - 1);
    return x;
}
const concat = (a: Float32Array, b: Float32Array) => { const o = new Float32Array(a.length + b.length); o.set(a); o.set(b, a.length); return o; };
const neg = (a: Float32Array) => a.map((v) => -v);

interface Row { name: string; expected: string; actual: number | null; pass: boolean }
const rows: Row[] = [];
function check(name: string, actual: number | null, lo: number, hi: number, expected: string, sink = rows): boolean {
    const pass = actual !== null && Number.isFinite(actual) && actual >= lo && actual <= hi;
    sink.push({ name, expected, actual, pass });
    return pass;
}
const near = (name: string, actual: number | null, target: number, tol: number, sink = rows) =>
    check(name, actual, target - tol, target + tol, `${target} +-${tol}`, sink);
function print(title: string, list: Row[]) {
    console.log(`\n== ${title}`);
    for (const r of list) {
        const a = r.actual === null ? 'null' : r.actual.toFixed(4);
        console.log(`${r.pass ? 'PASS' : 'FAIL'}  ${r.name.padEnd(42)} expected ${r.expected.padEnd(16)} actual ${a}`);
    }
}

function loudnessSuite(opts: AnalyzeOptions, sink: Row[]) {
    const s23 = sine(20, 1000, -23);
    const r23 = analyzeSamples(s23, s23, opts);
    near('sine -23 dBFS integrated LUFS', r23.integratedLufs, -23, 0.1, sink);
    near('sine -23 short-term max LUFS', r23.shortTermMaxLufs, -23, 0.1, sink);
    const s33 = sine(20, 1000, -33);
    near('sine -33 dBFS integrated LUFS', analyzeSamples(s33, s33, opts).integratedLufs, -33, 0.1, sink);
    const g = concat(sine(20, 1000, -36), sine(20, 1000, -23));
    near('-36 then -23 (relative gate) LUFS', analyzeSamples(g, g, opts).integratedLufs, -23, 0.2, sink);
}
// Short files: under 400 ms no gating block exists, so integrated must be null;
// under 3 s no full short-term window exists, so short-term max must be null.
function shortSuite(sink: Row[]) {
    const isNull = (name: string, v: number | null) => { sink.push({ name, expected: 'null', actual: v, pass: v === null }); };
    const s200 = sine(0.2, 1000, -23);
    const r200 = analyzeSamples(s200, s200);
    isNull('200 ms sine integrated LUFS', r200.integratedLufs);
    isNull('200 ms sine short-term max', r200.shortTermMaxLufs);
    const s2 = sine(2, 1000, -23);
    const r2 = analyzeSamples(s2, s2);
    near('2 s sine integrated LUFS', r2.integratedLufs, -23, 0.2, sink);
    isNull('2 s sine short-term max', r2.shortTermMaxLufs);
}
function truePeakSuite(opts: AnalyzeOptions, sink: Row[]) {
    const s = sine(5, FS / 4, 0, Math.PI / 4);
    const r = analyzeSamples(s, s, opts);
    near('fs/4 45deg sample peak dBFS', r.samplePeakDbfs, -3.01, 0.05, sink);
    near('fs/4 45deg true peak dBTP', r.truePeakDbtp, 0, 0.2, sink);
}

async function main() {
    console.log(`analyzer version ${ANALYZER_VERSION}`);
    loudnessSuite({}, rows);
    truePeakSuite({}, rows);

    const z = new Float32Array(FS * 5);
    const sil = analyzeSamples(z, z);
    rows.push({ name: 'silence integrated is null', expected: 'null', actual: sil.integratedLufs, pass: sil.integratedLufs === null && sil.truePeakDbtp === null && sil.rmsDbfs === null });

    const s1k = sine(10, 1000, -6);
    const r1k = analyzeSamples(s1k, s1k);
    near('1 kHz centroid Hz', r1k.centroidHz, 1000, 20);
    check('1 kHz flatness', r1k.flatness, 0, 0.01, '< 0.01');
    near('sine RMS dBFS (plain RMS, -6 peak)', r1k.rmsDbfs, -9.01, 0.05);
    near('sine crest dB', r1k.crestDb, 3.01, 0.05);
    const wn = noise(10);
    check('white noise flatness', analyzeSamples(wn, noise(10)).flatness, 0.5, 0.62, '0.5 to 0.62');

    near('corr L=R', analyzeSamples(s1k, s1k).stereoCorr, 1, 0.005);
    near('corr L=-R', analyzeSamples(s1k, neg(s1k)).stereoCorr, -1, 0.005);
    near('corr independent noise', analyzeSamples(noise(10), noise(10)).stereoCorr, 0, 0.05);

    const dt = diffTraceSamples({ left: s1k, right: s1k }, { left: sine(10, 1000, -12), right: sine(10, 1000, -12) });
    near('diffTrace delta (-12 vs -6) LU', dt.loudnessDeltaLu[10], -6, 0.05);
    check('diffTrace LSD length', dt.logSpectralDistanceDb.length, 20, 20, '20');
    const nz = noise(10), nzHalf = nz.map((v) => v * 0.5);
    const dn = diffTraceSamples({ left: nz, right: nz }, { left: nzHalf, right: nzHalf });
    near('diffTrace LSD (noise, -6 dB gain) dB', dn.logSpectralDistanceDb[10], 6.02, 0.05);

    // End-to-end: write a float32 stereo WAV and decode it through ffmpeg.
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hs-analyzer-test-'));
    const wav = path.join(dir, 'sine.wav');
    const e2e = sine(10, 1000, -23);
    const buf = Buffer.alloc(44 + e2e.length * 8);
    buf.write('RIFF', 0); buf.writeUInt32LE(36 + e2e.length * 8, 4); buf.write('WAVE', 8);
    buf.write('fmt ', 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(3, 20); buf.writeUInt16LE(2, 22);
    buf.writeUInt32LE(FS, 24); buf.writeUInt32LE(FS * 8, 28); buf.writeUInt16LE(8, 32); buf.writeUInt16LE(32, 34);
    buf.write('data', 36); buf.writeUInt32LE(e2e.length * 8, 40);
    for (let i = 0; i < e2e.length; i++) { buf.writeFloatLE(e2e[i], 44 + i * 8); buf.writeFloatLE(e2e[i], 48 + i * 8); }
    fs.writeFileSync(wav, buf);
    const re = await analyze(wav);
    near('e2e WAV integrated LUFS', re.integratedLufs, -23, 0.1);
    near('e2e WAV duration s', re.durationSec, 10, 0.01);
    check('e2e WAV peak buckets', re.peaks.max.length, 2000, 2000, '2000');
    fs.rmSync(dir, { recursive: true, force: true });

    // 3-minute timing (in-memory analysis only).
    const long = noise(180);
    const t0 = Date.now();
    analyzeSamples(long, noise(180));
    console.log(`3-min analyzeSamples time: ${Date.now() - t0} ms`);

    shortSuite(rows);
    print('Main tests', rows);

    const teethK: Row[] = [], teethO: Row[] = [];
    loudnessSuite({ bypassKWeighting: true }, teethK);
    truePeakSuite({ bypassOversampling: true }, teethO);
    print('TEETH: K-weighting bypassed (loudness should FAIL)', teethK);
    print('TEETH: oversampling bypassed (true peak should FAIL)', teethO);

    const mainOk = rows.every((r) => r.pass);
    const teethOk = teethK.every((r) => !r.pass) && !teethO[1].pass;
    console.log(`\nmain: ${mainOk ? 'PASS' : 'FAIL'}   teeth: ${teethOk ? 'PASS (bypassed variants fail)' : 'FAIL'}`);
    process.exit(mainOk && teethOk ? 0 : 1);
}
main().catch((e) => { console.error(e); process.exit(1); });
