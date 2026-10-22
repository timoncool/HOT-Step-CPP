/**
 * Clean-room audio analyzer for the Compare studio.
 *
 * Implemented from public standards:
 *  - ITU-R BS.1770-4 loudness (K-weighting, gating) and true peak (Annex 2)
 *  - EBU Tech 3341 short-term loudness (3 s window)
 *
 * Conventions:
 *  - rmsDbfs uses plain RMS relative to digital full scale (a full-scale sine
 *    reads -3.01 dBFS, NOT the AES17 +3.01 dB convention). crestDb is
 *    samplePeakDbfs - rmsDbfs, so a sine has a crest of about 3.01 dB.
 *  - flatness is the spectral flatness (geomean/mean of the power spectrum)
 *    computed per 2048-pt Hann frame and averaged over frames. centroidHz is
 *    taken from the frame-averaged power spectrum.
 *  - Values that are undefined for digital silence are returned as null.
 */
import { decodeFile, DECODE_RATE } from './decode.js';

export const ANALYZER_VERSION = '1.0.0';

export interface AnalyzeOptions {
    /** Test-only: skip K-weighting (loudness becomes plain mean-square). */
    bypassKWeighting?: boolean;
    /** Test-only: true peak uses raw samples instead of 4x oversampling. */
    bypassOversampling?: boolean;
}

export interface PeakBuckets {
    min: number[];
    max: number[];
}

export interface AnalysisResult {
    analyzerVersion: string;
    integratedLufs: number | null;
    shortTermMaxLufs: number | null;
    truePeakDbtp: number | null;
    samplePeakDbfs: number | null;
    rmsDbfs: number | null;
    crestDb: number | null;
    centroidHz: number | null;
    flatness: number | null;
    stereoCorr: number | null;
    durationSec: number;
    peaks: PeakBuckets;
}

export interface DiffTrace {
    hopSec: number;
    /** Short-term loudness of b minus a, per 500 ms (null where either is silent). */
    loudnessDeltaLu: (number | null)[];
    /** Log-spectral distance in dB, per 500 ms segment. */
    logSpectralDistanceDb: number[];
}

export interface StereoSamples {
    left: Float32Array;
    right: Float32Array;
}

const FS = DECODE_RATE;
const SUB_BLOCK = FS / 10; // 100 ms gating step
const FFT_SIZE = 2048;
const FFT_HOP = 1024;
const PEAK_BUCKETS = 2000;
const EPS = 1e-20;

const toDb = (v: number): number | null => (v > 0 ? 20 * Math.log10(v) : null);

// ---------------------------------------------------------------- K-weighting
// BS.1770-4 published coefficients for 48 kHz.
const PRE_B = [1.53512485958697, -2.69169618940638, 1.19839281085285];
const PRE_A = [-1.69065929318241, 0.73248077421585];
const RLB_B = [1.0, -2.0, 1.0];
const RLB_A = [-1.99004745483398, 0.99007225036621];

function kWeight(x: Float32Array): Float64Array {
    const y = new Float64Array(x.length);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0; // pre-filter state
    let u1 = 0, u2 = 0, v1 = 0, v2 = 0; // RLB state
    for (let i = 0; i < x.length; i++) {
        const xi = x[i];
        const yi = PRE_B[0] * xi + PRE_B[1] * x1 + PRE_B[2] * x2 - PRE_A[0] * y1 - PRE_A[1] * y2;
        x2 = x1; x1 = xi; y2 = y1; y1 = yi;
        const vi = RLB_B[0] * yi + RLB_B[1] * u1 + RLB_B[2] * u2 - RLB_A[0] * v1 - RLB_A[1] * v2;
        u2 = u1; u1 = yi; v2 = v1; v1 = vi;
        y[i] = vi;
    }
    return y;
}

/** Per-100 ms sum over channels of mean square (K-weighted unless bypassed). */
function subBlockPower(left: Float32Array, right: Float32Array, opts: AnalyzeOptions): Float64Array {
    const chans = opts.bypassKWeighting
        ? [Float64Array.from(left), Float64Array.from(right)]
        : [kWeight(left), kWeight(right)];
    const n = Math.floor(left.length / SUB_BLOCK);
    const out = new Float64Array(n);
    for (const z of chans) {
        for (let k = 0; k < n; k++) {
            let s = 0;
            const base = k * SUB_BLOCK;
            for (let i = 0; i < SUB_BLOCK; i++) s += z[base + i] * z[base + i];
            out[k] += s / SUB_BLOCK;
        }
    }
    return out;
}

const powerToLufs = (p: number): number => -0.691 + 10 * Math.log10(p);

/** Mean power of windows of `len` sub-blocks, hopped one sub-block at a time. */
function windowPowers(sub: Float64Array, len: number): Float64Array {
    if (sub.length === 0) return new Float64Array(0);
    const L = Math.min(len, sub.length);
    const out = new Float64Array(sub.length - L + 1);
    let acc = 0;
    for (let i = 0; i < L; i++) acc += sub[i];
    out[0] = acc / L;
    for (let i = L; i < sub.length; i++) {
        acc += sub[i] - sub[i - L];
        out[i - L + 1] = acc / L;
    }
    return out;
}

function integratedLoudness(sub: Float64Array): number | null {
    // 400 ms blocks with 75% overlap = 4 sub-blocks hopped by 1.
    const blocks = windowPowers(sub, 4);
    const absGated: number[] = [];
    for (const p of blocks) if (p > 0 && powerToLufs(p) > -70) absGated.push(p);
    if (absGated.length === 0) return null;
    const relThresh = powerToLufs(absGated.reduce((a, b) => a + b, 0) / absGated.length) - 10;
    let s = 0, c = 0;
    for (const p of absGated) if (powerToLufs(p) > relThresh) { s += p; c++; }
    return c ? powerToLufs(s / c) : null;
}

// ---------------------------------------------------------------- true peak
// 4x polyphase interpolator: Hann-windowed sinc, 12 taps per phase (48 total,
// same length as the BS.1770-4 Annex 2 example filter).
const OS = 4;
const TAPS_PER_PHASE = 12;
const PHASES: Float64Array[] = (() => {
    const N = OS * TAPS_PER_PHASE;
    const centre = (N - 1) / 2;
    const h = new Float64Array(N);
    for (let n = 0; n < N; n++) {
        const t = (n - centre) / OS;
        const sinc = t === 0 ? 1 : Math.sin(Math.PI * t) / (Math.PI * t);
        const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * (n + 0.5)) / N);
        h[n] = sinc * w;
    }
    const phases: Float64Array[] = [];
    for (let p = 0; p < OS; p++) {
        const ph = new Float64Array(TAPS_PER_PHASE);
        let g = 0;
        for (let k = 0; k < TAPS_PER_PHASE; k++) { ph[k] = h[p + OS * k]; g += ph[k]; }
        for (let k = 0; k < TAPS_PER_PHASE; k++) ph[k] /= g; // unity DC gain per phase
        phases.push(ph);
    }
    return phases;
})();

function truePeakAbs(x: Float32Array): number {
    let peak = 0;
    for (let i = 0; i < x.length; i++) { const a = Math.abs(x[i]); if (a > peak) peak = a; }
    const n = x.length;
    for (let m = 0; m < n + TAPS_PER_PHASE; m++) {
        for (let p = 0; p < OS; p++) {
            const ph = PHASES[p];
            let acc = 0;
            for (let k = 0; k < TAPS_PER_PHASE; k++) {
                const idx = m - k;
                if (idx >= 0 && idx < n) acc += x[idx] * ph[k];
            }
            const a = acc < 0 ? -acc : acc;
            if (a > peak) peak = a;
        }
    }
    return peak;
}

// ---------------------------------------------------------------- FFT
function fftInPlace(re: Float64Array, im: Float64Array): void {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            let t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (let len = 2; len <= n; len <<= 1) {
        const ang = (-2 * Math.PI) / len;
        const wr = Math.cos(ang), wi = Math.sin(ang);
        const half = len >> 1;
        for (let i = 0; i < n; i += len) {
            let cr = 1, ci = 0;
            for (let k = 0; k < half; k++) {
                const a = i + k, b = a + half;
                const tr = re[b] * cr - im[b] * ci;
                const ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr; im[a] += ti;
                const ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr; cr = ncr;
            }
        }
    }
}

const HANN = (() => {
    const w = new Float64Array(FFT_SIZE);
    for (let i = 0; i < FFT_SIZE; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / FFT_SIZE);
    return w;
})();

/** Power spectrum (bins 0..N/2) of one Hann frame of `mono` starting at `start`. */
function framePower(mono: Float32Array, start: number): Float64Array {
    const re = new Float64Array(FFT_SIZE), im = new Float64Array(FFT_SIZE);
    for (let i = 0; i < FFT_SIZE; i++) {
        const idx = start + i;
        re[i] = idx < mono.length ? mono[idx] * HANN[i] : 0;
    }
    fftInPlace(re, im);
    const p = new Float64Array(FFT_SIZE / 2 + 1);
    for (let k = 0; k <= FFT_SIZE / 2; k++) p[k] = re[k] * re[k] + im[k] * im[k];
    return p;
}

function frameFlatness(p: Float64Array): number | null {
    let logSum = 0, sum = 0;
    const n = p.length - 1; // bins 1..N/2, DC excluded
    for (let k = 1; k < p.length; k++) { logSum += Math.log(p[k] + EPS); sum += p[k]; }
    if (sum <= 1e-12) return null;
    return Math.exp(logSum / n) / (sum / n);
}

/** Average power spectrum over frames starting in [from, to - FFT_SIZE]. */
function avgSpectrum(mono: Float32Array, from: number, to: number): { avg: Float64Array; flat: number | null } {
    const avg = new Float64Array(FFT_SIZE / 2 + 1);
    let frames = 0, flatSum = 0, flatN = 0;
    const last = Math.max(from, to - FFT_SIZE);
    for (let s = from; s <= last; s += FFT_HOP) {
        const p = framePower(mono, s);
        for (let k = 0; k < p.length; k++) avg[k] += p[k];
        const f = frameFlatness(p);
        if (f !== null) { flatSum += f; flatN++; }
        frames++;
    }
    if (frames) for (let k = 0; k < avg.length; k++) avg[k] /= frames;
    return { avg, flat: flatN ? flatSum / flatN : null };
}

function monoMix(left: Float32Array, right: Float32Array): Float32Array {
    const m = new Float32Array(left.length);
    for (let i = 0; i < m.length; i++) m[i] = 0.5 * (left[i] + right[i]);
    return m;
}

// ---------------------------------------------------------------- main
export function analyzeSamples(left: Float32Array, right: Float32Array, opts: AnalyzeOptions = {}): AnalysisResult {
    const n = left.length;
    const sub = subBlockPower(left, right, opts);
    // Files shorter than one full window report null instead of a partial-window value:
    // integrated needs >= 400 ms (one gating block), short-term max needs >= 3 s.
    const integratedLufs = n >= FS * 0.4 ? integratedLoudness(sub) : null;
    let stMax = 0;
    if (n >= FS * 3) for (const p of windowPowers(sub, 30)) if (p > stMax) stMax = p;
    const shortTermMaxLufs = stMax > 0 ? powerToLufs(stMax) : null;

    let peak = 0, sumSq = 0, sl = 0, sr = 0, sll = 0, srr = 0, slr = 0;
    for (let i = 0; i < n; i++) {
        const l = left[i], r = right[i];
        const al = Math.abs(l), ar = Math.abs(r);
        if (al > peak) peak = al;
        if (ar > peak) peak = ar;
        sl += l; sr += r; sll += l * l; srr += r * r; slr += l * r;
    }
    sumSq = sll + srr;
    const tp = opts.bypassOversampling ? peak : Math.max(truePeakAbs(left), truePeakAbs(right));
    const rms = n ? Math.sqrt(sumSq / (2 * n)) : 0;
    const samplePeakDbfs = toDb(peak);
    const rmsDbfs = toDb(rms);
    const crestDb = samplePeakDbfs !== null && rmsDbfs !== null ? samplePeakDbfs - rmsDbfs : null;

    let stereoCorr: number | null = null;
    if (n > 1) {
        const ml = sl / n, mr = sr / n;
        const cov = slr / n - ml * mr;
        const vl = sll / n - ml * ml, vr = srr / n - mr * mr;
        if (vl > 1e-20 && vr > 1e-20) stereoCorr = Math.max(-1, Math.min(1, cov / Math.sqrt(vl * vr)));
    }

    const mono = monoMix(left, right);
    const { avg, flat } = avgSpectrum(mono, 0, Math.max(n, FFT_SIZE));
    let num = 0, den = 0;
    for (let k = 1; k < avg.length; k++) { num += ((k * FS) / FFT_SIZE) * avg[k]; den += avg[k]; }
    const centroidHz = den > 1e-12 ? num / den : null;

    const min: number[] = new Array(PEAK_BUCKETS).fill(0);
    const max: number[] = new Array(PEAK_BUCKETS).fill(0);
    if (n > 0) {
        for (let b = 0; b < PEAK_BUCKETS; b++) {
            const s = Math.floor((b * n) / PEAK_BUCKETS);
            const e = Math.min(n, Math.max(s + 1, Math.floor(((b + 1) * n) / PEAK_BUCKETS)));
            let lo = Infinity, hi = -Infinity;
            for (let i = s; i < e; i++) { const v = mono[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
            if (lo !== Infinity) { min[b] = lo; max[b] = hi; }
        }
    }

    return {
        analyzerVersion: ANALYZER_VERSION,
        integratedLufs,
        shortTermMaxLufs,
        truePeakDbtp: toDb(tp),
        samplePeakDbfs,
        rmsDbfs,
        crestDb,
        centroidHz,
        flatness: flat,
        stereoCorr,
        durationSec: n / FS,
        peaks: { min, max },
    };
}

export async function analyze(filePath: string, opts: AnalyzeOptions = {}): Promise<AnalysisResult> {
    const { left, right } = await decodeFile(filePath);
    return analyzeSamples(left, right, opts);
}

/** Per-500 ms comparison of two signals (b relative to a).
 *  Assumes both takes start together at sample 0: no alignment or offset search is done,
 *  and the trace covers only the shorter of the two. */
export function diffTraceSamples(a: StereoSamples, b: StereoSamples): DiffTrace {
    const hop = FS / 2;
    const n = Math.min(a.left.length, b.left.length);
    const segs = Math.floor(n / hop);
    const subA = subBlockPower(a.left.subarray(0, n), a.right.subarray(0, n), {});
    const subB = subBlockPower(b.left.subarray(0, n), b.right.subarray(0, n), {});
    // Short-term loudness over the (up to) 3 s window ending at each segment end.
    const stLoud = (sub: Float64Array, endSub: number): number | null => {
        const start = Math.max(0, endSub - 30);
        let s = 0;
        for (let i = start; i < endSub; i++) s += sub[i];
        const p = endSub > start ? s / (endSub - start) : 0;
        return p > 0 ? powerToLufs(p) : null;
    };
    const monoA = monoMix(a.left, a.right), monoB = monoMix(b.left, b.right);
    const loudnessDeltaLu: (number | null)[] = [];
    const logSpectralDistanceDb: number[] = [];
    for (let t = 0; t < segs; t++) {
        const endSub = (t + 1) * 5;
        const la = stLoud(subA, endSub), lb = stLoud(subB, endSub);
        loudnessDeltaLu.push(la !== null && lb !== null ? lb - la : null);
        const from = t * hop, to = from + hop;
        const sa = avgSpectrum(monoA, from, to).avg, sb = avgSpectrum(monoB, from, to).avg;
        let acc = 0;
        for (let k = 1; k < sa.length; k++) {
            const d = 10 * Math.log10((sa[k] + 1e-12) / (sb[k] + 1e-12));
            acc += d * d;
        }
        logSpectralDistanceDb.push(Math.sqrt(acc / (sa.length - 1)));
    }
    return { hopSec: 0.5, loudnessDeltaLu, logSpectralDistanceDb };
}

export async function diffTrace(aPath: string, bPath: string): Promise<DiffTrace> {
    const [a, b] = await Promise.all([decodeFile(aPath), decodeFile(bPath)]);
    return diffTraceSamples(a, b);
}
