/**
 * Adapter directory layout — one hierarchy per training base.
 *
 * Adapters trained on different architectures are not interchangeable, so they
 * live in per-base folders (2026-07-28, Rob's call):
 *
 *   <adapters>/
 *     lm-06b/<artist>/<run>/    planner-LM LoRAs per base size, one dir per
 *     lm-17b/<artist>/<run>/    training run (run = YYYY-MM-DD_HH-MM-SS, the
 *     lm-4b/<artist>/<run>/     logs/ convention: name-sorted = time-sorted)
 *     dit-xl-thirds/<artist>/<run>/   DiT adapters per base, dit-<shorthand>
 *     dit-xl-base-turbo/<artist>/<run>/
 *     ...
 *
 * The artist folder no longer carries a `-<size>` suffix — the parent folder
 * says what the adapter runs on — and retraining an artist never overwrites an
 * earlier adapter: each run gets its own stamped subfolder. Two legacy forms
 * are still READ everywhere so pre-migration installs keep working: an
 * unversioned adapter directly in the artist folder, and the original flat
 * layout (`lm/<artist>-4B`, DiT dirs at the root). Nothing new is ever written
 * to either.
 *
 * `server/scripts/migrate-adapter-layout.mjs` moves an existing corpus over.
 */
import fs from 'fs';
import path from 'path';
import { config } from '../../config.js';
import type { LmSize } from './types.js';
import { randomUUID } from 'node:crypto';

export function sa3RunsRoot(datasetSlug: string): string {
  if (!/^[a-z0-9][a-z0-9_-]*$/i.test(datasetSlug)) throw new Error('Invalid SA3 dataset slug');
  return path.join(config.aceServer.adapters, 'sa3-medium', datasetSlug);
}

/** Validate our F32 dora-rows export, including every tensor's bounds and
 * finite values. Raw resume adapters never enter the inference catalogue. */
export function validateSa3Export(file: string, expectedBase?: string): string {
  const fd = fs.openSync(file, 'r');
  try {
    const size = fs.fstatSync(fd).size;
    let pos = 0;
    const read = (n: number): Buffer => {
      if (!Number.isSafeInteger(n) || n < 0 || n > 1_048_576 || pos + n > size) throw new Error('Truncated SA3 GGUF');
      const b = Buffer.alloc(n);
      if (fs.readSync(fd, b, 0, n, pos) !== n) throw new Error('Truncated SA3 GGUF');
      pos += n;
      return b;
    };
    const u32 = () => read(4).readUInt32LE();
    const u64 = (): number => {
      const n = Number(read(8).readBigUInt64LE());
      if (!Number.isSafeInteger(n)) throw new Error('Oversized SA3 GGUF');
      return n;
    };
    const str = () => read(u64()).toString('utf8');
    if (read(4).toString() !== 'GGUF' || u32() !== 3) throw new Error('Invalid SA3 GGUF header');
    const count = u64(), keys = u64();
    if (count < 3 || count > 10000 || count % 3 || keys > 1000) throw new Error('Invalid SA3 tensor count');
    const meta = new Map<string, string | number>();
    for (let i = 0; i < keys; i++) {
      const key = str(), type = u32();
      if (meta.has(key)) throw new Error('Duplicate SA3 metadata');
      if (type === 8) meta.set(key, str());
      else if (type === 4) meta.set(key, u32());
      else throw new Error('Unsupported SA3 export metadata');
    }
    const base = meta.get('sa3.inference_base_id');
    if (meta.get('general.architecture') !== 'sa3-adapter' || meta.get('sa3.adapter_format') !== 'dora-rows-v1'
      || typeof base !== 'string' || !/^stabilityai\/stable-audio-3-medium sha256:[a-f0-9]{64}$/.test(base)
      || (expectedBase && base !== expectedBase)) throw new Error('SA3 export format or inference provenance mismatch');
    const tensors = new Map<string, { dims: number[]; offset: number; bytes: number }>();
    for (let i = 0; i < count; i++) {
      const name = str(), nd = u32();
      if (nd < 1 || nd > 2 || !/^model\..*\.weight\.(lora_a|lora_b|row_scale)$/.test(name)
        || tensors.has(name)) throw new Error('Invalid SA3 tensor name or dimensions');
      const dims = Array.from({ length: nd }, u64);
      if (dims.some(n => n < 1 || n > 1_000_000) || u32() !== 0) throw new Error('SA3 export must contain F32 tensors');
      const offset = u64(), bytes = dims.reduce((a, b) => a * b, 4);
      if (!Number.isSafeInteger(bytes)) throw new Error('Oversized SA3 tensor');
      tensors.set(name, { dims, offset, bytes });
    }
    const alignment = meta.get('general.alignment') ?? 32;
    if (typeof alignment !== 'number' || alignment < 1 || alignment > 4096 || (alignment & (alignment - 1))) throw new Error('Invalid SA3 alignment');
    const start = Math.ceil(pos / alignment) * alignment;
    let end = start;
    const chunk = Buffer.alloc(65536);
    for (const [name, tensor] of [...tensors].sort((a, b) => a[1].offset - b[1].offset)) {
      if (tensor.offset % alignment || start + tensor.offset < end || start + tensor.offset + tensor.bytes > size) throw new Error('Truncated or overlapping SA3 tensors');
      end = start + tensor.offset + tensor.bytes;
      if (name.endsWith('.lora_a')) {
        const stem = name.slice(0, -7), b = tensors.get(`${stem}.lora_b`), s = tensors.get(`${stem}.row_scale`);
        if (tensor.dims.length !== 2 || !b || b.dims.length !== 2 || !s || s.dims.length !== 1
          || tensor.dims[1] !== b.dims[0] || b.dims[1] !== s.dims[0]) throw new Error('Incomplete SA3 DoRA target');
      } else {
        const stem = name.replace(/\.(lora_b|row_scale)$/, '');
        if (!tensors.has(`${stem}.lora_a`)) throw new Error('Unpaired SA3 target');
      }
      for (let off = 0; off < tensor.bytes; off += chunk.length) {
        const n = Math.min(chunk.length, tensor.bytes - off);
        if (fs.readSync(fd, chunk, 0, n, start + tensor.offset + off) !== n) throw new Error('Truncated SA3 data');
        for (let j = 0; j < n; j += 4) if (!Number.isFinite(chunk.readFloatLE(j))) throw new Error('Non-finite SA3 tensor');
      }
    }
    return base;
  } finally { fs.closeSync(fd); }
}

export function listSa3Adapters(modelsDir = config.aceServer.models): Array<{ name: string; sizeMb: number }> {
  const dir = path.join(modelsDir, 'sa3-adapters');
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).filter(f => f.endsWith('.gguf') && fs.statSync(path.join(dir, f)).isFile())
    .map(f => ({ name: f.slice(0, -5), sizeMb: Math.round(fs.statSync(path.join(dir, f)).size / 1e6) }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

/** Copy and validate privately, then link atomically with no overwrite. The
 * temporary file lives on the catalogue volume, so cross-drive runs work. */
export function publishSa3Export(file: string, name: string, expectedBase: string, modelsDir = config.aceServer.models): string {
  if (!/^[a-z0-9][a-z0-9_-]{0,180}$/i.test(name)) throw new Error('Invalid SA3 publication name');
  const dir = path.join(modelsDir, 'sa3-adapters');
  fs.mkdirSync(dir, { recursive: true });
  const temp = path.join(dir, `.${randomUUID()}.part`);
  try {
    fs.copyFileSync(file, temp, fs.constants.COPYFILE_EXCL);
    validateSa3Export(temp, expectedBase);
    for (let i = 1; i < 10000; i++) {
      const dest = path.join(dir, `${name}${i === 1 ? '' : `-${i}`}.gguf`);
      try { fs.linkSync(temp, dest); return path.basename(dest, '.gguf'); }
      catch (err: any) { if (err.code !== 'EEXIST') throw err; }
    }
    throw new Error('SA3 catalogue name exhausted');
  } finally { fs.rmSync(temp, { force: true }); }
}

// ── planner-LM sizes ────────────────────────────────────────────────────────

const LM_SIZE_SLUGS: Record<LmSize, string> = {
  '0.6B': 'lm-06b',
  '1.7B': 'lm-17b',
  '4B': 'lm-4b',
};

/** 'lm-06b' | 'lm-17b' | 'lm-4b' — the per-size folder under <adapters>. */
export function lmSizeSlug(size: LmSize): string {
  return LM_SIZE_SLUGS[size] ?? `lm-${String(size).toLowerCase().replace(/\./g, '')}`;
}

/** Inverse of lmSizeSlug. '' when the folder name is not a size slug. */
export function lmSizeFromSlug(dirName: string): LmSize | '' {
  const hit = (Object.entries(LM_SIZE_SLUGS) as Array<[LmSize, string]>)
    .find(([, slug]) => slug === dirName.toLowerCase());
  return hit ? hit[0] : '';
}

/** Every folder GET /api/adapters/lm scans: the three size roots plus the
 *  legacy flat `lm/` so a pre-migration install keeps working. */
export function lmAdapterRoots(): Array<{ dir: string; size: LmSize | '' }> {
  const root = config.aceServer.adapters;
  return [
    { dir: path.join(root, 'lm-06b'), size: '0.6B' },
    { dir: path.join(root, 'lm-17b'), size: '1.7B' },
    { dir: path.join(root, 'lm-4b'), size: '4B' },
    { dir: path.join(root, 'lm'), size: '' },   // legacy — size read off the -<size> suffix
  ];
}

/**
 * The base size an LM adapter path implies: the parent folder's slug first
 * (new layout), then the legacy `-<size>` name suffix. '' when neither says.
 *
 * This is what pins the audition's base LM — before the folder layout the
 * suffix was the only signal, and losing it silently put a 4B adapter on the
 * sticky 0.6B base ("36 layers but model has 28").
 */
export function lmSizeOfAdapterPath(adapterPath: string): LmSize | '' {
  const parent = path.basename(path.dirname(path.resolve(adapterPath)));
  const bySlug = lmSizeFromSlug(parent);
  if (bySlug) return bySlug;
  const m = /-(0\.6B|1\.7B|4B)$/i.exec(path.basename(adapterPath));
  if (!m) return '';
  const norm = m[1].toUpperCase().replace('B', 'B');
  return (['0.6B', '1.7B', '4B'] as LmSize[]).find(s => s.toUpperCase() === norm) ?? '';
}

// ── DiT base shorthands ─────────────────────────────────────────────────────

/**
 * Confirmed with Rob 2026-07-28 — the folder a DiT adapter trained on each
 * base lands in. Keys are the model-name stem (no quant token, no .gguf).
 */
const DIT_SHORTHANDS: Record<string, string> = {
  'acestep-v15-merge-base-sft-turbo-xl-thirds': 'dit-xl-thirds',
  'acestep-v15-merge-base-turbo-xl-ta-0.5': 'dit-xl-base-turbo',
  'acestep-v15-xl-sftturbo50': 'dit-xl-sft-turbo',
  'acestep-v15-merge-base-sft-xl-ta-0.5': 'dit-xl-base-sft',
  'acestep-v15-merge-sft-turbo-xl-ta-0.3': 'dit-xl-sft-turbo-ta03',
  'acestep-v15-merge-sft-turbo-xl-ta-0.7': 'dit-xl-sft-turbo-ta07',
  'acestep-v15-xl-base': 'dit-xl-base',
  'acestep-v15-xl-sft': 'dit-xl-sft',
  'acestep-v15-xl-turbo': 'dit-xl-turbo',
  'acestep-v15-base': 'dit-base',
  'acestep-v15-sft': 'dit-sft',
  'acestep-v15-turbo': 'dit-turbo',
  'acestep-v15-sftturbo50': 'dit-sft-turbo',
  'acestep-v15-turbo-continuous': 'dit-turbo-continuous',
  'acestep-v15-turbo-shift1': 'dit-turbo-shift1',
  'acestep-v15-turbo-shift3': 'dit-turbo-shift3',
  'acestep-v15-merge-base-sft-turbo-xl-thirds-convrot-ref': 'dit-xl-thirds-convrot',
  'sa3-dit': 'dit-sa3',
};

/** Quant/precision tokens that end a model filename. */
const QUANT_RE = /-(BF16|F16|F32|MXFP4|NVFP4|IQ\d[A-Z_]*|Q\d[\w]*)$/i;

/** Model name/filename/path → its stem: no dir, no .gguf, no quant token. */
export function ditModelStem(model: string): string {
  let stem = path.basename(String(model ?? '')).replace(/\.gguf$/i, '');
  stem = stem.replace(QUANT_RE, '');
  return stem;
}

/**
 * Shorthand folder for a DiT base. Unknown bases degrade to a recognisable
 * slug (name minus the family prefix and quant token) rather than failing —
 * a future model should never block training on a folder name.
 */
export function ditShorthand(model: string): string {
  const stem = ditModelStem(model);
  if (!stem) return 'dit-unknown-base';
  const mapped = DIT_SHORTHANDS[stem];
  if (mapped) return mapped;
  const fallback = stem.replace(/^acestep-v15-/i, '').replace(/\./g, '');
  return fallback ? `dit-${fallback}` : 'dit-unknown-base';
}

// ── training-run versioning ─────────────────────────────────────────────────

/** logs/-convention timestamp: YYYY-MM-DD_HH-MM-SS, local time. */
export function runStamp(when = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0');
  return `${when.getFullYear()}-${p(when.getMonth() + 1)}-${p(when.getDate())}` +
    `_${p(when.getHours())}-${p(when.getMinutes())}-${p(when.getSeconds())}`;
}

/** True when a directory holds adapter weights directly. Both trained layouts
 *  count: PEFT (`adapter_model.safetensors`) and the LyCORIS LoKR export
 *  ace-train `--adapter-type lokr` writes (`lokr_weights.safetensors`, no
 *  adapter_config.json — lokr-dit-training plan §2.4). Checking only the PEFT
 *  leaf made every LoKR run dir invisible to latestRunDir(). */
export function hasWeights(dir: string): boolean {
  return fs.existsSync(path.join(dir, 'adapter_model.safetensors'))
    || fs.existsSync(path.join(dir, 'lokr_weights.safetensors'));
}

/**
 * A FRESH run dir under `artistDir` for a training run to write into —
 * `<artistDir>/<stamp>`, bumped with `-2`, `-3`… in the (retry-within-a-second)
 * case where the stamp already exists. Retraining an artist therefore never
 * overwrites an earlier adapter. Callers pass an already-sanitised artist dir.
 */
export function newRunDir(artistDir: string): string {
  const base = path.join(artistDir, runStamp());
  if (!fs.existsSync(base)) return base;
  for (let i = 2; i < 100; i++) {
    const c = `${base}-${i}`;
    if (!fs.existsSync(c)) return c;
  }
  return `${base}-${Date.now().toString(36)}`;
}

/**
 * The newest trained adapter under `artistDir`: the lexicographically-last run
 * subfolder holding weights (stamps sort chronologically), else the artist dir
 * itself when it holds weights directly (unversioned legacy), else ''.
 */
export function latestRunDir(artistDir: string): string {
  try {
    const runs = fs.readdirSync(artistDir, { withFileTypes: true })
      .filter(e => e.isDirectory() && !e.name.startsWith('.') && hasWeights(path.join(artistDir, e.name)))
      .map(e => e.name)
      .sort();
    if (runs.length) return path.join(artistDir, runs[runs.length - 1]);
  } catch { /* no artist dir yet */ }
  return hasWeights(artistDir) ? artistDir : '';
}
