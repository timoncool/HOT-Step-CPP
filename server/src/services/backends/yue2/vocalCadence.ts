import type { FloatWav } from '../../generation/audioLevel.js';
import { scoreBarClock } from './scoreClock.js';
import { barsIn, isScoreVocalVoice, scoreBarSegments, vocalVoiceIds } from './scoreHealth.js';

const HOP_SECONDS = 0.01;
/** A frame counts as voiced within this many dB of the stem's loud frames. */
const VOICED_DB = 24;
/** Never voiced below this, however quiet the whole stem is (separation residue). */
const SILENCE_DB = -50;
/** A rise this steep inside a voiced run starts a new syllable. */
const ONSET_RISE_DB = 6;

/** Per-hop loudness of the stem in dB, channels mixed down. */
function frameDb(wav: FloatWav): Float32Array {
  const hop = Math.max(1, Math.round(wav.sampleRate * HOP_SECONDS));
  const frames = Math.floor((wav.channels[0]?.length ?? 0) / hop);
  const db = new Float32Array(frames);
  for (let f = 0; f < frames; f++) {
    let sum = 0;
    for (const ch of wav.channels) for (let i = f * hop; i < (f + 1) * hop; i++) sum += ch[i] * ch[i];
    db[f] = 10 * Math.log10(sum / (hop * wav.channels.length) + 1e-12);
  }
  return db;
}

/** Syllable onset times and a voiced test, from the vocal stem's loudness. */
function vocalActivity(wav: FloatWav) {
  const db = frameDb(wav);
  const sorted = Array.from(db).sort((a, b) => a - b);
  const floor = Math.max((sorted[Math.floor(sorted.length * 0.95)] ?? 0) - VOICED_DB, SILENCE_DB);
  const voiced = (f: number) => f >= 0 && f < db.length && db[f] > floor;
  const onsets: number[] = [];
  for (let f = 0; f < db.length; f++) {
    if (!voiced(f)) continue;
    const before = Math.min(db[f - 1] ?? -Infinity, db[f - 2] ?? -Infinity, db[f - 3] ?? -Infinity);
    if (!voiced(f - 1) || db[f] - before >= ONSET_RISE_DB) onsets.push(f * HOP_SECONDS);
    // One onset per rise: skip on while still climbing.
    while (f + 1 < db.length && voiced(f + 1) && db[f + 1] >= db[f]) f++;
  }
  const voicedShare = (from: number, to: number) => {
    let n = 0, on = 0;
    for (let f = Math.floor(from / HOP_SECONDS); f < Math.ceil(to / HOP_SECONDS); f++, n++) if (voiced(f)) on++;
    return n ? on / n : 0;
  };
  return { onsets, voicedShare };
}

function unitWholeNotes(abc: string): number {
  const m = /^L:\s*(\d+)\s*\/\s*(\d+)\s*$/m.exec(abc);
  return m && Number(m[1]) > 0 && Number(m[2]) > 0 ? Number(m[1]) / Number(m[2]) : 1 / 8;
}

/** The key's tonic as a bare letter, which the key signature then sharpens or flattens. */
function tonicLetter(abc: string): string {
  return /^K:\s*([A-G])/m.exec(abc)?.[1] ?? 'C';
}

/** n L units as plain, dotted or whole lengths ABC can write, tied for a note. */
function spell(n: number, symbol: string, tie: string): string {
  const parts: string[] = [];
  for (const len of [16, 12, 8, 6, 4, 3, 2, 1]) {
    while (n >= len) { parts.push(`${symbol}${len === 1 ? '' : len}`); n -= len; }
  }
  return parts.join(tie);
}

/** Chord symbols of one silent bar, keyed by their offset in L units. */
function barChords(segment: string): Map<number, string> {
  const chords = new Map<number, string>();
  let at = 0;
  for (const m of segment.matchAll(/"([^"]*)"|z(\d*)(?:\/(\d*))?/g)) {
    if (m[1] !== undefined) chords.set(at, m[1]);
    else at += (m[2] ? Number(m[2]) : 1) / (m[3] !== undefined ? Number(m[3] || 2) : 1);
  }
  return chords;
}

/**
 * Write a silent Vocal line as the source vocal's rhythm on one note: each
 * syllable onset in the stem starts a note on the score's L grid, held while
 * the stem stays voiced. Chords, the Ins line and the bar grid are kept.
 * The score clock starts at the audio's first sample, as scoreClock assumes.
 * ponytail: the score's whole-number Q: drifts from the recording's real
 * beats, so notes follow the source's density and sections, not its exact
 * syllables; exporting SheetSage2's beat times would pin them. One note (the
 * tonic) from a loudness onset detector; add a pitch track if a missed sung
 * melody needs its contour too.
 */
export function vocalCadenceAbc(abc: string, wav: FloatWav): string {
  const bars = scoreBarClock(abc);
  const unit = unitWholeNotes(abc);
  const note = tonicLetter(abc);
  const { onsets, voicedShare } = vocalActivity(wav);
  const slotsIn = (b: number) => Math.round((bars[b].wholeEnd - bars[b].wholeStart) / unit);
  const slotSeconds = (b: number) => (bars[b].end - bars[b].start) / slotsIn(b);

  // Flat slot grid over every bar: start time, and whether a syllable lands on it.
  const slots: { bar: number; start: number; onset: boolean; voiced: boolean }[] = [];
  bars.forEach((bar, b) => {
    for (let s = 0; s < slotsIn(b); s++) {
      const start = bar.start + s * slotSeconds(b);
      slots.push({ bar: b, start, onset: false, voiced: voicedShare(start, start + slotSeconds(b)) >= 0.5 });
    }
  });
  let k = 0;
  for (const t of onsets) {
    while (k + 1 < slots.length && Math.abs(slots[k + 1].start - t) <= Math.abs(slots[k].start - t)) k++;
    if (slots.length) slots[k].onset = true;
  }

  const barText = (b: number, chords: Map<number, string>) => {
    const own = slots.filter(slot => slot.bar === b);
    let out = '';
    let i = 0;
    let sounding = false;
    while (i < own.length) {
      const chord = chords.get(i);
      if (chord !== undefined) out += `"${chord}"`;
      const starts = own[i].onset;
      const holds: boolean = !starts && sounding && own[i].voiced;
      let n = 1;
      const same = (j: number) => !chords.has(j) && !own[j].onset && (starts || holds ? own[j].voiced : true);
      while (i + n < own.length && same(i + n)) n++;
      if (starts) out += spell(n, note, '-');
      else if (holds) out = out.replace(/("[^"]*")?$/, '-$1') + spell(n, note, '-');
      else out += spell(n, 'z', '');
      sounding = starts || holds;
      i += n;
    }
    return out;
  };

  const vocalIds = vocalVoiceIds(abc);
  let inVocal = false;
  let bar = 0;
  return abc.split('\n').map(raw => {
    const line = raw.trim();
    if (!line || line.startsWith('%')) return raw;
    if (/^V:/.test(line)) { inVocal = isScoreVocalVoice(line, vocalIds); return raw; }
    if (/^[A-Za-z]:/.test(line) || !inVocal) return raw;
    const out = scoreBarSegments(line).flatMap(segment => {
      const n = barsIn(segment);
      const chords = n === 1 ? barChords(segment) : new Map<number, string>();
      return Array.from({ length: n }, () => barText(bar++, chords));
    });
    return `${out.join('|')}|`;
  }).join('\n');
}
