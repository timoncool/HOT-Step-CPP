// contracts/yue2Prepare.ts — the YuE2 Cover Studio Prepare document
// (version 2): a confirmed score structure, then lyric blocks assigned to its
// sections. Shared by the server and the UI, so it imports nothing and runs in
// a browser as it runs in Node.
//
// The ABC stays the musical authority. `sectionIds[i]` names the i-th section
// marker of `abc` as scoreSections.ts counts them; ids survive edits that keep
// a section, so lyric assignments can follow it. Words live in `blocks`;
// `assignments` maps each section id to the blocks sung there, in order, or
// marks it instrumental. Every pasted block is either assigned or excluded.
//
// Two confirmations, both plain values: `structure` holds the hash of the ABC
// the user confirmed, `lyrics` that hash plus the lyric revision confirmed.
// Every change to words, assignments, instrumental or excluded status bumps
// `lyricRevision`, so an old lyric confirmation stops matching. Generate and
// Build call preparedPair(), the one submission check: both confirmations
// current, every section and block accounted for, then the tagged lyrics in
// score order.

export const PREPARE_V2_VERSION = 2;

export interface PrepareLyricBlock {
  id: string;
  /** The words, tag line excluded. Lines are LF-separated. */
  text: string;
  /** The tag it was pasted under, a hint only: it never names a section. */
  tag: string | null;
  /** Set on an accepted repeat ("reuse for Chorus 2"): the block it copies. */
  copyOf?: string;
}

export type PrepareAssignment = { kind: 'words'; blockIds: string[] } | { kind: 'instrumental' };

export interface PrepareV2 {
  version: typeof PREPARE_V2_VERSION;
  /** Which source this preparation belongs to; a different source starts a new one. */
  sourceKey: string;
  abc: string;
  /** One id per section marker of `abc`, in score order. */
  sectionIds: string[];
  /** The last pasted lyrics, as pasted. */
  paste: string;
  blocks: PrepareLyricBlock[];
  /** By section id. A missing entry is a section with no words yet. */
  assignments: Record<string, PrepareAssignment>;
  /** Block ids kept in the draft but not sung. */
  excluded: string[];
  lyricRevision: number;
  /** Sections whose assignment an edit cleared, for review. */
  flagged: string[];
  /** Counter behind new section (s<n>) and block (b<n>) ids. */
  nextId: number;
  confirmed: {
    structure: { structureHash: string } | null;
    lyrics: { structureHash: string; lyricRevision: number } | null;
  };
}

/** The ABC exactly as a render sends it: LF line ends, trimmed (generate.ts
 *  trims yue2Abc). Hashing and export both use this form. */
export const normalizeAbc = (abc: string): string => abc.replace(/\r\n?/g, '\n').trim();

/** A fixed-width hash of the normalized ABC, for change detection (not
 *  security): two 32-bit FNV-1a lanes over UTF-16 code units. */
export function structureHash(abc: string): string {
  const text = normalizeAbc(abc);
  let a = 0x811c9dc5, b = 0x01000193 ^ text.length;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    a = Math.imul(a ^ c, 0x01000193);
    b = Math.imul(b ^ c, 0x5bd1e995) ^ (b >>> 15);
  }
  const hex = (n: number) => (n >>> 0).toString(16).padStart(8, '0');
  return `v1:${hex(a)}${hex(b)}`;
}

/** Section labels in score order, counted as the server's scoreSections.ts
 *  counts them: every whole-line `%` comment with text after the `%`. */
export function prepareSectionLabels(abc: string): string[] {
  return normalizeAbc(abc).split('\n').map(line => line.trim())
    .filter(line => line.startsWith('%') && line.slice(1).trim()).map(line => line.slice(1).trim());
}

const TAG_LINE = /^\s*\[[^\]\n]*\]/;
const isObject = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const isCount = (v: unknown) => Number.isInteger(v) && (v as number) >= 0;
const strings = (v: unknown): v is string[] => Array.isArray(v) && v.every(x => typeof x === 'string');

/** Why `value` is not a well-formed Prepare document, or null. Checks shape
 *  and internal references only; completeness is preparedPair()'s job. */
export function prepareV2Problem(value: unknown): string | null {
  if (!isObject(value)) return 'Prepare data must be an object.';
  const p = value;
  if (p.version !== PREPARE_V2_VERSION) return 'Unsupported Prepare data version.';
  if (typeof p.sourceKey !== 'string' || typeof p.abc !== 'string' || typeof p.paste !== 'string') return 'Prepare data is missing its source, score or paste.';
  if (!strings(p.sectionIds) || !strings(p.excluded) || !strings(p.flagged)) return 'Prepare section, excluded and flagged lists must be text.';
  if (!isCount(p.lyricRevision) || !isCount(p.nextId)) return 'Prepare revisions must be whole numbers.';
  if (new Set(p.sectionIds).size !== p.sectionIds.length) return 'Two sections share an id.';
  if (p.sectionIds.length !== prepareSectionLabels(p.abc).length) return 'The section ids do not match the score\'s sections.';
  if (!Array.isArray(p.blocks)) return 'Prepare blocks must be a list.';
  const blockIds = new Set<string>();
  for (const block of p.blocks) {
    if (!isObject(block) || typeof block.id !== 'string' || typeof block.text !== 'string' ||
      !(block.tag === null || typeof block.tag === 'string') || !(block.copyOf === undefined || typeof block.copyOf === 'string'))
      return 'A lyric block is malformed.';
    if (blockIds.has(block.id)) return 'Two lyric blocks share an id.';
    blockIds.add(block.id);
  }
  if (!isObject(p.assignments)) return 'Prepare assignments must be an object.';
  const sections = new Set(p.sectionIds);
  for (const [section, a] of Object.entries(p.assignments)) {
    if (!sections.has(section)) return 'An assignment names a section the score does not have.';
    if (!isObject(a) || !(a.kind === 'instrumental' || (a.kind === 'words' && strings(a.blockIds) && a.blockIds.length)))
      return 'An assignment is malformed.';
    if (a.kind === 'words' && (a.blockIds as string[]).some(id => !blockIds.has(id))) return 'An assignment names a missing lyric block.';
  }
  if (p.excluded.some(id => !blockIds.has(id))) return 'An excluded block is missing.';
  const c = p.confirmed;
  if (!isObject(c)) return 'Prepare confirmations are malformed.';
  if (!(c.structure === null || (isObject(c.structure) && typeof c.structure.structureHash === 'string'))) return 'The structure confirmation is malformed.';
  if (!(c.lyrics === null || (isObject(c.lyrics) && typeof c.lyrics.structureHash === 'string' && isCount(c.lyrics.lyricRevision))))
    return 'The lyrics confirmation is malformed.';
  return null;
}

/** Why the structure cannot be confirmed as it stands, or null. */
export function structureProblem(p: Pick<PrepareV2, 'abc' | 'sectionIds'>): string | null {
  const labels = prepareSectionLabels(p.abc);
  if (!labels.length) return 'The score has no sections.';
  if (labels.some(label => label.startsWith('%'))) return 'The score has a %% directive line, which counts as a section; remove it first.';
  if (labels.some(label => /[[\]]/.test(label))) return 'A section name has a bracket, which would break its lyric tag.';
  if (labels.length !== p.sectionIds.length) return 'The section ids do not match the score\'s sections.';
  return null;
}

/** Why the words and assignments are not complete, or null: every section
 *  has words or is instrumental, every block is sung once or excluded. */
export function assignmentProblem(p: Pick<PrepareV2, 'abc' | 'sectionIds' | 'blocks' | 'assignments' | 'excluded'>): string | null {
  const labels = prepareSectionLabels(p.abc);
  const missing = p.sectionIds.findIndex(id => !p.assignments[id]);
  if (missing >= 0) return `${labels[missing] ?? 'A section'} (section ${missing + 1}) needs words or Instrumental.`;
  const used = new Map<string, number>();
  for (const id of p.sectionIds) {
    const a = p.assignments[id];
    if (a.kind === 'words') for (const block of a.blockIds) used.set(block, (used.get(block) ?? 0) + 1);
  }
  const excluded = new Set(p.excluded);
  for (const block of p.blocks) {
    const n = used.get(block.id) ?? 0;
    if (n > 1) return 'A lyric block is assigned twice; reuse it as a copy instead.';
    if (n && excluded.has(block.id)) return 'A lyric block is both assigned and excluded.';
    if (!n && !excluded.has(block.id)) return 'Assign or exclude every lyric block.';
    if (n && block.text.split('\n').some(line => TAG_LINE.test(line))) return 'A lyric block has a line that looks like a section tag.';
  }
  return null;
}

/** The tagged lyrics for the score as assigned: one `[label]` per section in
 *  score order, its blocks' words below, an empty tag for an instrumental
 *  section. Assumes assignmentProblem() passed. */
export function exportPreparedLyrics(p: Pick<PrepareV2, 'abc' | 'sectionIds' | 'blocks' | 'assignments'>): string {
  const labels = prepareSectionLabels(p.abc);
  const text = new Map(p.blocks.map(block => [block.id, block.text.replace(/\r\n?/g, '\n').replace(/^\n+|\s+$/g, '')]));
  return p.sectionIds.map((id, i) => {
    const a = p.assignments[id];
    const words = a?.kind === 'words' ? a.blockIds.map(block => text.get(block) ?? '').filter(Boolean).join('\n') : '';
    return words ? `[${labels[i]}]\n${words}` : `[${labels[i]}]`;
  }).join('\n\n');
}

/** `instrumental`: no section has words. A render's Instrumental mode must
 *  agree with it (instrumentalConflict). */
export interface PreparedPair { abc: string; lyrics: string; structureHash: string; lyricRevision: number; instrumental: boolean }

/** Why a render's Instrumental mode contradicts the confirmed lyrics, or null. */
export const instrumentalConflict = (pair: PreparedPair, instrumental: unknown): string | null =>
  (instrumental === true) === pair.instrumental ? null
    : pair.instrumental ? 'Every section is confirmed Instrumental; turn on Instrumental mode.'
      : 'The confirmed lyrics have words; turn off Instrumental mode or mark every section Instrumental.';

/** The one submission check Generate and Build share: the score and tagged
 *  lyrics to render, or why not. `scoreAbc`, when given, is the score the
 *  caller is about to render (the approved ABC); it must be the confirmed one. */
export function preparedPair(value: unknown, scoreAbc?: string): PreparedPair | { error: string } {
  const shape = prepareV2Problem(value);
  if (shape) return { error: shape };
  const p = value as PrepareV2;
  const hash = structureHash(p.abc);
  if (p.confirmed.structure?.structureHash !== hash) return { error: 'Confirm the score structure first.' };
  const structure = structureProblem(p);
  if (structure) return { error: structure };
  if (scoreAbc !== undefined && structureHash(scoreAbc) !== hash) return { error: 'The score changed since its structure was confirmed; confirm it again.' };
  const lyrics = p.confirmed.lyrics;
  if (lyrics?.structureHash !== hash || lyrics.lyricRevision !== p.lyricRevision) return { error: 'Confirm the lyrics first.' };
  const assigned = assignmentProblem(p);
  if (assigned) return { error: assigned };
  return { abc: normalizeAbc(p.abc), lyrics: exportPreparedLyrics(p), structureHash: hash, lyricRevision: p.lyricRevision,
    instrumental: p.sectionIds.every(id => p.assignments[id].kind === 'instrumental') };
}

/** preparedPair for /api/generate-style params (`yue2Prepare`, `yue2Abc`,
 *  `lyrics`, `instrumental`), as Build and the direct generate route send
 *  them: the document must be current, the score the confirmed one, the mode
 *  agree with it and, unless instrumental, the lyrics be the ones it exports. */
export function preparedParams(params: Record<string, unknown>): PreparedPair | { error: string } {
  const pair = preparedPair(params.yue2Prepare, typeof params.yue2Abc === 'string' ? params.yue2Abc : '');
  if ('error' in pair) return pair;
  const conflict = instrumentalConflict(pair, params.instrumental);
  if (conflict) return { error: conflict };
  const text = typeof params.lyrics === 'string' ? params.lyrics.replace(/\r\n?/g, '\n').trim() : '';
  if (!pair.instrumental && text !== pair.lyrics) return { error: 'These lyrics are not the confirmed Prepare lyrics; confirm them on the Prepare page.' };
  return pair;
}
