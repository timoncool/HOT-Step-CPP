// training/lyricTags.ts — one section-tag vocabulary for every lyrics source.
//
// Sources disagree. Genius writes [Verse 1: Name], [Chorus - Name], speaker tags
// ([Duet], a singer's name), "Chorus:" lines, (Chorus), *piano solo*, typos
// ([Chrous]) and bracketed spoken lines; Music Flamingo writes [Verse 1],
// [Pre-Chorus: names], [Spoken Intro], [Piano Solo]; the ACE-Step transcriber
// writes [Chorus - Layered Male Vocals] and [Synth arpeggio intro].
//
// normaliseLyricTags maps all of it onto SECTIONS, unnumbered, one tag per line
// with a blank line before it, and never touches a lyric line. Dry run on 160
// real tracks: 99 changed, every resulting tag in SECTIONS.
//
// transferTags salvages lyrics with NO tags (Genius often has none): it aligns a
// tagged transcript of the same song (Music Flamingo) word by word and copies
// the section starts onto the untagged text, keeping the better words.

export const SECTIONS = ['Intro', 'Verse', 'Pre-Chorus', 'Chorus', 'Post-Chorus', 'Bridge',
  'Breakdown', 'Interlude', 'Instrumental', 'Outro'] as const;
export type Section = typeof SECTIONS[number];
type Raw = Section | 'Hook';

// Longest phrases first: "post chorus" must win over "chorus". The FIRST keyword by
// position wins: "Outro Chorus" is an outro, "Spoken Intro" an intro.
const KEYWORDS: Array<[RegExp, Raw]> = [
  [/\bpre[\s-]?chorus\b|\bpre[\s-]?hook\b|\bbuild[\s-]?up\b/, 'Pre-Chorus'],
  [/\bpost[\s-]?chorus\b/, 'Post-Chorus'],
  [/\bintro\b|\bopening\b/, 'Intro'],
  [/\boutro\b|\bcoda\b|\bending\b|\bfade[\s-]?out\b/, 'Outro'],
  [/\bverse\b/, 'Verse'],
  [/\bhook\b/, 'Hook'],
  [/\bchorus\b|\brefrain\b/, 'Chorus'],
  [/\bbridge\b|\bmiddle[\s-]?8\b/, 'Bridge'],
  [/\bbreakdown\b|\bdrop\b/, 'Breakdown'],
  [/\binterlude\b|\bskit\b/, 'Interlude'],
  [/\binstrumental\b|\binst\b|\bsolo\b|\bbreak\b|\briff\b/, 'Instrumental'],
];
const FUZZY: Record<string, Section> = {
  verse: 'Verse', chorus: 'Chorus', bridge: 'Bridge', intro: 'Intro', outro: 'Outro', interlude: 'Interlude',
};

/** Edit distance <= 1, counting one adjacent swap as one edit ("chrous"). */
function nearlyEqual(a: string, b: string): boolean {
  if (Math.abs(a.length - b.length) > 1) return false;
  let i = 0;
  while (i < a.length && a[i] === b[i]) i++;
  if (i === a.length && i === b.length) return true;
  return a.slice(i + 1) === b.slice(i + 1) || a.slice(i) === b.slice(i + 1) || a.slice(i + 1) === b.slice(i)
    || (a[i] === b[i + 1] && a[i + 1] === b[i] && a.slice(i + 2) === b.slice(i + 2));
}

function classify(text: string): Raw | null {
  const t = text.toLowerCase().replace(/[‐‑–—]/g, '-').trim();
  let best: { at: number; section: Raw } | null = null;
  for (const [re, section] of KEYWORDS) {
    const m = re.exec(t);
    if (m && (!best || m.index < best.at)) best = { at: m.index, section };
  }
  if (best) return best.section;
  for (const w of t.split(/[^a-z]+/)) {
    if (w.length >= 5) for (const k of Object.keys(FUZZY)) if (nearlyEqual(w, k)) return FUZZY[k];
  }
  return null;
}

/** A header's text ("Verse 1: Name", "Outro Chorus", "Chrous") -> section, or null. */
export function classifyHeader(text: string): Section | null {
  const raw = classify(text);
  return raw === 'Hook' ? 'Chorus' : raw;
}

// A parenthesised line is a header only when it is NOTHING but a section name:
// "(Chorus)", "(Verse 2)", "(drum solo)" yes; "(It's the lost verse!)" is sung.
const ONLY_SECTION = /^(?:intro|outro|verse|pre[\s-]?chorus|post[\s-]?chorus|chorus|refrain|hook|bridge|breakdown|interlude|instrumental|(?:\w+\s)?solo)(?:\s*\d{1,2})?(?:\s*x\s*\d)?$/i;
const SECTION_WORD = /(?:intro|outro|verse|pre[\s-]?chorus|post[\s-]?chorus|chorus|refrain|hook|bridge|breakdown|interlude|instrumental|solo)/i;
const TAG_LINE = /^\[[^\]]+\]$/;

/** What a line is: a section header (raw section, or '' to drop), or lyric text. */
function readHeader(line: string): { header: string } | null {
  let m: RegExpExecArray | null;
  if ((m = /^\[([^\]]*)\]$/.exec(line))) return { header: m[1] };
  if ((m = /^\(([^)]*)\)$/.exec(line)) && ONLY_SECTION.test(m[1].trim())) return { header: m[1] };
  if ((m = /^\*([^*]+)\*$/.exec(line))) return { header: /solo|instrumental/i.test(m[1]) ? m[1] : '' };
  // "Chorus:" is a header; "Richard:" / "Spoken:" a speaker label (dropped); a
  // longer line ending in a colon is a lyric.
  if ((m = /^([A-Za-z][A-Za-z0-9 \-]{0,24}):$/.exec(line))
    && (SECTION_WORD.test(m[1]) || m[1].trim().split(/\s+/).length <= 2)) {
    return { header: SECTION_WORD.test(m[1]) ? m[1] : '' };
  }
  return null;
}

/**
 * Normalise every section tag to SECTIONS, unnumbered. [Hook] is a [Chorus],
 * except in a track that ALSO has a chorus, where it is the [Post-Chorus]
 * (0.4% of 5,334 dataset lyrics have both). Lyric lines are never rewritten
 * beyond dropping a leading "(Spoken:)" cue.
 */
export function normaliseLyricTags(lyrics: string): string {
  const lines = String(lyrics ?? '').split(/\r\n|\r|\n/);
  const hasChorus = lines.some(l => {
    const h = readHeader(l.trim());
    return !!h && classify(h.header) === 'Chorus';
  });
  const out: string[] = [];
  const trimBlanks = () => { while (out.length && out[out.length - 1] === '') out.pop(); };
  // An empty section is noise, except [Instrumental], which has no words by design.
  const dropEmptyTag = () => {
    trimBlanks();
    if (out.length && TAG_LINE.test(out[out.length - 1]) && out[out.length - 1] !== '[Instrumental]') {
      out.pop();
      trimBlanks();
    }
  };
  const pushTag = (section: Section) => {
    dropEmptyTag();
    if (out.length && out[out.length - 1] === `[${section}]`) return;
    if (out.length) out.push('');
    out.push(`[${section}]`);
  };
  for (const rawLine of lines) {
    const line = rawLine.trim();
    const h = readHeader(line);
    if (h) {
      // Genius brackets spoken lines too ("[And good night.]"): keep the words.
      if (line.startsWith('[') && /[.!?]$/.test(h.header.trim()) && !classify(h.header)) {
        out.push(h.header.trim());
        continue;
      }
      const raw = h.header ? classify(h.header) : null;
      if (raw) pushTag(raw === 'Hook' ? (hasChorus ? 'Post-Chorus' : 'Chorus') : raw);
      continue;
    }
    const text = rawLine.replace(/^\s*\((?:spoken|sung|whispered|shouted)\s*:?\)\s*/i, '').trimEnd();
    if (/^\(\s*(?:rpt|repeat|x\s*\d)[^)]*\)$/i.test(text.trim())) continue;
    if (!text.trim()) {
      if (out.length && out[out.length - 1] !== '') out.push('');
      continue;
    }
    out.push(collapseLoops(text));
  }
  dropEmptyTag();   // a trailing empty tag
  return out.join('\n');
}

/**
 * Cut a decode loop inside one line: a phrase of 3+ words repeated back to back
 * more than twice keeps two copies. Flamingo with no repetition penalty (the
 * app's setting, best median WER) loops on rare tracks: "and the sun and the
 * moon and the stars" x7 took one track from 0.2 to 1.36 WER. A chorus sung
 * twice on separate lines is untouched; this only looks within a line.
 */
export function collapseLoops(line: string): string {
  // P sep P (sep P)+  ->  P sep P, for the SHORTEST phrase P of 3 to 13 words,
  // repeated until nothing changes (a long loop can nest a shorter one).
  const re = /(?<![\w'])((?:[\w'’-]+[\s,]+){2,12}?[\w'’-]+)([\s,.;!?]+)\1(?:[\s,.;!?]+\1)+(?![\w'])/gi;
  let s = line;
  for (let guard = 0; guard < 8; guard++) {
    const next = s.replace(re, '$1$2$1');
    if (next === s) break;
    s = next;
  }
  return s;
}

/**
 * A transcript that fell into a decode loop: one line five or more times in a row,
 * a phrase looping within a line, or the same parenthetical tacked onto five or
 * more lines in a row (an invented backing vocal). The caller redoes the track
 * with a light repetition penalty, which fixes these at the cost of a few WER
 * points elsewhere — so it is a per-track fallback, not the default.
 */
export function looksLooped(lyrics: string): boolean {
  const lines = String(lyrics ?? '').split(/\r\n|\r|\n/).map(l => l.trim()).filter(Boolean);
  let run = 1, tailRun = 1;
  for (let i = 1; i < lines.length; i++) {
    run = lines[i].toLowerCase() === lines[i - 1].toLowerCase() ? run + 1 : 1;
    const tail = (l: string) => /\([^()]{6,}\)\s*$/.exec(l)?.[0].toLowerCase() ?? null;
    const t = tail(lines[i]);
    tailRun = t && t === tail(lines[i - 1]) ? tailRun + 1 : 1;
    if (run >= 5 || tailRun >= 5) return true;
  }
  return lines.some(l => collapseLoops(l) !== l);
}

/** True when the lyrics carry at least one recognisable section tag for SUNG
 *  material. Solos alone ([Instrumental], "*piano solo*") say nothing about where
 *  the verses and choruses are, so lyrics with only those count as untagged. */
export function hasSectionTags(lyrics: string): boolean {
  return String(lyrics ?? '').split(/\r\n|\r|\n/).some(l => {
    const h = readHeader(l.trim());
    const s = h && h.header ? classify(h.header) : null;
    return !!s && s !== 'Instrumental';
  });
}

function tokens(line: string): string[] {
  return line.toLowerCase().replace(/[’']/g, '').match(/[a-z0-9]+/g) ?? [];
}

/**
 * Copy section starts from `tagged` (a transcript of the same song with tags) onto
 * `untagged`, by word alignment (LCS). A tag lands on the target line holding the
 * first aligned word after it, moved up to its stanza's first line when it lands on
 * the second. [Instrumental] is not transferred: it marks audio with no words, and
 * on a lyric sheet that is the gap between stanzas, not a line. Returns `untagged`
 * unchanged when too little aligns to trust (under 40% of its words).
 */
export function transferTags(untagged: string, tagged: string): string {
  const target = String(untagged ?? '').split(/\r\n|\r|\n/);
  const tw: Array<{ w: string; line: number }> = [];
  target.forEach((l, i) => { for (const w of tokens(l)) tw.push({ w, line: i }); });

  const sw: string[] = [];
  const starts: Array<{ at: number; section: Section }> = [];
  for (const l of String(tagged ?? '').split(/\r\n|\r|\n/)) {
    const h = readHeader(l.trim());
    if (h) {
      const s = h.header ? classifyHeader(h.header) : null;
      if (s) starts.push({ at: sw.length, section: s });
      continue;
    }
    sw.push(...tokens(l));
  }
  if (!tw.length || !sw.length || !starts.length) return untagged;

  // LCS over words; lyrics are a few hundred words, so O(n*m) is fine.
  const n = sw.length, m = tw.length;
  const dp: Uint16Array[] = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = sw[i] === tw[j].w ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const s2t = new Map<number, number>();
  for (let i = 0, j = 0; i < n && j < m;) {
    if (sw[i] === tw[j].w) { s2t.set(i, j); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) i++;
    else j++;
  }
  if (dp[0][0] < 0.4 * m) return untagged;

  const insert = new Map<number, Section>();
  for (const { at, section } of starts) {
    if (section === 'Instrumental') continue;
    let j: number | undefined;
    for (let k = at; k < Math.min(at + 12, n) && j === undefined; k++) j = s2t.get(k);
    if (j === undefined) continue;
    let line = tw[j].line;
    // Landed on a stanza's second line: the section starts with the stanza.
    if (line > 0 && target[line - 1].trim() && (line < 2 || !target[line - 2].trim())) line--;
    if (!insert.has(line)) insert.set(line, section);
  }
  const out: string[] = [];
  target.forEach((l, i) => {
    const s = insert.get(i);
    if (s) {
      if (out.length && out[out.length - 1].trim()) out.push('');
      out.push(`[${s}]`);
    }
    out.push(l);
  });
  return out.join('\n');
}
