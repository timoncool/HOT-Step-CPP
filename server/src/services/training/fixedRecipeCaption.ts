// Lyric cue cleanup and caption parsing for the fixed-recipe method.
// Applied only by the fixed-recipe preparation mode.
const KNOWN: Record<string, string> = {
  intro: 'Intro', verse: 'Verse', 'pre-chorus': 'Pre-Chorus', prechorus: 'Pre-Chorus',
  chorus: 'Chorus', bridge: 'Bridge', interlude: 'Interlude', inst: 'Inst',
  instrumental: 'Inst', outro: 'Outro',
};
const SONG_SECTION = /^\s*\[[^\]]+\]\s*$/;
const CUE = /^\s*\([^()]*\)\s*$/;
const HEADER = /^\s*\[(Tags|Lyrics|Duration)\]\s*$/i;

export const FIXED_RECIPE_CAPTION_GRAMMAR = 'fixed-recipe-v3';

export function fixedRecipeTrigger(trigger: string | undefined, slug: string): string {
  return trigger?.trim() || slug;
}

// Unknown tags (e.g. [Breakdown]) are kept in Ruach's Title case, not dropped.
const titleCase = (key: string) => key.replace(/(^|[\s-])([a-z])/g, (_, lead: string, letter: string) => lead + letter.toUpperCase());

function splitLines(text: string): string[] {
  return text.split(/\r\n|\r|\n/).filter((_, i, lines) => i < lines.length - 1 || lines[i] !== '');
}

export function cleanFixedRecipeLyrics(text: string): { lyrics: string; hints: string[] } {
  const lines: string[] = [], hints: string[] = [];
  let started = false;
  for (const line of splitLines(text || '')) {
    if (SONG_SECTION.test(line) || line.trimStart().startsWith('[')) {
      started = true;
      for (const match of line.matchAll(/\[([^\]]+)\]/g)) {
        let key = match[1].trim().toLowerCase().split(/\s+-\s+|:/, 1)[0].trim().replace(/\s*\d+$/, '');
        if (key.includes('instrumental')) key = 'inst';
        else if (/(intro|outro|verse|chorus|bridge)$/.test(key) && KNOWN[key.split(' ').at(-1)!]) key = key.split(' ').at(-1)!;
        const name = KNOWN[key] ?? titleCase(key);
        const section = name && `[${name}]`;
        if (section && lines.at(-1) !== section) lines.push(section);
      }
      const rest = line.replace(/\[[^\]]*\]/g, '').trim();
      if (rest && !CUE.test(rest)) lines.push(rest.replace(/\s*\([^()]*\)/g, '').trim());
    } else if (!started) {
      if (line.trim()) hints.push(line.trim().replace(/^\(+|\)+$/g, ''));
    } else if (!CUE.test(line)) {
      const clean = line.replace(/\s*\([^()]*\)/g, '').trimEnd();
      if (clean.trim() || !lines.length || lines.at(-1)) lines.push(clean);
    }
  }
  const lyrics = lines.join('\n').replace(/\n{3,}/g, '\n\n').trim()
    .replace(/(\[[^\]]+\])\s*\n(?=\[)/g, '');
  return { lyrics, hints };
}

function normalizeSection(line: string): string {
  if (!SONG_SECTION.test(line)) return line;
  const left = line.indexOf('['), right = line.lastIndexOf(']');
  let body = line.slice(left + 1, right).split(' ').map(word => word.length > 1 && word === word.toUpperCase()
    ? word[0].toUpperCase() + word.slice(1).toLowerCase() : word).join(' ');
  body = (body[0]?.toUpperCase() ?? '') + body.slice(1);
  body = body.replace(/-([a-z])/g, (_, letter: string) => `-${letter.toUpperCase()}`);
  return line.slice(0, left + 1) + body + line.slice(right);
}

export function parseFixedRecipeCaption(text: string): { style: string; lyrics: string } {
  const sections: Record<string, string[]> = { tags: [] };
  let current = 'tags';
  for (const line of splitLines(text || '')) {
    const header = HEADER.exec(line);
    if (header) { current = header[1].toLowerCase(); sections[current] ??= []; continue; }
    if (current === 'tags' && !sections.lyrics && SONG_SECTION.test(line)) {
      current = 'lyrics'; sections.lyrics = [];
    }
    sections[current] ??= [];
    sections[current].push(current === 'lyrics' ? normalizeSection(line) : line);
  }
  return { style: sections.tags.join('\n').trim(), lyrics: (sections.lyrics ?? []).join('\n').trim() };
}

export function prepareFixedRecipeCaption(caption: string, lyrics: string, trigger = ''): { style: string; lyrics: string } {
  const parsed = parseFixedRecipeCaption(caption);
  let style = parsed.style.split(/\s+/).join(' ').normalize('NFC');
  const prefix = trigger.trim().normalize('NFC');
  const suffix = style.slice(prefix.length);
  if (prefix && !(style.startsWith(prefix) && (!suffix || /^[\s,.;:!?]/.test(suffix)))) {
    style = style ? `${prefix}, ${style}` : prefix;
  }
  return { style,
    lyrics: cleanFixedRecipeLyrics(lyrics || parsed.lyrics).lyrics.normalize('NFC') };
}
