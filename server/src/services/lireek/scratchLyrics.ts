import { z } from 'zod/v4';
import * as db from '../../db/lireekDb.js';
import { registerWorkflowKind } from '../../routes/workflows.js';
import type { WorkflowContext } from '../workflows/workflowJobs.js';
import { getProvider } from './llm/registry.js';
import { postprocessLyrics, stripThinkingBlocks, estimateDuration } from './llm/postprocess.js';
import { buildScratchLyricsPrompt, SCRATCH_LYRICS_SYSTEM_PROMPT } from './prompts.js';

export const scratchLyricsInput = z.object({
  lyricsSetId: z.number().int().positive(),
  provider: z.string().trim().min(1).default('ollama'),
  model: z.string().trim().max(200).optional(),
  theme: z.string().trim().max(4000).default(''),
  style: z.string().trim().max(2000).default(''),
  structure: z.string().trim().max(2000).default(''),
  existingLyrics: z.string().trim().max(12000).default(''),
}).refine(value => !!(value.theme || value.existingLyrics), {
  message: 'Enter a theme or some starter lyrics',
});
export type ScratchLyricsInput = z.infer<typeof scratchLyricsInput>;

const outputSchema = z.object({
  title: z.string().trim().min(1).max(300),
  lyrics: z.string().trim().min(1).max(50000),
  caption: z.string().trim().max(4000).default(''),
  bpm: z.number().finite().min(30).max(300).optional(),
  key: z.string().trim().max(100).default(''),
});

export async function writeScratchLyrics(input: ScratchLyricsInput) {
  const provider = getProvider(input.provider);
  const userPrompt = buildScratchLyricsPrompt(input);
  const raw = stripThinkingBlocks(await provider.call(
    SCRATCH_LYRICS_SYSTEM_PROMPT, userPrompt, input.model, undefined, { noThink: true },
  )).replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();
  let value: unknown;
  try { value = JSON.parse(raw); }
  catch { throw new Error('The lyric model returned invalid JSON. Try generating again.'); }
  const parsed = outputSchema.safeParse(value);
  if (!parsed.success) throw new Error('The lyric model returned incomplete lyrics or invalid metadata. Try generating again.');
  if (!parsed.data.lyrics.split('\n').some(line => line.trim() && !/^\[.*\]$/.test(line.trim()))) {
    throw new Error('The lyric model returned section tags without lyrics. Try generating again.');
  }
  return { ...parsed.data, lyrics: postprocessLyrics(parsed.data.lyrics),
    model: input.model || provider.defaultModel, systemPrompt: SCRATCH_LYRICS_SYSTEM_PROMPT, userPrompt };
}

export async function runScratchLyrics(ctx: WorkflowContext<ScratchLyricsInput>, write = writeScratchLyrics) {
  const input = ctx.input;
  const database = db.getLireekDb();
  database.exec(`CREATE TABLE IF NOT EXISTS lyric_scratch_results (
    job_id TEXT PRIMARY KEY, result TEXT NOT NULL
  )`);
  const previous = database.prepare('SELECT result FROM lyric_scratch_results WHERE job_id = ?')
    .get(ctx.jobId) as { result: string } | undefined;
  if (previous) return JSON.parse(previous.result) as Record<string, unknown>;
  if (!db.getLyricsSet(input.lyricsSetId)) throw new Error('Album not found');
  ctx.throwIfCancelled();
  ctx.emit('phase', { phase: 'Writing lyrics' });
  const result = await write(input);
  ctx.throwIfCancelled();
  return database.transaction(() => {
    const album = db.getLyricsSet(input.lyricsSetId);
    if (!album) throw new Error('Album was deleted during generation');
    // A prompt profile supplies the existing library relationship, without an LLM analysis pass.
    const profile = db.saveProfile(input.lyricsSetId, input.provider, result.model, {
      artist: album.artist_name, album: album.album, themes: input.theme ? [input.theme] : [],
      common_subjects: [], rhyme_schemes: [], avg_verse_lines: 4, avg_chorus_lines: 4,
      tone_and_mood: input.style, structural_patterns: input.structure,
      additional_notes: 'Written from a prompt', style_caption: result.caption,
    });
    const saved = db.saveGeneration({
      profileId: profile.id as number, provider: input.provider, model: result.model,
      lyrics: result.lyrics, title: result.title, subject: input.theme, caption: result.caption,
      bpm: result.bpm, key: result.key,
      duration: estimateDuration(result.lyrics, result.bpm || 120),
      systemPrompt: result.systemPrompt, userPrompt: result.userPrompt,
    });
    database.prepare('INSERT INTO lyric_scratch_results (job_id, result) VALUES (?, ?)')
      .run(ctx.jobId, JSON.stringify(saved));
    return saved;
  })();
}

registerWorkflowKind({ kind: 'lyric-from-prompt', input: scratchLyricsInput, run: runScratchLyrics });
