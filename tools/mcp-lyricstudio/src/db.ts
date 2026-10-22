// db.ts — Database access layer for mcp-lyricstudio
//
// Connects directly to hotstep.db via better-sqlite3.
// Query functions mirror those in server/src/db/lireekDb.ts.

import Database from 'better-sqlite3';
import { existsSync } from 'node:fs';
import path from 'path';
// Canonical key spelling, shared with the in-app path via the re-exported
// prompts module (server/src/services/lireek/prompts.ts).
import { normalizeKeyScale } from './prompts.js';
import { refreshSidecarCaptions } from '../../../server/src/services/lireek/mm3CaptionSync.js';
import { locateData } from '../../../server/src/dataroot/locate.js';

let db: Database.Database;

/**
 * Resolve without running app startup or applying pending data moves.
 * HOTSTEP_DB is an explicit override; otherwise use the app's shared resolver.
 */
export function resolveDbPath(env: NodeJS.ProcessEnv = process.env): { dbPath: string; source: string } {
  if (env.HOTSTEP_DB) {
    return { dbPath: path.resolve(env.HOTSTEP_DB), source: 'HOTSTEP_DB override' };
  }
  const report = locateData({ env });
  if (report.pointerError) {
    throw new Error(`[mcp-lyricstudio] Cannot read data-folder pointer ${report.pointerPath}: ${report.pointerError}`);
  }
  if (report.pendingMove) {
    const move = report.pendingMove;
    throw new Error(`[mcp-lyricstudio] Pending data-folder ${move.kind} to ${move.destination} in ${report.pointerPath}. Finish the operation in HOT-Step before connecting.${move.error ? ` Reason: ${move.error}` : ''}`);
  }
  if (!report.children) {
    throw new Error(`[mcp-lyricstudio] No database location resolved from ${report.pointerPath}. Configure the data folder in HOT-Step or set HOTSTEP_DB.`);
  }
  return {
    dbPath: report.children.db,
    source: `${report.source} (${report.pointerPath}; settings: ${report.children.envFile})`,
  };
}

export function initDb(): void {
  const { dbPath, source } = resolveDbPath();
  console.error(`[mcp-lyricstudio] Opening database: ${dbPath} from ${source}`);
  if (!existsSync(dbPath)) {
    throw new Error(`[mcp-lyricstudio] Database not found: ${dbPath} from ${source}. Check the configured data folder or set HOTSTEP_DB to the existing hotstep.db file.`);
  }
  db = new Database(dbPath, { fileMustExist: true });
  db.pragma('journal_mode = WAL');
  db.pragma('foreign_keys = ON');
}

export function getDb(): Database.Database {
  if (!db) throw new Error('Database not initialized');
  return db;
}

// ── Artists ──────────────────────────────────────────────────────────────────

export function listArtists(): any[] {
  return getDb().prepare(
    `SELECT a.*, COUNT(ls.id) AS lyrics_set_count
     FROM artists a LEFT JOIN lyrics_sets ls ON ls.artist_id = a.id
     GROUP BY a.id ORDER BY a.name`
  ).all();
}

export function getArtist(id: number): any {
  return getDb().prepare('SELECT * FROM artists WHERE id = ?').get(id);
}

// ── Lyrics Sets ─────────────────────────────────────────────────────────────

export function getLyricsSet(id: number): any {
  const row = getDb().prepare(
    `SELECT ls.*, a.name as artist_name FROM lyrics_sets ls
     JOIN artists a ON a.id = ls.artist_id WHERE ls.id = ?`
  ).get(id) as any;
  if (!row) return null;
  row.songs = JSON.parse(row.songs);
  row.total_songs = row.songs.length;
  refreshSidecarCaptions(getDb(), row);
  return row;
}

export function getLyricsSets(artistId?: number): any[] {
  const db = getDb();
  // profile_id is NULL when the set has never been profiled — this is what makes
  // "which artists still need a profile?" answerable without querying SQLite directly.
  const profileSubquery =
    `(SELECT p.id FROM profiles p WHERE p.lyrics_set_id = ls.id
      ORDER BY p.created_at DESC LIMIT 1) AS profile_id`;
  const query = artistId
    ? db.prepare(
        `SELECT ls.*, a.name as artist_name, ${profileSubquery} FROM lyrics_sets ls
         JOIN artists a ON a.id = ls.artist_id
         WHERE ls.artist_id = ? ORDER BY ls.fetched_at DESC`
      )
    : db.prepare(
        `SELECT ls.*, a.name as artist_name, ${profileSubquery} FROM lyrics_sets ls
         JOIN artists a ON a.id = ls.artist_id
         ORDER BY a.name, ls.fetched_at DESC`
      );
  const rows = (artistId ? query.all(artistId) : query.all()) as any[];
  return rows.map(r => {
    const songs = JSON.parse(r.songs);
    const { songs: _, ...rest } = r;
    return { ...rest, total_songs: songs.length };
  });
}

// ── Profiles ────────────────────────────────────────────────────────────────

export function listProfiles(artistId?: number): any[] {
  const db = getDb();
  let rows: any[];
  if (artistId) {
    rows = db.prepare(
      `SELECT p.*, ls.album, ls.artist_id, a.name as artist_name
       FROM profiles p
       JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
       JOIN artists a ON a.id = ls.artist_id
       WHERE ls.artist_id = ?
       ORDER BY p.created_at DESC`
    ).all(artistId);
  } else {
    rows = db.prepare(
      `SELECT p.*, ls.album, ls.artist_id, a.name as artist_name
       FROM profiles p
       JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
       JOIN artists a ON a.id = ls.artist_id
       ORDER BY a.name, p.created_at DESC`
    ).all();
  }
  return rows.map(r => ({
    ...r,
    profile_data: JSON.parse(r.profile_data),
  }));
}

export function getProfile(id: number): any {
  const row = getDb().prepare(
    `SELECT p.*, ls.album, ls.artist_id, a.name as artist_name
     FROM profiles p
     JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
     JOIN artists a ON a.id = ls.artist_id
     WHERE p.id = ?`
  ).get(id) as any;
  if (!row) return null;
  row.profile_data = JSON.parse(row.profile_data);
  return row;
}

export function saveProfile(
  lyricsSetId: number, provider: string, model: string, profileData: any
): any {
  const now = new Date().toISOString();
  const result = getDb().prepare(
    'INSERT INTO profiles (lyrics_set_id, provider, model, profile_data, created_at) VALUES (?, ?, ?, ?, ?)'
  ).run(lyricsSetId, provider, model, JSON.stringify(profileData), now);
  return {
    id: result.lastInsertRowid, lyrics_set_id: lyricsSetId,
    provider, model, profile_data: profileData, created_at: now,
  };
}

// ── Generations ─────────────────────────────────────────────────────────────

export interface SaveGenerationParams {
  profileId: number;
  provider: string;
  model: string;
  lyrics: string;
  title?: string;
  subject?: string;
  bpm?: number;
  key?: string;
  caption?: string;
  /** MiniMax-Music3 Structured Caption. Separate field, not a variant of
   *  `caption` — the two backends want differently-formatted text. */
  captionMm3?: string;
  /** YuE2 planner caption — one sentence, fixed order. Third format. */
  captionYue2?: string;
  duration?: number;
  systemPrompt?: string;
  userPrompt?: string;
  parentGenerationId?: number | null;
}

export function saveGeneration(p: SaveGenerationParams): any {
  // Canonicalise the key on write — the engine's metadata FSM only accepts a
  // lower-case mode, so "D Major" is out of vocabulary. Mirrors the in-app
  // path in server/src/db/lireekDb.ts.
  const key = normalizeKeyScale(p.key);
  const now = new Date().toISOString();
  const result = getDb().prepare(
    `INSERT INTO generations
     (profile_id, provider, model, extra_instructions, title, subject, bpm, key, caption, caption_mm3, caption_yue2, duration, lyrics, system_prompt, user_prompt, parent_generation_id, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
  ).run(
    p.profileId, p.provider, p.model, null,
    p.title ?? '', p.subject ?? '', p.bpm ?? 0, key, p.caption ?? '', p.captionMm3 ?? '', p.captionYue2 ?? '', p.duration ?? 0,
    p.lyrics, p.systemPrompt ?? '', p.userPrompt ?? '', p.parentGenerationId ?? null, now,
  );
  return {
    id: result.lastInsertRowid, profile_id: p.profileId, provider: p.provider,
    model: p.model, title: p.title ?? '', subject: p.subject ?? '',
    bpm: p.bpm ?? 0, key, caption: p.caption ?? '',
    caption_mm3: p.captionMm3 ?? '', caption_yue2: p.captionYue2 ?? '', duration: p.duration ?? 0,
    lyrics: p.lyrics, parent_generation_id: p.parentGenerationId ?? null, created_at: now,
  };
}

export function getGeneration(id: number): any {
  return (getDb().prepare('SELECT * FROM generations WHERE id = ?').get(id) as any) ?? null;
}

export function listGenerations(profileId?: number, artistId?: number, limit = 20): any[] {
  const db = getDb();
  if (profileId) {
    return db.prepare(
      `SELECT g.*, a.name AS artist_name, ls.album
       FROM generations g
       JOIN profiles p ON p.id = g.profile_id
       JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
       JOIN artists a ON a.id = ls.artist_id
       WHERE g.profile_id = ?
       ORDER BY g.created_at DESC LIMIT ?`
    ).all(profileId, limit);
  }
  if (artistId) {
    return db.prepare(
      `SELECT g.*, a.name AS artist_name, ls.album
       FROM generations g
       JOIN profiles p ON p.id = g.profile_id
       JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
       JOIN artists a ON a.id = ls.artist_id
       WHERE ls.artist_id = ?
       ORDER BY g.created_at DESC LIMIT ?`
    ).all(artistId, limit);
  }
  return db.prepare(
    `SELECT g.*, a.name AS artist_name, ls.album
     FROM generations g
     JOIN profiles p ON p.id = g.profile_id
     JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
     JOIN artists a ON a.id = ls.artist_id
     ORDER BY g.created_at DESC LIMIT ?`
  ).all(limit);
}

/**
 * Get generation history for diversity tracking (used subjects, BPMs, keys, titles).
 */
export function getGenerationHistory(artistId: number): {
  usedSubjects: string[];
  usedBpms: number[];
  usedKeys: string[];
  usedTitles: string[];
  usedDurations: number[];
} {
  const rows = getDb().prepare(
    `SELECT g.subject, g.bpm, g.key, g.title, g.duration
     FROM generations g
     JOIN profiles p ON p.id = g.profile_id
     JOIN lyrics_sets ls ON ls.id = p.lyrics_set_id
     WHERE ls.artist_id = ?`
  ).all(artistId) as any[];

  return {
    usedSubjects: rows.map(r => r.subject).filter(Boolean),
    usedBpms: rows.map(r => r.bpm).filter((b: number) => b > 0),
    usedKeys: rows.map(r => r.key).filter(Boolean),
    usedTitles: rows.map(r => r.title).filter(Boolean),
    usedDurations: rows.map(r => r.duration).filter((d: number) => d > 0),
  };
}
