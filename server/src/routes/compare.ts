// compare.ts - Compare studio: A/B listening sessions, scoring, audio analysis.
// Mounts at: /api/compare
// Routes:
//   GET    /api/compare/sessions                   list (?archived=0|1, ?group=)
//   POST   /api/compare/sessions                   create {name, songIds[], criteria?, baselineSongId?, group?}
//   GET    /api/compare/sessions/:id               session with takes, params and scores
//   DELETE /api/compare/sessions/:id               delete session, takes and scores
//   POST   /api/compare/sessions/:id/takes         add {songIds[]}
//   DELETE /api/compare/sessions/:id/takes/:songId remove one take
//   PUT    /api/compare/sessions/:id/scores        upsert [{songId, criterion, value?, keep?, note?}]
//   POST   /api/compare/sessions/:id/archive       {archived: bool}
//   GET    /api/compare/analysis/:songId           loudness/spectral metrics + peaks (cached)
//   GET    /api/compare/diff?a=&b=                 per-segment difference trace between two songs
//   GET    /api/compare/songs?q=&limit=&offset=    paginated library search for the session picker

import { Router, type Response } from 'express';
import { diffTrace } from '../services/compare/analyzer.js';
import * as store from '../services/compare/store.js';

const router = Router();

function fail(res: Response, err: any, what: string) {
  if (err instanceof store.NotFoundError) return res.status(404).json({ error: err.message });
  if (err instanceof store.BadRequestError) return res.status(400).json({ error: err.message });
  console.error(`[Compare] ${what} failed:`, err?.message);
  return res.status(500).json({ error: err?.message || String(err) });
}

router.get('/songs', (req, res) => {
  try {
    const q = typeof req.query.q === 'string' ? req.query.q : '';
    const limit = parseInt(String(req.query.limit ?? ''), 10) || 25;
    const offset = parseInt(String(req.query.offset ?? ''), 10) || 0;
    res.json(store.searchSongs({ q, limit, offset }));
  } catch (err) { fail(res, err, 'song search'); }
});

router.get('/sessions', (req, res) => {
  try {
    const a = req.query.archived;
    const archived = a === undefined ? undefined : a === '1' || a === 'true';
    const group = typeof req.query.group === 'string' ? req.query.group : undefined;
    res.json({ sessions: store.listSessions({ archived, group }) });
  } catch (err) { fail(res, err, 'list'); }
});

router.post('/sessions', (req, res) => {
  try { res.status(201).json(store.createSession(req.body)); } catch (err) { fail(res, err, 'create'); }
});

router.get('/sessions/:id', (req, res) => {
  try { res.json(store.getSession(req.params.id)); } catch (err) { fail(res, err, 'get'); }
});

router.delete('/sessions/:id', (req, res) => {
  try { store.deleteSession(req.params.id); res.json({ ok: true }); } catch (err) { fail(res, err, 'delete'); }
});

router.post('/sessions/:id/takes', (req, res) => {
  try { res.json(store.addTakes(req.params.id, req.body?.songIds)); } catch (err) { fail(res, err, 'add takes'); }
});

router.delete('/sessions/:id/takes/:songId', (req, res) => {
  try { res.json(store.removeTake(req.params.id, req.params.songId)); } catch (err) { fail(res, err, 'remove take'); }
});

router.put('/sessions/:id/scores', (req, res) => {
  try { res.json(store.putScores(req.params.id, req.body)); } catch (err) { fail(res, err, 'scores'); }
});

router.post('/sessions/:id/archive', (req, res) => {
  try { res.json(store.setArchived(req.params.id, req.body?.archived)); } catch (err) { fail(res, err, 'archive'); }
});

router.get('/analysis/:songId', async (req, res) => {
  try {
    const { cached, result } = await store.getAnalysis(req.params.songId);
    res.json({ songId: req.params.songId, cached, ...result });
  } catch (err) { fail(res, err, 'analysis'); }
});

router.get('/diff', async (req, res) => {
  try {
    const a = typeof req.query.a === 'string' ? req.query.a : '';
    const b = typeof req.query.b === 'string' ? req.query.b : '';
    if (!a || !b) throw new store.BadRequestError('a and b song ids are required');
    const pa = store.songAudioPath(a);
    const pb = store.songAudioPath(b);
    res.json({ a, b, ...(await store.runQueued(() => diffTrace(pa, pb))) });
  } catch (err) { fail(res, err, 'diff'); }
});

export default router;
