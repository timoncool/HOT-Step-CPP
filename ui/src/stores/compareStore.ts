/**
 * compareStore.ts - Compare studio state: session list, current session,
 * per-take analysis and diff. useSyncExternalStore pattern (see abCompareStore).
 */

import { useSyncExternalStore } from 'react';
import {
  compareApi,
  type CompareAnalysis, type CompareDiff, type CompareScore, type CompareSession,
  type CompareSessionSummary, type CompareStatus, type CreateSessionInput, type ScoreInput, type ScoreResult,
} from '../services/compareApi';

export interface CompareState {
  sessions: CompareSessionSummary[];
  current: CompareSession | null;
  analysis: Record<string, CompareAnalysis>;
  analysisLoading: Record<string, boolean>;
  diff: CompareDiff | null;
  loading: boolean;
  error: string | null;
}

let _state: CompareState = {
  sessions: [], current: null, analysis: {}, analysisLoading: {}, diff: null, loading: false, error: null,
};

const _listeners = new Set<() => void>();
function notify(): void { _listeners.forEach(cb => cb()); }
function setState(u: Partial<CompareState>): void { _state = { ..._state, ...u }; notify(); }
function subscribe(cb: () => void): () => void { _listeners.add(cb); return () => { _listeners.delete(cb); }; }
function getSnapshot(): CompareState { return _state; }

export function getCompareState(): CompareState { return _state; }
export function useCompareStore(): CompareState { return useSyncExternalStore(subscribe, getSnapshot); }

// ── Pure helpers (exported for testing) ──────────────────────────────────────

/** Local status estimate, same rule as the server: a take counts as scored if
 *  any criterion has a value, keep flag, or non-empty note. */
export function localStatus(session: Pick<CompareSession, 'takes'>): { status: CompareStatus; scored: number; total: number } {
  const total = session.takes.length;
  const scored = session.takes.filter(t =>
    t.scores.some(s => s.value !== null || s.keep !== null || (s.note ?? '').length > 0)).length;
  const status: CompareStatus = total > 0 && scored >= total ? 'scored' : scored > 0 ? 'partial' : 'unscored';
  return { status, scored, total };
}

/** Apply score inputs to a session immutably. Fields absent from an input are left alone. */
export function applyScores(session: CompareSession, items: ScoreInput[]): CompareSession {
  const takes = session.takes.map(t => {
    const mine = items.filter(i => i.songId === t.songId);
    if (!mine.length) return t;
    const scores = [...t.scores];
    for (const it of mine) {
      const idx = scores.findIndex(s => s.criterion === it.criterion);
      const base: CompareScore = idx >= 0 ? scores[idx] : { criterion: it.criterion, value: null, keep: null, note: '' };
      const next: CompareScore = { ...base };
      if ('value' in it) next.value = it.value ?? null;
      if ('keep' in it) next.keep = it.keep ?? null;
      if ('note' in it) next.note = it.note ?? '';
      if (idx >= 0) scores[idx] = next; else scores.push(next);
    }
    return { ...t, scores };
  });
  const s = { ...session, takes };
  // Optimistic: mirror the server auto-archive rule (never un-archives).
  const ls = localStatus(s);
  return { ...s, ...ls, archived: session.archived || ls.status === 'scored' };
}

/** Server result wins for status, archived and counts. */
export function reconcileScoreResult<T extends CompareSessionSummary>(session: T, r: ScoreResult): T {
  return { ...session, status: r.status, archived: r.archived, scored: r.scored, total: r.total };
}

function patchSummary(id: string, fn: (s: CompareSessionSummary) => CompareSessionSummary): CompareSessionSummary[] {
  return _state.sessions.map(s => (s.id === id ? fn(s) : s));
}

function summaryOf(s: CompareSession): CompareSessionSummary {
  const { criteria: _c, takes: _t, ...rest } = s;
  void _c; void _t;
  return rest;
}

function setCurrent(s: CompareSession): void {
  const exists = _state.sessions.some(x => x.id === s.id);
  setState({
    current: s,
    sessions: exists ? patchSummary(s.id, () => summaryOf(s)) : [summaryOf(s), ..._state.sessions],
  });
}

function errMsg(e: unknown): string { return e instanceof Error ? e.message : String(e); }

// ── Actions ──────────────────────────────────────────────────────────────────

export async function loadSessions(opts: { archived?: boolean; group?: string } = {}): Promise<void> {
  setState({ loading: true, error: null });
  try {
    const { sessions } = await compareApi.listSessions(opts);
    setState({ sessions, loading: false });
  } catch (e) { setState({ loading: false, error: errMsg(e) }); }
}

export async function loadSession(id: string): Promise<void> {
  setState({ loading: true, error: null, diff: null });
  try {
    const s = await compareApi.getSession(id);
    setState({ loading: false });
    setCurrent(s);
  } catch (e) { setState({ loading: false, error: errMsg(e) }); }
}

export async function createSession(body: CreateSessionInput): Promise<CompareSession | null> {
  try {
    const s = await compareApi.createSession(body);
    setState({ diff: null, error: null });
    setCurrent(s);
    return s;
  } catch (e) { setState({ error: errMsg(e) }); return null; }
}

export async function deleteSession(id: string): Promise<void> {
  try {
    await compareApi.deleteSession(id);
    setState({
      sessions: _state.sessions.filter(s => s.id !== id),
      current: _state.current?.id === id ? null : _state.current,
    });
  } catch (e) { setState({ error: errMsg(e) }); }
}

export async function addTakes(songIds: string[]): Promise<void> {
  const cur = _state.current;
  if (!cur) return;
  try { setCurrent(await compareApi.addTakes(cur.id, songIds)); } catch (e) { setState({ error: errMsg(e) }); }
}

export async function removeTake(songId: string): Promise<void> {
  const cur = _state.current;
  if (!cur) return;
  try { setCurrent(await compareApi.removeTake(cur.id, songId)); } catch (e) { setState({ error: errMsg(e) }); }
}

/** Optimistic score update, then reconcile status/archived with the server.
 *  On failure, roll back to the pre-update session. */
export async function setScore(items: ScoreInput | ScoreInput[]): Promise<void> {
  const cur = _state.current;
  if (!cur) return;
  const list = Array.isArray(items) ? items : [items];
  const prevSummary = _state.sessions.find(s => s.id === cur.id);
  setCurrent(applyScores(cur, list));
  try {
    const r = await compareApi.putScores(cur.id, list);
    const now = _state.current;
    setState({
      current: now && now.id === cur.id ? reconcileScoreResult(now, r) : now,
      sessions: patchSummary(cur.id, s => reconcileScoreResult(s, r)),
    });
  } catch (e) {
    setState({
      current: _state.current?.id === cur.id ? cur : _state.current,
      sessions: prevSummary ? patchSummary(cur.id, () => prevSummary) : _state.sessions,
      error: errMsg(e),
    });
  }
}

export async function toggleArchive(id?: string): Promise<void> {
  const sid = id ?? _state.current?.id;
  if (!sid) return;
  const was = _state.current?.id === sid
    ? _state.current.archived
    : !!_state.sessions.find(s => s.id === sid)?.archived;
  try {
    const summary = await compareApi.setArchived(sid, !was);
    const now = _state.current;
    setState({
      sessions: patchSummary(sid, () => summary),
      current: now && now.id === sid ? { ...now, ...summary } : now,
    });
  } catch (e) { setState({ error: errMsg(e) }); }
}

const _analysisInflight = new Map<string, Promise<CompareAnalysis | null>>();

/** Lazy per-take analysis; cached in state and deduped while in flight. */
export function loadAnalysis(songId: string): Promise<CompareAnalysis | null> {
  const have = _state.analysis[songId];
  if (have) return Promise.resolve(have);
  const inflight = _analysisInflight.get(songId);
  if (inflight) return inflight;
  setState({ analysisLoading: { ..._state.analysisLoading, [songId]: true } });
  const p = compareApi.getAnalysis(songId)
    .then(a => { setState({ analysis: { ..._state.analysis, [songId]: a } }); return a; })
    .catch((e: unknown) => { setState({ error: errMsg(e) }); return null; })
    .finally(() => {
      _analysisInflight.delete(songId);
      const rest = { ..._state.analysisLoading };
      delete rest[songId];
      setState({ analysisLoading: rest });
    });
  _analysisInflight.set(songId, p);
  return p;
}

export async function loadDiff(a: string, b: string): Promise<void> {
  try { setState({ diff: await compareApi.getDiff(a, b) }); } catch (e) { setState({ error: errMsg(e) }); }
}

export function clearError(): void { setState({ error: null }); }
