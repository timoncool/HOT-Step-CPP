// instance.ts — which server process owns a job, and whether that process is
// still alive.
//
// Two copies of the app (dev and prod) may run at once on one data root and
// one hotstep.db. Every job record a server writes carries an owner stamp
// (this process's instance id and PID); every startup sweep, poll, claim and
// cancel acts only on jobs this process owns, or on jobs whose owner is dead.
//
//   instance  a random id per server start. A restart is a new instance.
//   copy      which copy of the app this is (host and HTTP port). Stable over
//             restarts, so a pending item queued in one copy waits for that
//             copy rather than being taken by the other.
//
// Liveness: the owner's row in server_instances, a heartbeat every 10 s, and
// (on the same host) its PID. The PID catches a crash at once; a stale
// heartbeat catches a PID the OS has since given to another process.
//
// Legacy rule: a job record with no owner was written by a server from before
// ownership existed, so the process that wrote it is gone (upgrading restarted
// it). It counts as dead-owned and is recovered as before. Running a copy from
// before ownership beside one from after it is unsupported: the old copy's
// sweeps still fail the other copy's jobs.
//
// Without a registry (unit tests that never open one) every owner other than
// this process counts as dead: the single-server behaviour.

import os from 'node:os';
import { randomUUID } from 'node:crypto';
import type Database from 'better-sqlite3';

export interface Owner { instance: string; pid: number }

const HEARTBEAT_MS = 10_000;
/** A heartbeat older than this is a dead owner. Generous, because a long
 *  synchronous step can hold the event loop and delay a beat. */
export const STALE_MS = 120_000;

let current: Owner = { instance: randomUUID(), pid: process.pid };
let copy = `${os.hostname()}:${process.env.SERVER_PORT || '3001'}`;
let registry: Database.Database | null = null;
let timer: NodeJS.Timeout | null = null;
let now: () => number = Date.now;

interface Row { id: string; pid: number; host: string; copy: string; heartbeat_at: number; stopped_at: number | null }

export function ensureInstanceSchema(db: Database.Database): void {
  db.exec(`CREATE TABLE IF NOT EXISTS server_instances (
    id TEXT PRIMARY KEY,
    pid INTEGER NOT NULL,
    host TEXT NOT NULL,
    copy TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    heartbeat_at INTEGER NOT NULL,
    stopped_at INTEGER
  )`);
}

/** Register this process. Called once at startup, after initDb(). */
export function startInstance(db: Database.Database, opts: { copy?: string; now?: () => number; heartbeat?: boolean } = {}): void {
  registry = db;
  if (opts.copy) copy = opts.copy;
  if (opts.now) now = opts.now;
  ensureInstanceSchema(db);
  const t = now();
  // A week of finished rows is plenty for any message that names one.
  db.prepare('DELETE FROM server_instances WHERE stopped_at IS NOT NULL AND stopped_at < ?').run(t - 7 * 86_400_000);
  db.prepare(`INSERT INTO server_instances (id, pid, host, copy, started_at, heartbeat_at) VALUES (?, ?, ?, ?, ?, ?)`)
    .run(current.instance, current.pid, os.hostname(), copy, t, t);
  if (opts.heartbeat !== false && !timer) {
    timer = setInterval(beat, HEARTBEAT_MS);
    timer.unref();
  }
  console.log(`[Instance] ${current.instance} (pid ${current.pid}, copy ${copy})`);
}

function beat(): void {
  try { registry?.prepare('UPDATE server_instances SET heartbeat_at = ? WHERE id = ?').run(now(), current.instance); }
  catch (err) { console.warn(`[Instance] heartbeat failed: ${(err as Error).message}`); }
}

/** Mark this process stopped, so the other copy recovers its jobs at once. */
export function stopInstance(): void {
  if (timer) clearInterval(timer);
  timer = null;
  try { registry?.prepare('UPDATE server_instances SET stopped_at = ? WHERE id = ?').run(now(), current.instance); }
  catch { /* the database may already be closed */ }
}

export const owner = (): Owner => ({ ...current });
export const instanceId = (): string => current.instance;
export const copyId = (): string => copy;
export const isMine = (o: Partial<Owner> | null | undefined): boolean => o?.instance === current.instance;

function pidAlive(pid: number): boolean {
  try { process.kill(pid, 0); return true; }
  catch (err) { return (err as NodeJS.ErrnoException).code === 'EPERM'; }
}

/** True while the process that owns a job is still running. No owner (a
 *  legacy record) is dead; this process is alive. */
export function ownerAlive(o: Partial<Owner> | null | undefined): boolean {
  if (!o?.instance) return false;
  if (o.instance === current.instance) return true;
  if (!registry) return false;
  let row: Row | undefined;
  try { row = registry.prepare('SELECT * FROM server_instances WHERE id = ?').get(o.instance) as Row | undefined; }
  catch { return false; }
  if (!row || row.stopped_at !== null) return false;
  if (now() - row.heartbeat_at > STALE_MS) return false;
  return row.host !== os.hostname() || pidAlive(row.pid);
}

/** Same, by instance id alone (database columns keep only the id). */
export const instanceAlive = (id: string | null | undefined): boolean => ownerAlive(id ? { instance: id } : null);

/** "the copy on host:port" for a message about a job another copy owns. */
export function describeOwner(o: Partial<Owner> | string | null | undefined): string {
  const id = typeof o === 'string' ? o : o?.instance;
  let row: Row | undefined;
  try { row = id ? registry?.prepare('SELECT * FROM server_instances WHERE id = ?').get(id) as Row | undefined : undefined; }
  catch { row = undefined; }
  return row ? `the other HOT-Step copy (${row.copy}, pid ${row.pid})` : 'another HOT-Step copy';
}

// ── Claims ────────────────────────────────────────────────────────────────
// A named resource (e.g. "dataset:<id>") one process at a time may work on.
// Within a process the caller's own bookkeeping decides; a claim only keeps
// the other copy out. A claim held by a dead process is free.

interface ClaimRow { key: string; holder: string; instance: string; claimed_at: number }

function ensureClaims(db: Database.Database): void {
  db.exec(`CREATE TABLE IF NOT EXISTS instance_claims (
    key TEXT PRIMARY KEY, holder TEXT NOT NULL, instance TEXT NOT NULL, claimed_at INTEGER NOT NULL)`);
}

/** Take `key` for `holder` (a job id). Returns null on success, or the claim
 *  another live process holds. `stillActive` lets the caller say a recorded
 *  holder has in fact finished (a claim its owner failed to release). */
export function claim(key: string, holder: string, stillActive: (holder: string) => boolean = () => true): ClaimRow | null {
  const db = registry;
  if (!db) return null;
  ensureClaims(db);
  return db.transaction(() => {
    const row = db.prepare('SELECT * FROM instance_claims WHERE key = ?').get(key) as ClaimRow | undefined;
    if (row && row.instance !== current.instance && instanceAlive(row.instance) && stillActive(row.holder)) return row;
    db.prepare('INSERT OR REPLACE INTO instance_claims (key, holder, instance, claimed_at) VALUES (?, ?, ?, ?)')
      .run(key, holder, current.instance, now());
    return null;
  }).immediate();
}

/** The claim on `key` another live process holds, or null. */
export function foreignClaim(key: string, stillActive: (holder: string) => boolean = () => true): ClaimRow | null {
  if (!registry) return null;
  let row: ClaimRow | undefined;
  try { row = registry.prepare('SELECT * FROM instance_claims WHERE key = ?').get(key) as ClaimRow | undefined; }
  catch { return null; }   // no claims table yet: nothing claimed
  return row && row.instance !== current.instance && instanceAlive(row.instance) && stillActive(row.holder) ? row : null;
}

/** Give `key` back, if `holder` of this process still has it. */
export function release(key: string, holder: string): void {
  try { registry?.prepare('DELETE FROM instance_claims WHERE key = ? AND holder = ? AND instance = ?').run(key, holder, current.instance); }
  catch { /* no claims table, or the database is closed */ }
}

/** Tests: act as a freshly started process (new instance id), as a restart
 *  would. The old instance is marked stopped when a registry is open. */
export function restartInstanceForTests(opts: { pid?: number } = {}): Owner {
  const before = current;
  try { registry?.prepare('UPDATE server_instances SET stopped_at = ? WHERE id = ?').run(now(), before.instance); }
  catch { /* no row */ }
  current = { instance: randomUUID(), pid: opts.pid ?? process.pid };
  if (registry) registry.prepare(`INSERT INTO server_instances (id, pid, host, copy, started_at, heartbeat_at) VALUES (?, ?, ?, ?, ?, ?)`)
    .run(current.instance, current.pid, os.hostname(), copy, now(), now());
  return before;
}

/** Tests: drop the registry and timer (single-server behaviour again). */
export function resetInstanceForTests(): void {
  if (timer) clearInterval(timer);
  timer = null;
  registry = null;
  now = Date.now;
}
