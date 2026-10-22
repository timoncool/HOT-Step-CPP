// profiles.ts — Parameter profile REST API
//
// A profile is a named snapshot of every generation parameter (the UI's
// 'hot-step-preset' JSON — same format as the preset export file), stored
// server-side so the user can switch configs in-app without juggling files.
// Files are plain preset JSON wrapped with { name, saved_at, data }, so an
// exported preset can be dropped in by hand and vice versa.
//
// Profiles are per engine. Every route takes `?backend=<id>` (POST: a
// `backend` body field); omitted means 'ace'. ACE-Step's live in the folder
// root, where every profile from before this split already is; each other
// engine gets a subfolder named after its backend id.
//
// Mounts at: /api/profiles
// Routes:
//   GET    /api/profiles         — list one engine's profiles (full data inline)
//   GET    /api/profiles/:name   — load one profile
//   POST   /api/profiles         — save/overwrite { name, data, backend }
//   DELETE /api/profiles/:name   — delete profile

import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { config } from '../config.js';
import type {
  ProfileFile, ListProfilesResponse, SaveProfileResponse, RenameProfileResponse, DeleteProfileResponse,
} from '../contracts/profiles.js';

const router = Router();

/** A backend id fit for a folder name; anything else is ACE-Step. */
function backendOf(raw: unknown): string {
  return typeof raw === 'string' && /^[a-z0-9][a-z0-9-]{0,31}$/.test(raw) ? raw : 'ace';
}

function profilesDir(backend: string): string {
  const root = path.join(config.data.dir, 'profiles');
  const dir = backend === 'ace' ? root : path.join(root, backend);
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

// Sanitize name — strip path separators and shell-hostile chars (same regex as seeds.ts)
function safeName(name: string): string {
  return name.replace(/[<>:"/\\|?*\x00-\x1f]/g, '_').slice(0, 100).trim();
}

function profilePath(name: string, backend: string): string {
  return path.join(profilesDir(backend), `${safeName(name)}.json`);
}

function readProfile(name: string, backend: string): ProfileFile | null {
  const p = profilePath(name, backend);
  if (!fs.existsSync(p)) return null;
  try {
    const raw = JSON.parse(fs.readFileSync(p, 'utf8'));
    // Tolerate a bare preset JSON dropped into the folder by hand
    if (raw && typeof raw === 'object' && raw.data === undefined && raw._format === 'hot-step-preset') {
      return { name, saved_at: '', backend, data: raw };
    }
    return { ...raw, backend };
  } catch { return null; }
}

// GET /api/profiles — list all with data inline (profiles are a few KB each)
router.get('/', (req, res) => {
  const backend = backendOf(req.query.backend);
  try {
    const names = fs.readdirSync(profilesDir(backend))
      .filter(f => f.endsWith('.json'))
      .map(f => f.slice(0, -5))
      .sort((a, b) => a.localeCompare(b));
    const profiles = names
      .map(name => readProfile(name, backend))
      .filter((p): p is ProfileFile => p !== null && !!p.data);
    res.json({ profiles, count: profiles.length } satisfies ListProfilesResponse);
  } catch (err: any) {
    console.error('[Profiles] list failed:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/profiles/:name — load one
router.get('/:name', (req, res) => {
  const profile = readProfile(req.params.name, backendOf(req.query.backend));
  if (!profile) {
    res.status(404).json({ error: `profile '${req.params.name}' not found` });
    return;
  }
  res.json(profile);
});

// POST /api/profiles — save/overwrite { name, data }
router.post('/', (req, res) => {
  const { name, data } = req.body ?? {};
  const backend = backendOf(req.body?.backend);
  if (!name || typeof name !== 'string' || !safeName(name)) {
    res.status(400).json({ error: 'name is required' });
    return;
  }
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    res.status(400).json({ error: 'data must be an object' });
    return;
  }
  const profile: ProfileFile = {
    name: safeName(name),
    saved_at: new Date().toISOString(),
    backend,
    data,
  };
  try {
    fs.writeFileSync(profilePath(name, backend), JSON.stringify(profile, null, 2), 'utf8');
    console.log(`[Profiles] Saved '${profile.name}' (${backend})`);
    res.json({ ok: true, name: profile.name, saved_at: profile.saved_at } satisfies SaveProfileResponse);
  } catch (err: any) {
    console.error('[Profiles] save failed:', err.message);
    res.status(500).json({ error: 'failed to write profile' });
  }
});

// PATCH /api/profiles/:name — rename { newName } (data unchanged)
router.patch('/:name', (req, res) => {
  const oldName = req.params.name;
  const backend = backendOf(req.query.backend);
  const { newName } = req.body ?? {};
  if (!newName || typeof newName !== 'string' || !safeName(newName)) {
    res.status(400).json({ error: 'newName is required' });
    return;
  }
  const src = profilePath(oldName, backend);
  if (!fs.existsSync(src)) {
    res.status(404).json({ error: `profile '${oldName}' not found` });
    return;
  }
  const dstName = safeName(newName);
  const dst = profilePath(newName, backend);
  const sameFile = path.resolve(dst) === path.resolve(src);
  if (fs.existsSync(dst) && !sameFile) {
    res.status(409).json({ error: `a profile named '${dstName}' already exists` });
    return;
  }
  const profile = readProfile(oldName, backend);
  if (!profile) {
    res.status(404).json({ error: `profile '${oldName}' not found` });
    return;
  }
  profile.name = dstName;
  try {
    fs.writeFileSync(dst, JSON.stringify(profile, null, 2), 'utf8');
    if (!sameFile) fs.unlinkSync(src);
    console.log(`[Profiles] Renamed '${oldName}' -> '${dstName}'`);
    res.json({ ok: true, name: dstName } satisfies RenameProfileResponse);
  } catch (err: any) {
    console.error('[Profiles] rename failed:', err.message);
    res.status(500).json({ error: 'failed to rename profile' });
  }
});

// DELETE /api/profiles/:name — delete
router.delete('/:name', (req, res) => {
  const p = profilePath(req.params.name, backendOf(req.query.backend));
  if (!fs.existsSync(p)) {
    res.status(404).json({ error: `profile '${req.params.name}' not found` });
    return;
  }
  try {
    fs.unlinkSync(p);
    console.log(`[Profiles] Deleted '${req.params.name}'`);
    res.json({ ok: true, deleted: req.params.name } satisfies DeleteProfileResponse);
  } catch (err: any) {
    console.error('[Profiles] delete failed:', err.message);
    res.status(500).json({ error: 'delete failed' });
  }
});

export default router;
