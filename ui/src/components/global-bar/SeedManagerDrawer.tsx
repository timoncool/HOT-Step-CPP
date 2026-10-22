import { useQuery, useQueryClient } from '@tanstack/react-query';
import { localOptions } from '../../queries/global-models/options';
import { queryManager } from '../../queries/invalidation';
import { apiClient } from '../../services/httpClient';
// SeedManagerDrawer.tsx — Seed save/load popover for STORM Live Controls
// MDMAchine / A&E Concepts 2026
// GPL v3 — safe for public repo
//
// Renders as a floating panel anchored to the seed row.
// Triggered by a 💾 button added next to the existing 🎲 / 🔒 buttons.
//
// Features:
//   - List all saved seeds with search filter
//   - Click a seed → fires onLoad(seed) immediately (no confirm)
//   - Star toggle for favorites, shown first
//   - Save current seed with optional name + description
//   - Delete with single-click (pill turns red, second click confirms)
//   - Imports existing ComfyUI SeedSaver files with zero conversion

import React from 'react';
import { X, Star, Trash2, Save, Shuffle, Loader2 } from 'lucide-react';

// ─── Types ────────────────────────────────────────────────────────────────────

interface SavedSeed {
  name:        string;
  seed:        number;
  saved_at:    string | null;
  description: string;
  tags:        string[];
  favorite:    boolean;
}

interface SeedManagerDrawerProps {
  isOpen:       boolean;
  onClose:      () => void;
  currentSeed:  number;
  /** Called when user clicks a saved seed — apply it immediately */
  onLoad:       (seed: number) => void;
  /** Called when user hits the random-saved-seed button */
  onLoadRandom: (seed: number) => void;
}

// ─── API helpers ──────────────────────────────────────────────────────────────

async function apiSave(name: string, seed: number, description: string): Promise<void> {
  const res = await fetch('/api/seeds', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ name, seed, description }),
  });
  if (!res.ok) throw new Error('save failed');
}

async function apiDelete(name: string): Promise<void> {
  const res = await fetch(`/api/seeds/${encodeURIComponent(name)}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('delete failed');
}

async function apiToggleFavorite(name: string): Promise<boolean> {
  const res = await fetch(`/api/seeds/${encodeURIComponent(name)}/favorite`, { method: 'POST' });
  if (!res.ok) throw new Error('favorite toggle failed');
  const data = await res.json();
  return data.favorite;
}

async function apiRandom(): Promise<{ name: string; seed: number } | null> {
  return apiClient.get('/seeds/random');
}

// ─── Subcomponents ────────────────────────────────────────────────────────────

function fmtDate(iso: string | null): string {
  if (!iso) return '';
  try {
    return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch { return ''; }
}

// ─── Main component ───────────────────────────────────────────────────────────

export const SeedManagerDrawer: React.FC<SeedManagerDrawerProps> = ({
  isOpen, onClose, currentSeed, onLoad, onLoadRandom,
}) => {
  const client = useQueryClient();
  const options = localOptions<{ seeds: SavedSeed[] }>('seeds', '/seeds');
  const query = useQuery({ ...options, enabled: isOpen });
  const seeds = query.data?.seeds ?? [];
  const loading = query.isPending;
  const refresh = () => queryManager.keys(client, options.queryKey);
  const [filter,      setFilter]      = React.useState('');
  const [saveName,    setSaveName]    = React.useState('');
  const [saveDesc,    setSaveDesc]    = React.useState('');
  const [saving,      setSaving]      = React.useState(false);
  const [deleteConfirm, setDeleteConfirm] = React.useState<string | null>(null);
  const [flashLoaded, setFlashLoaded] = React.useState<string | null>(null);
  const [error,       setError]       = React.useState<string | null>(null);

  React.useEffect(() => { if (query.error) setError(query.error.message); }, [query.error]);

  // Clear delete confirm when filter changes
  React.useEffect(() => { setDeleteConfirm(null); }, [filter]);

  if (!isOpen) return null;

  // Sort: favorites first, then alpha
  const filtered = seeds
    .filter(s =>
      !filter ||
      s.name.toLowerCase().includes(filter.toLowerCase()) ||
      s.description.toLowerCase().includes(filter.toLowerCase()) ||
      s.tags.some(t => t.toLowerCase().includes(filter.toLowerCase()))
    )
    .sort((a, b) => {
      if (a.favorite !== b.favorite) return a.favorite ? -1 : 1;
      return a.name.localeCompare(b.name);
    });

  const handleLoad = (s: SavedSeed) => {
    onLoad(s.seed);
    setFlashLoaded(s.name);
    setTimeout(() => setFlashLoaded(null), 1200);
  };

  const handleSave = async () => {
    const name = saveName.trim() || `seed_${currentSeed}`;
    setSaving(true);
    setError(null);
    try {
      await apiSave(name, currentSeed, saveDesc.trim());
      await refresh();
      setSaveName('');
      setSaveDesc('');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'save failed');
    } finally { setSaving(false); }
  };

  const handleDelete = async (name: string) => {
    if (deleteConfirm !== name) { setDeleteConfirm(name); return; }
    setDeleteConfirm(null);
    try {
      await apiDelete(name);
      await refresh();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'delete failed');
    }
  };

  const handleFavorite = async (name: string) => {
    try {
      await apiToggleFavorite(name);
      await refresh();
    } catch {}
  };

  const handleRandom = async () => {
    try {
      const r = await apiRandom();
      if (r) {
        onLoadRandom(r.seed);
        setFlashLoaded(r.name);
        setTimeout(() => setFlashLoaded(null), 1200);
      }
    } catch {}
  };

  return (
    <>
      {/* Backdrop — click to close */}
      <div className="fixed inset-0 z-40 bg-black/40" onClick={onClose} />

      {/* Drawer panel */}
      <div
        className="fixed z-50 w-72 bg-zinc-900 rounded-xl border border-zinc-700 shadow-2xl overflow-hidden"
        style={{ bottom: '3.5rem', right: '1rem' }}   // anchors above seed row; adjust if needed
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-800">
          <span className="text-[11px] font-bold text-zinc-200">🎲 Seed Manager</span>
          <div className="flex items-center gap-1">
            <button onClick={handleRandom} title="Load a random saved seed"
              className="text-[9px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-700 transition-colors flex items-center gap-0.5">
              <Shuffle size={9} /> random
            </button>
            <button onClick={onClose}
              className="p-0.5 rounded text-zinc-600 hover:text-zinc-300 hover:bg-zinc-800 transition-colors">
              <X size={12} />
            </button>
          </div>
        </div>

        {/* Save current seed */}
        <div className="px-3 py-2 border-b border-zinc-800 space-y-1.5">
          <div className="text-[9px] text-zinc-600 uppercase tracking-wider">Save current seed</div>
          <div className="flex items-center gap-1">
            <div className="flex-1 text-[9px] text-amber-400 tabular-nums font-mono bg-zinc-800 rounded px-1.5 py-0.5 border border-zinc-700">
              {currentSeed}
            </div>
          </div>
          <input
            type="text"
            value={saveName}
            onChange={e => setSaveName(e.target.value)}
            placeholder={`name (default: seed_${currentSeed})`}
            onKeyDown={e => { if (e.key === 'Enter') handleSave(); }}
            className="w-full px-1.5 py-0.5 rounded bg-zinc-800 border border-zinc-700 text-[9px] text-zinc-200 placeholder:text-zinc-600 outline-none focus:border-amber-500/40 transition-colors"
          />
          <input
            type="text"
            value={saveDesc}
            onChange={e => setSaveDesc(e.target.value)}
            placeholder="description (optional)"
            onKeyDown={e => { if (e.key === 'Enter') handleSave(); }}
            className="w-full px-1.5 py-0.5 rounded bg-zinc-800 border border-zinc-700 text-[9px] text-zinc-200 placeholder:text-zinc-600 outline-none focus:border-amber-500/40 transition-colors"
          />
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-1 text-[9px] px-2 py-0.5 rounded bg-amber-600 hover:bg-amber-500 text-white font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {saving ? <Loader2 size={9} className="animate-spin" /> : <Save size={9} />}
            Save
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="px-3 py-1.5 text-[9px] text-red-400 bg-red-900/20 border-b border-zinc-800">
            ❌ {error}
          </div>
        )}

        {/* Search */}
        <div className="px-3 py-2 border-b border-zinc-800">
          <input
            type="text"
            value={filter}
            onChange={e => setFilter(e.target.value)}
            placeholder="search seeds…"
            className="w-full px-1.5 py-0.5 rounded bg-zinc-800 border border-zinc-700 text-[9px] text-zinc-300 placeholder:text-zinc-600 outline-none focus:border-zinc-500 transition-colors"
          />
        </div>

        {/* Seed list */}
        <div className="max-h-48 overflow-y-auto hide-scrollbar">
          {loading && (
            <div className="flex items-center justify-center py-4 gap-1.5 text-zinc-600">
              <Loader2 size={12} className="animate-spin" />
              <span className="text-[9px]">loading…</span>
            </div>
          )}

          {!loading && filtered.length === 0 && (
            <div className="text-center py-4 text-[9px] text-zinc-700">
              {seeds.length === 0 ? 'no seeds saved yet' : 'no matches'}
            </div>
          )}

          {!loading && filtered.map(s => (
            <div
              key={s.name}
              className={`group flex items-center gap-1.5 px-3 py-1.5 hover:bg-zinc-800/60 transition-colors border-b border-zinc-800/40 last:border-0 ${
                flashLoaded === s.name ? 'bg-green-900/20' : ''
              }`}
            >
              {/* Favorite star */}
              <button
                onClick={() => handleFavorite(s.name)}
                className={`flex-shrink-0 transition-colors ${
                  s.favorite ? 'text-amber-400 hover:text-amber-300' : 'text-zinc-700 hover:text-zinc-500'
                }`}
                title={s.favorite ? 'Remove from favorites' : 'Add to favorites'}
              >
                <Star size={9} fill={s.favorite ? 'currentColor' : 'none'} />
              </button>

              {/* Load button — name + seed value */}
              <button
                onClick={() => handleLoad(s)}
                className="flex-1 text-left min-w-0"
                title={`Load seed ${s.seed}${s.description ? ` — ${s.description}` : ''}`}
              >
                <div className="flex items-baseline gap-1.5 min-w-0">
                  <span className={`text-[9px] font-medium truncate ${
                    flashLoaded === s.name ? 'text-green-400' : 'text-zinc-300 group-hover:text-white'
                  }`}>
                    {flashLoaded === s.name ? '✓ loaded' : s.name}
                  </span>
                  <span className="text-[8px] text-zinc-600 tabular-nums shrink-0 font-mono">
                    {s.seed}
                  </span>
                </div>
                {(s.description || s.saved_at) && (
                  <div className="flex items-center gap-1 mt-0.5">
                    {s.description && (
                      <span className="text-[7px] text-zinc-600 truncate">{s.description}</span>
                    )}
                    {s.saved_at && (
                      <span className="text-[7px] text-zinc-700 shrink-0">{fmtDate(s.saved_at)}</span>
                    )}
                  </div>
                )}
              </button>

              {/* Delete */}
              <button
                onClick={() => handleDelete(s.name)}
                className={`flex-shrink-0 transition-all rounded px-1 py-0.5 ${
                  deleteConfirm === s.name
                    ? 'text-red-300 bg-red-900/40 text-[8px] font-medium'
                    : 'text-zinc-700 hover:text-red-400 opacity-0 group-hover:opacity-100'
                }`}
                title={deleteConfirm === s.name ? 'Click again to confirm delete' : 'Delete seed'}
              >
                {deleteConfirm === s.name ? '✕ sure?' : <Trash2 size={9} />}
              </button>
            </div>
          ))}
        </div>

        {/* Footer count */}
        <div className="px-3 py-1.5 border-t border-zinc-800">
          <span className="text-[8px] text-zinc-700">
            {seeds.length} seed{seeds.length !== 1 ? 's' : ''} saved
            {seeds.some(s => s.favorite) ? ` · ${seeds.filter(s => s.favorite).length} ★` : ''}
          </span>
        </div>
      </div>
    </>
  );
};
