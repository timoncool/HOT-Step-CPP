import React, { useEffect, useRef, useState } from 'react';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
import { EditableSlider } from '../shared/EditableSlider';
import { transposeKey } from './coverStudioUtils';
import type { Yue2SourceTrack } from '../../utils/yue2CaptionSource';
import type { Artist, AlbumPreset } from '../../services/lireekApi';
import { CoverArtistGrid } from './CoverArtistGrid';

interface Props {
  caption: string;
  onCaptionChange: (value: string) => void;
  captionMode: string;
  onCaptionMode: (value: string) => void;
  artists: Artist[];
  loadingArtists: boolean;
  selectedArtistId: number | null;
  onSelectArtist: (artist: Artist) => void;
  albums: Array<{ lsId: number; album: string; preset: AlbumPreset | null }>;
  selectedLyricsSetId: number | null;
  onSelectAlbum: (lyricsSetId: number) => void;
  coveringBand: string;
  onCoveringBand: (value: string) => void;
  captionTracks: Yue2SourceTrack[];
  instrumental: boolean;
  onInstrumentalChange: (value: boolean) => void;
  renderMode: 'base' | 'artist' | 'custom';
  /** Opens the Build-by-section page. */
  onOpenBuild: () => void;
  /** A build in progress, as one line, or ''. */
  buildStatus: string;
  onStartOverBuild?: () => void;
  onRenderMode: (value: 'base' | 'artist' | 'custom') => void;
  ar: string;
  nar: string;
  onAr: (value: string) => void;
  onNar: (value: string) => void;
  arOptions: Array<{ value: string; label: string }>;
  narOptions: Array<{ value: string; label: string }>;
  keepChords: boolean;
  onKeepChords: (value: boolean) => void;
  scoreAbc: string;
  scoreKeyLabel: string;
  voices: 'vocal' | 'both';
  onVoices: (value: 'vocal' | 'both') => void;
  tempoMode: 'free' | 'source' | 'set';
  onTempoMode: (value: 'free' | 'source' | 'set') => void;
  bpm: number;
  onBpm: (value: number) => void;
  keyShift: number;
  onKeyShift: (value: number) => void;
  cfgScale: number;
  onCfgScale: (value: number) => void;
  arStrength: number;
  onArStrength: (value: number) => void;
  lyricSchedule: 'off' | 'bias' | 'mask';
  onLyricSchedule: (value: 'off' | 'bias' | 'mask') => void;
  canGenerate: boolean;
  isGenerating: boolean;
  onGenerate: () => void;
  /** Under the Generate button (the Drafts fold). */
  footer?: React.ReactNode;
}

/** Cover Studio's own AR strength, 0 to 10. It replaces the adapter picker's
 *  AR scale in every Cover request; the same control sits on Prepare and Build. */
export const COVER_AR_DEFAULT = 0.6;
export const clampCoverAr = (v: number) => Number.isFinite(v) ? Math.max(0, Math.min(10, v)) : COVER_AR_DEFAULT;
export const CoverArStrength: React.FC<{ value: number; onChange: (v: number) => void; disabled?: boolean }> = ({ value, onChange, disabled }) =>
  <EditableSlider label="AR strength" value={value} min={0} max={10} step={0.05}
    onChange={v => onChange(clampCoverAr(v))} defaultValue={COVER_AR_DEFAULT} autoLabel=""
    tooltip="How hard the AR adapter steers the musical plan, for covers only (the adapter picker's AR scale is ignored here). Values above 0.6 hurt adherence to the score: on one seed, 1.0 matched 11% of chords against 89% at 0.6."
    disabled={disabled} />;

/** Column width in px, padding included, above which the settings switch from
 *  one column to paired rows. */
export const YUE2_WIDE_PANEL_PX = 500;

export const Yue2CoverPanel: React.FC<Props> = p => {
  const keyLabel = p.scoreKeyLabel;
  const sourceBpm = Number(p.scoreAbc.match(/^Q:[^=\r\n]*=\s*(\d+(?:\.\d+)?)/m)?.[1] ?? 120);
  const rootRef = useRef<HTMLDivElement>(null);
  const [wide, setWide] = useState(false);
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => setWide(el.clientWidth > YUE2_WIDE_PANEL_PX));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const renderModeEl = <div className="space-y-2">
    <ParamLabel label="Render mode" info="Choose base YuE2, a trained Lyric Studio artist, or a custom adapter pair." />
    <div role="group" aria-label="Render mode" className="flex rounded-lg border border-zinc-200 dark:border-white/10 overflow-hidden">
      {([['base', 'Base YuE2'], ['artist', 'Existing Artist'], ['custom', 'Custom']] as const).map(([value, label]) =>
        <button key={value} type="button" aria-pressed={p.renderMode === value} onClick={() => p.onRenderMode(value)}
          className={`flex-1 px-2 py-2 text-xs ${p.renderMode === value ? 'bg-pink-500/20 text-pink-700 dark:text-pink-300' : 'text-zinc-600 dark:text-zinc-400'}`}>{label}</button>)}
    </div>
  </div>;
  const artistGridEl = p.renderMode === 'artist'
    && <CoverArtistGrid artists={p.artists} loading={p.loadingArtists} selectedArtistId={p.selectedArtistId} onSelectArtist={p.onSelectArtist} />;
  // What names the voice besides the artist grid: the album, or the covering
  // band plus (Custom) the adapter pair. Empty for an artist with one album.
  const sourceEl = p.renderMode === 'artist'
    ? p.albums.length > 1 && <div className="space-y-2">
      <ParamLabel label="Album" info="Choose which Lyric Studio album's paired YuE2 adapters to load." />
      <StyledSelect accent="pink" value={p.selectedLyricsSetId ?? ''} onChange={v => p.onSelectAlbum(Number(v))}
        options={p.albums.map(a => ({ value: a.lsId, label: a.album }))} />
    </div>
    : <div className="space-y-2">
      <ParamLabel label="Covering band" info="Used in the saved filename." />
      <input value={p.coveringBand} onChange={e => p.onCoveringBand(e.target.value)}
        className="w-full rounded-xl bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 px-3 py-2 text-xs text-zinc-900 dark:text-white" />
      {p.renderMode === 'custom' && <>
        <ParamLabel label="AR composer" info="The AR adapter steers the musical plan and semantic tokens." />
        <StyledSelect accent="pink" value={p.ar} onChange={p.onAr} options={p.arOptions} />
        <ParamLabel label="NAR renderer" info="The NAR adapter steers the audio rendering stage." />
        <StyledSelect accent="pink" value={p.nar} onChange={p.onNar} options={p.narOptions} />
      </>}
    </div>;
  const captionSourceEl = p.captionTracks.length > 0 && p.renderMode !== 'base' && <div className="space-y-2">
    <ParamLabel label="Caption source" info="Use your own caption, the training track nearest the detected BPM, or a named training track." />
    <StyledSelect accent="pink" value={p.captionMode} onChange={p.onCaptionMode}
      options={[
        { value: 'custom', label: 'Custom caption' },
        { value: 'auto', label: 'Automatic from dataset' },
        ...p.captionTracks.map(track => ({ value: `track:${track.name}`, label: `Track: ${track.name}` })),
      ]} />
  </div>;
  const captionEl = <textarea value={p.caption} onChange={e => p.onCaptionChange(e.target.value)}
    placeholder="Style caption: genre, instruments, mood and vocal style" aria-label="YuE2 style caption"
    className="block w-full h-24 resize-none rounded-xl bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 px-3 py-2 text-xs text-zinc-900 dark:text-white" />;
  const instrumentalEl = <div className="flex items-center justify-between">
    <ParamLabel label="Instrumental" info="Skip lyrics and render the approved melody without vocals." />
    <Toggle accent="cyan" checked={p.instrumental} onChange={p.onInstrumentalChange} aria-label="Instrumental" />
  </div>;
  const keepChordsEl = <div className="flex items-center justify-between">
    <ParamLabel label="Keep chords" info="Retain the original harmony; off lets the style decide." />
    <Toggle accent="cyan" checked={p.keepChords} onChange={p.onKeepChords} aria-label="Keep chords" />
  </div>;
  const voicesEl = <div className="space-y-2">
    <ParamLabel label="Voices" info="Vocal melody alone gives the new style room to choose its instruments. Both keeps the transcribed instrumental line too." />
    <StyledSelect accent="pink" value={p.voices} onChange={v => p.onVoices(v as 'vocal' | 'both')}
      options={[{ value: 'vocal', label: 'Vocal melody only' }, { value: 'both', label: 'Vocal + instrumental line' }]} />
  </div>;
  const tempoEl = <div className="space-y-2">
    <ParamLabel label="Tempo" info="Free removes the score tempo; Source retains it; Set writes a target BPM using the score's beat unit." />
    <StyledSelect accent="pink" value={p.tempoMode} onChange={v => {
      if (v === 'set' && p.tempoMode !== 'set') p.onBpm(Math.max(20, Math.min(300, sourceBpm)));
      p.onTempoMode(v as 'free' | 'source' | 'set');
    }} options={[{ value: 'free', label: 'Free (style decides)' }, { value: 'source', label: 'Source tempo' }, { value: 'set', label: 'Set BPM' }]} />
  </div>;
  const bpmEl = p.tempoMode === 'set' && <EditableSlider label="Target BPM" value={p.bpm} min={20} max={300} step={1}
    onChange={p.onBpm} helpText={`Source: ${sourceBpm} BPM`} />;
  const pitchEl = <EditableSlider label="Pitch Shift" value={p.keyShift} min={-12} max={12} step={1}
    onChange={p.onKeyShift} defaultValue={0} disabled={!keyLabel}
    formatDisplay={v => v === 0 ? 'Source key' : `${v > 0 ? '+' : ''}${v} st → ${transposeKey(keyLabel, v)}`}
    helpText={keyLabel ? `Source: ${keyLabel}. Shift within twelve semitones.` : 'No supported score key found.'} />;
  const arStrengthEl = <CoverArStrength value={p.arStrength} onChange={p.onArStrength} />;
  const advancedEl = <details className="text-xs text-zinc-600 dark:text-zinc-400">
    <summary className="cursor-pointer">Advanced</summary>
    <EditableSlider label="Condition strength" value={p.cfgScale} min={0.1} max={2} step={0.05}
      onChange={p.onCfgScale} defaultValue={1}
      helpText="Tightens style, lyrics and score together." />
    <ParamLabel label="Lyric Timing (C6)" info="Ties each lyric block to its score section's timing. Off by default; condition strength 1 with it on ruins lyric adherence. Soft nudges the composer toward that timing; Hard hides a section's lyrics until its first Vocal note. Needs CFG 1." />
    <StyledSelect accent="pink" value={p.lyricSchedule} onChange={v => p.onLyricSchedule(v as 'off' | 'bias' | 'mask')}
      options={[
        { value: 'off', label: 'Off' },
        { value: 'bias', label: 'Soft (bias toward the score\'s timing)' },
        { value: 'mask', label: 'Hard (hide lyrics until timed)' },
      ]} />
  </details>;
  // Render progress and cancel live in the queue; this only adds to it.
  const generateEl = <div className="space-y-2">
    <div className="flex gap-2">
      <button onClick={p.onGenerate} disabled={!p.canGenerate}
        className="flex-1 rounded-xl bg-gradient-to-r from-pink-600 to-purple-600 py-3 text-sm font-bold text-white disabled:opacity-40 disabled:cursor-not-allowed">
        {p.isGenerating ? 'Add to queue' : 'Generate cover'}
      </button>
      <button type="button" onClick={p.onOpenBuild} title="Compose the cover a few sections at a time and keep the best takes"
        className="rounded-xl border border-pink-500/60 px-4 py-3 text-sm font-semibold text-pink-700 dark:text-pink-300 hover:bg-pink-500/10">
        Build by section →</button>
    </div>
    {p.buildStatus && <p className="text-xs text-zinc-500">{p.buildStatus} ·{' '}
      <button type="button" onClick={p.onOpenBuild} className="text-cyan-700 dark:text-cyan-300">Open</button>
      {p.onStartOverBuild && <> · <button type="button" onClick={p.onStartOverBuild} className="text-red-500">Start over</button></>}</p>}
    {!p.canGenerate && <p className="text-xs text-zinc-500">Choose audio, approve an ABC score, and enter lyrics or select Instrumental.</p>}
    {p.footer}
  </div>;

  return (
  <div ref={rootRef} className="w-full min-w-0 flex-1 overflow-y-auto scrollbar-hide p-4">
    {wide
      // Paired rows, each pair top-aligned: full-width render mode and artist
      // grid, source beside caption source over a full-width caption, the two
      // toggles, voices beside tempo, pitch beside the target BPM, then
      // full-width Advanced and Generate. A missing half leaves its slot to
      // the next control, so a lone caption source sits on the left.
      ? <div className="grid grid-cols-2 items-start gap-x-6 gap-y-4">
        <div className="col-span-2">{renderModeEl}</div>
        {artistGridEl && <div className="col-span-2">{artistGridEl}</div>}
        {sourceEl}{captionSourceEl}
        <div className="col-span-2">{captionEl}</div>
        {instrumentalEl}{keepChordsEl}
        {voicesEl}{tempoEl}
        {pitchEl}{bpmEl}
        <div className="col-span-2">{arStrengthEl}</div>
        <div className="col-span-2">{advancedEl}</div>
        <div className="col-span-2">{generateEl}</div>
      </div>
      : <div className="space-y-3">
        {renderModeEl}{artistGridEl}{sourceEl}{captionSourceEl}{captionEl}{instrumentalEl}
        {voicesEl}{keepChordsEl}{tempoEl}{bpmEl}{pitchEl}{arStrengthEl}{advancedEl}{generateEl}
      </div>}
  </div>
  );
};
