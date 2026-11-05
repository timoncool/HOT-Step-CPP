// training/beatThisClient.ts — dataset-label tempo from Beat This! (ace-midi --track)
//
// The labeling default since HOTSTEP-730: on 24 reference tracks it landed
// within 3% on 23, where Essentia made 5 half/double-time errors. Beats only,
// no transcription model; the same grid MIDI Studio writes its tempo from.
// Returns null on any failure so the caller falls back to Essentia.

import fs from 'fs';
import os from 'os';
import path from 'path';
import crypto from 'crypto';
import { execFile } from 'child_process';
import { getFFmpegPath } from '../../config.js';
import { aceMidiExe, BEAT_MODEL, isModelDownloaded, modelDir, readGrid } from '../muscriptor.js';
import { runOnGpuLane } from '../generation/gpuLane.js';

/** ace-midi decodes these itself; anything else goes through ffmpeg first. */
const NATIVE_EXTS = ['.wav', '.mp3'];

export function beatThisAvailable(): boolean {
  return !!aceMidiExe() && isModelDownloaded(BEAT_MODEL);
}

function run(bin: string, args: string[], signal?: AbortSignal): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(bin, args, { timeout: 300_000, maxBuffer: 10 * 1024 * 1024, signal, windowsHide: true }, (error, _out, stderr) => {
      if (error) reject(error); else resolve(String(stderr));
    });
  });
}

/** The fitted grid tempo in BPM, or null (no binary/model, decode failure, or
 *  too few beats, where ace-midi writes a 120 BPM placeholder). */
export async function analyzeTempoWithBeatThis(audioPath: string, signal?: AbortSignal): Promise<number | null> {
  const exe = aceMidiExe();
  if (!exe || !isModelDownloaded(BEAT_MODEL)) return null;
  const stamp = `${Date.now()}_${crypto.randomBytes(4).toString('hex')}`;
  const beats = path.join(os.tmpdir(), `hs_training_beats_${stamp}.json`);
  let wav: string | null = null;
  try {
    let input = audioPath;
    if (!NATIVE_EXTS.includes(path.extname(audioPath).toLowerCase())) {
      const ffmpeg = getFFmpegPath();
      if (!ffmpeg) return null;
      wav = path.join(os.tmpdir(), `hs_training_beats_${stamp}.wav`);
      await run(ffmpeg, ['-y', '-loglevel', 'error', '-i', audioPath, '-ar', '44100', '-ac', '2', wav], signal);
      input = wav;
    }
    const stderr = await runOnGpuLane(
      () => run(exe, ['--track', input, '--beat-model', modelDir(BEAT_MODEL), '--beats-out', beats], signal),
      { label: 'label:beat-this', family: 'midi' });
    if (/writing 120 BPM/.test(stderr)) return null;
    const grid = readGrid(fs.readFileSync(beats, 'utf8'));
    return grid && Number.isFinite(grid.bpm) && grid.bpm > 0 ? grid.bpm : null;
  } catch (err: any) {
    if (err?.name !== 'AbortError') console.warn(`[Training] beat_this failed for ${path.basename(audioPath)}: ${err?.message || err}`);
    return null;
  } finally {
    for (const f of [beats, wav]) if (f) { try { fs.unlinkSync(f); } catch { /* never written */ } }
  }
}
