// yue2/durationCap.ts: YuE2 has no length slot on the wire (the plan/semantic
// stages end on their own terminator), so a requested duration is honoured as
// a ceiling by trimming the rendered WAV server-side with a short fade-out.

import { cropWavFile, wavDurationSec } from '../../audioCrop.js';

/** Fade-out applied at the trim point so the cut reads as an ending. */
export const YUE2_TRIM_FADE_SEC = 1.5;
/** Renders within this much of the cap are left alone (no pointless 0.1 s trims). */
const TRIM_SLACK_SEC = 0.25;

/** Requested ceiling in seconds, or null for auto length (duration <= 0 or absent).
 *  cropWavFile refuses results under 1 s, so the ceiling is floored at 1. */
export function yue2DurationCapSec(params: { duration?: unknown } | null | undefined): number | null {
  const d = Number(params?.duration);
  if (!Number.isFinite(d) || d <= 0) return null;
  return Math.max(1, d);
}

/** Trim `filepath` in place to the requested ceiling. Returns the resulting
 *  duration and whether a trim happened. */
export function applyYue2DurationCap(filepath: string, params: { duration?: unknown } | null | undefined): { durationSec: number; trimmed: boolean } {
  const cap = yue2DurationCapSec(params);
  const measured = wavDurationSec(filepath);
  if (cap === null || measured <= cap + TRIM_SLACK_SEC) return { durationSec: measured, trimmed: false };
  const fade = Math.min(YUE2_TRIM_FADE_SEC, cap / 4);
  const { newDurationSec } = cropWavFile(filepath, 0, cap, fade);
  return { durationSec: newDurationSec, trimmed: true };
}
