// Safety net for the YuE2 composer (semantic) stage: unlimited top-k (0) with
// temperature above 1.0 samples from the whole flattened tail and the song
// falls apart. Clamp temperature to 1.0 and leave a warning in the job notes.
// Pure (no DB/engine imports) so it can be checked in isolation.
//
// top_k is truncated to int first because the engine does the same
// (engine/src/yue2/yue2-request.h), so 0.5 means unlimited there too.
// When temperature is unset the engine uses the checkpoint's own semantic
// temperature, so that value (if known) is what gets checked.
export function clampYue2SemanticSampling(
  overrides: Record<string, number>,
  notes: string[],
  checkpointTemperature?: number,
): void {
  if (overrides.semantic_top_k === undefined || Math.trunc(Number(overrides.semantic_top_k)) !== 0) return;
  const explicit = overrides.semantic_temperature !== undefined;
  const fallback = typeof checkpointTemperature === 'number' && Number.isFinite(checkpointTemperature) ? checkpointTemperature : 1.0;
  const temp = explicit ? overrides.semantic_temperature : fallback;
  if (!(temp > 1.0)) return;
  notes.push(`WARNING: semantic_top_k=0 (unlimited) with semantic_temperature=${temp}${explicit ? '' : ' (checkpoint default)'} breaks the composer stage; temperature clamped to 1.0`);
  overrides.semantic_temperature = 1.0;
}
