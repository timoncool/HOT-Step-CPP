// training/heartmulaPreviewSource.ts — where a HeartMuLa run's checkpoint-preview
// prompt comes from, by the same rules as YuE2's (yue2JointPreview.ts,
// withGeneratedPreviewLyrics):
//   tags    the override, else the tags of the dataset's first sung track
//           (what training saw; the first track when none is sung)
//   lyrics  the override, else the newest Lyric Studio generation for the
//           dataset's artist, else the same track's lyrics; lyricsSource
//           'dataset' skips the generation
// Resolved once per run and stored in run.json, so every take (base, steps,
// final, retries, resumes) renders the same words.

import { HEARTMULA_PREVIEW_DEFAULTS, type HeartmulaPreviewSettings, type HeartmulaPreviewSource } from '../../contracts/heartmulaTraining.js';
import type { TrainingDatasetRow, TrainingSample } from './types.js';
import { buildSamples } from './datasetScan.js';
import { heartmulaSampleInput, trainableSamples } from './heartmulaPrepare.js';
import { saveHeartmulaRunRecord, type HeartmulaRunRecord } from './heartmulaRuns.js';
import { resolveHeartmulaPreviewSettings } from './heartmulaTrain.js';
import { datasetGenerationLyrics } from './yue2JointTrainRunner.js';

/** The first sung track, as YuE2 picks it: an instrumental intro would make a lyric-less preview. */
const sungTrack = (samples: TrainingSample[]): TrainingSample | undefined =>
  samples.find(s => !s.isInstrumental && s.lyrics.trim()) ?? samples[0];

/** Fill blank tags and lyrics of `settings` from the dataset. Throws when neither an override nor a track gives tags. */
export function resolveHeartmulaPreviewPrompt(
  ds: Pick<TrainingDatasetRow, 'customTag' | 'lyricsSetId' | 'sourceDir'>, samples: TrainingSample[], settings: HeartmulaPreviewSettings,
): { settings: HeartmulaPreviewSettings; source: HeartmulaPreviewSource } {
  const track = sungTrack(trainableSamples(samples));
  const trackInput = track ? heartmulaSampleInput(ds, track) : undefined;
  const tags = settings.tags.trim() || trackInput?.tags || '';
  if (!tags) throw new Error('The preview needs tags: the dataset has no usable track to take them from, so give tags in the preview settings');
  const generated = settings.lyrics.trim() || settings.lyricsSource === 'dataset' ? undefined : datasetGenerationLyrics(ds);
  const lyrics = settings.lyrics.trim() ? settings.lyrics : generated?.lyrics ?? trackInput?.lyrics ?? '';
  return {
    settings: { ...settings, tags, lyrics },
    source: {
      tags: settings.tags.trim() ? 'override' : 'dataset',
      lyrics: settings.lyrics.trim() ? 'override' : generated ? 'lyric-studio' : 'dataset',
      sampleId: track?.sampleId ?? null, sampleName: track?.filename ?? null,
      generationId: generated?.id ?? null, generationTitle: generated?.title || null,
    },
  };
}

/** The run's preview settings with the prompt resolved, resolving and saving it first on a run that has none yet. */
export async function ensureHeartmulaRunPreview(
  ds: TrainingDatasetRow, run: { dir: string; record: HeartmulaRunRecord },
): Promise<{ settings: HeartmulaPreviewSettings; source: HeartmulaPreviewSource }> {
  const stored = resolveHeartmulaPreviewSettings(run.record.preview, { ...HEARTMULA_PREVIEW_DEFAULTS, everySteps: 0 });
  if (run.record.previewSource) return { settings: stored, source: run.record.previewSource };
  const resolved = resolveHeartmulaPreviewPrompt(ds, await buildSamples(ds), stored);
  saveHeartmulaRunRecord(run.dir, { ...run.record, preview: resolved.settings, previewSource: resolved.source });
  run.record.preview = resolved.settings; run.record.previewSource = resolved.source;
  return resolved;
}
