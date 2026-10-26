// MM3 keeps its preview records and routes; only the presentation is shared.
import React from 'react';
import { useTranslation } from 'react-i18next';
import { useTrainingStore, type TrainingPreviewRow } from '../../stores/trainingStore';
import { TrainingPreviewRung, type PreviewTake } from './TrainingPreviewPanel';

const KINDS = ['artist', 'control'] as const;

export const Mm3PreviewList: React.FC<{ previews: TrainingPreviewRow[] }> = ({ previews }) => {
  const { t } = useTranslation();
  if (!previews.length) return null;
  return <section className="mt-4 space-y-3">
    <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
      {t('trainingStudio.mm3.previews', 'Previews')}
    </h3>
    <p className="text-[11px] text-zinc-500">
      {t('trainingStudio.mm3.previewsHint', 'Same caption, same seed, one variable: the checkpoint.')}
    </p>
    {KINDS.map(kind => {
      const rows = previews.filter(p => p.kind === kind);
      if (!rows.length) return null;
      const runs = [...new Set(rows.map(p => p.run))];
      return <div key={kind} className="space-y-2">
        <h4 className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">
          {kind === 'artist'
            ? t('trainingStudio.mm3.previewArtistCaption', 'Artist caption')
            : t('trainingStudio.mm3.previewControlCaption', 'Control caption')}
        </h4>
        {runs.flatMap(run => {
          // Keep different runs separate even when checkpoint steps match.
          const runRows = rows.filter(p => p.run === run).sort((a, b) => Number(b.base) - Number(a.base) || a.step - b.step);
          const groups = new Map<string, TrainingPreviewRow[]>();
          for (const row of runRows) {
            const key = row.base ? 'base' : String(row.step);
            groups.set(key, [...(groups.get(key) ?? []), row]);
          }
          return [...groups].map(([key, records]) => {
            const first = records[0];
            const title = first.base
              ? t('trainingStudio.mm3.previewBase', 'Base')
              : t('trainingStudio.mm3.previewStep', 'Step {{step}}', { step: first.step });
            const takes: PreviewTake[] = records.map(row => ({
              id: `${row.run}-${row.id}`,
              src: `/api/training/mm3/preview?run=${encodeURIComponent(row.run)}&file=${encodeURIComponent(row.file)}`,
              label: t('trainingStudio.mm3.previewTake', '{{seconds}} s · seed {{seed}} · rendered in {{renderSeconds}} s', {
                seconds: row.seconds, seed: row.seed, renderSeconds: (row.ms / 1000).toFixed(0),
              }),
              sublabel: row.caption,
            }));
            return <TrainingPreviewRung key={`${run}-${key}`} title={title} takes={takes}
              className={first.base ? 'border-sky-500/30 bg-sky-500/5' : undefined}
              badges={first.loss !== undefined && <span className="text-[11px] text-zinc-500 tabular-nums">
                {t('trainingStudio.mm3.previewLoss', 'Loss {{loss}}', { loss: first.loss.toFixed(3) })}
              </span>} />;
          });
        })}
      </div>;
    })}
  </section>;
};

export const Mm3PreviewStrip: React.FC = () => {
  const previews = useTrainingStore(s => s.trainPreviews);
  return <Mm3PreviewList previews={previews} />;
};

export default Mm3PreviewStrip;
