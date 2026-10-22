// PipelineStageProgress.tsx — live progress for the stage a bulk run is in
//
// The Monitor page listed items and per-stage chips but showed nothing about
// the stage actually executing, so a multi-hour train-dit inside a bulk run had
// no visible progress anywhere (2026-07-31). Every piece needed already
// existed — PipelineStageResult carries the stage's jobId, and the Train panel
// renders the same job through JobProgress/TrainingRunStats/TrainingChart.
// This adopts the running stage's job so those components light up here too.
//
// WHY IT WRITES activeJob: the store fills the chart series only while
// `activeJob.kind` is train-lm/train-dit (trainingStore's `metric` handler), so
// subscribing to the stream alone would leave the chart empty. It restores the
// previous activeJob on unmount and never clobbers a job the Train panel is
// already showing.

import React, { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { blankTrainSeries, useTrainingStore } from '../../stores/trainingStore';
import type { PipelineItem, PipelineStage } from '../../services/trainingApi';
import { queryClient } from '../../queries/client';
import { trainingQueries } from '../../queries/training-data/queries';
import { JobProgress } from './JobProgress';
import { TrainingChart } from './TrainingChart';
import { TrainingRunStats } from './TrainingRunStats';
import { useTrainingStream } from './useTrainingStream';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';

interface Props {
  /** The item currently executing, or null when nothing is running. */
  item: PipelineItem | null;
}

function runningStage(item: PipelineItem | null): { stage: PipelineStage; jobId: string } | null {
  if (!item || item.status !== 'running') return null;
  for (const s of item.stages) {
    if (s.status === 'running' || s.status === 'creating') {
      return { stage: s.stage, jobId: s.jobId };
    }
  }
  return null;
}

export const PipelineStageProgress: React.FC<Props> = ({ item }) => {
  const { t } = useTranslation();
  const active = runningStage(item);
  const jobId = active?.jobId || '';

  const activeJob = useTrainingStore(s => s.activeJob);
  const trainLmEpochs = useTrainingStore(s => s.trainLmEpochs);
  const trainDitEpochs = useTrainingStore(s => s.trainDitEpochs);
  const trainStepSeries = useTrainingStore(s => s.trainStepSeries);
  const trainMilestones = useTrainingStore(s => s.trainMilestones);
  const trainMaxEpochs = useTrainingStore(s => s.trainMaxEpochs);

  const [adopted, setAdopted] = useState('');
  // Only ever restore a job WE replaced, so navigating away cannot wipe a job
  // the Train panel legitimately owns.
  const previous = useRef<typeof activeJob>(null);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    previous.current = useTrainingStore.getState().activeJob;
    // The pipeline runs on the worker selected now; adopt only while it still is.
    const worker = useTrainingStore.getState().trainingWorker;
    void queryClient.query(trainingQueries.job(worker, jobId)).then(
      (job) => {
        if (cancelled || useTrainingStore.getState().trainingWorker !== worker) return;
        // BLANK THE CHART BEFORE ADOPTING. trainStepSeries is shared by both
        // training kinds and the epoch arrays are per-kind but not per-run, so
        // carrying them across a stage change drew the new run into the old
        // one's axis — an LM at epoch 15 rendered in the left 3% of a scale
        // still set to the DiT's 500 epochs. blankTrainSeries is also the only
        // thing allowed to reset the step-dedup high-water mark, which would
        // otherwise swallow the new run's early steps.
        //
        // Safe against the SSE replay: the stream is subscribed only after
        // `adopted` is set, and it replays the whole buffer on connect.
        useTrainingStore.setState({
          ...blankTrainSeries(),
          trainLmEpochs: [], trainLmLast: null,
          trainDitEpochs: [], trainDitLast: null,
          jobLog: [],
          activeJob: job,
        });
        setAdopted(jobId);
      },
      () => { /* job may have finished between poll and fetch — chips still show it */ },
    );
    return () => {
      cancelled = true;
      const cur = useTrainingStore.getState().activeJob;
      if (cur && cur.id === jobId) useTrainingStore.setState({ activeJob: previous.current });
      previous.current = null;
    };
  }, [jobId]);

  useTrainingStream(adopted && adopted === jobId ? jobId : null);

  if (!item || !active) return null;

  const kind = activeJob?.id === jobId ? activeJob.kind : null;
  const isLm = kind === 'train-lm';
  const isDit = kind === 'train-dit';
  const epochs = isLm ? trainLmEpochs : isDit ? trainDitEpochs : [];
  const showChart = (isLm || isDit) && (epochs.length >= 2 || trainStepSeries.length >= 2);

  return (
    <div className={`${CARD} flex flex-col gap-2`}>
      <div className="flex items-baseline gap-2 flex-wrap">
        <span className="text-xs font-bold text-zinc-900 dark:text-white">
          {t('trainingStudio.monitor.nowRunning')}
        </span>
        <span className="text-xs text-zinc-600 dark:text-zinc-400 truncate max-w-[420px]">
          {item.name || item.sourceDir}
        </span>
      </div>

      {/* Renders from the first frame — elapsed time is useful before any curve
          exists, and on a long-epoch run this is the only live feedback. */}
      {(isLm || isDit) && <TrainingRunStats />}
      <JobProgress />
      {showChart && (
        <TrainingChart
          epochs={epochs}
          steps={trainStepSeries}
          milestones={trainMilestones}
          target={0}
          maxEpochs={trainMaxEpochs || 0}
        />
      )}
    </div>
  );
};

export default PipelineStageProgress;
