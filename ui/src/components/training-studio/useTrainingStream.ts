// useTrainingStream.ts — EventSource hook for training job progress
//
// The server replays the whole event buffer on connect, so every handler in
// the store must be idempotent (sample rows are replaced by id, never
// appended). EventSource reconnects on its own; we only close on unmount.
//
// The stream is opened for the worker selected when it starts and stays on
// it. A frame that arrives after "Train on" moved elsewhere, before React
// has torn the subscription down, is dropped: it describes the old worker.

import { useEffect } from 'react';
import { useTrainingStore } from '../../stores/trainingStore';
import { jobStreamUrl, type TrainingStreamEvent } from '../../services/trainingApi';
import { queryClient } from '../../queries/client';
import { trainingQueries } from '../../queries/training-data/queries';
import { subscribeEventSource } from '../../services/sharedEventSource';

export function useTrainingStream(jobId: string | null) {
  const applyStreamEvent = useTrainingStore(s => s.applyStreamEvent);
  const worker = useTrainingStore(s => s.trainingWorker);

  useEffect(() => {
    if (!jobId) return;
    const apply = (ev: TrainingStreamEvent) => {
      if (useTrainingStore.getState().trainingWorker === worker) applyStreamEvent(ev);
    };
    return subscribeEventSource(jobStreamUrl(jobId, { worker }), {
      message: (data) => {
        try { apply(JSON.parse(data) as TrainingStreamEvent); } catch { /* ignore malformed frame */ }
      },
      error: () => {
        // EventSource auto-reconnects and the buffer replay makes that safe. But
        // if the server restarted, the in-memory job is gone (stream 404s forever)
        // and no terminal event will ever arrive — the UI would show "running"
        // and keep Start disabled indefinitely. Poll once per error: a live job
        // is a no-op; a terminal or vanished job synthesizes its final status.
        // Through the cache, so every view streaming this job shares the probe.
        void queryClient.query(trainingQueries.job(worker, jobId)).then(
          (job) => {
            if (job.status !== 'queued' && job.status !== 'running') {
              apply({ type: 'status', status: job.status, ...(job.error ? { error: job.error } : {}) });
            }
          },
          () => apply({ type: 'status', status: 'failed', error: 'Job lost — the server restarted' }),
        );
      },
    });
  }, [jobId, worker, applyStreamEvent]);
}
