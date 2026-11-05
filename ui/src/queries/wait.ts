// wait.ts — the one imperative snapshot waiter.
//
// Workflow services and stores cannot use hooks, but they still want Query to
// own the read, the polling cadence and the cache entry. This observer bridges
// the two: it attaches to the canonical key, polls with `pollWhile`, resolves
// when the snapshot leaves its active state, and detaches on completion.
//
// Two divergent copies of this helper used to live in `studios/` and
// `cover-lyrics/`; only the cover copy carried the auth-session guard, so a
// wait started under one token could resolve against another session's cache.
// The guarded implementation is the canonical one.

import { QueryObserver, type QueryKey } from '@tanstack/react-query';
import { queryClient } from './client';
import { getSessionEpoch, scopeOf } from './keys';
import { isRetryableError, pollWhile } from './contract';

export interface WaitOptions {
  pollMs?: number;
  timeoutMs?: number;
  signal?: AbortSignal;
  /** Deliver unchanged snapshots to `onData` again. Callers whose `onData`
   *  checks cancellation on every tick need this; progress callers do not. */
  deliverRepeats?: boolean;
}

/** Wait on a query-options factory instead of rebuilding its key and fn.
 *  Callers that already have the canonical factory should prefer this form. */
export function waitForQuery<T>(
  options: { queryKey: QueryKey; queryFn?: unknown },
  active: (data: T) => boolean,
  onData?: (data: T) => void,
  opts: WaitOptions = {},
): Promise<T> {
  if (typeof options.queryFn !== 'function') throw new Error('waitForQuery needs a queryFn');
  const queryFn = options.queryFn as (ctx: {
    client: typeof queryClient; queryKey: QueryKey; signal: AbortSignal;
  }) => Promise<T> | T;
  const read = (signal: AbortSignal) => Promise.resolve(queryFn({
    client: queryClient, queryKey: options.queryKey, signal,
  }));
  return waitForSnapshot(options.queryKey, read, active, onData, opts);
}

export function waitForSnapshot<T>(
  key: QueryKey,
  read: (signal: AbortSignal) => Promise<T>,
  active: (data: T) => boolean,
  onData?: (data: T) => void,
  opts: WaitOptions = {},
): Promise<T> {
  return new Promise((resolve, reject) => {
    if (opts.signal?.aborted) { reject(new DOMException('Observation cancelled', 'AbortError')); return; }

    const scope = scopeOf(key);
    const authScoped = scope?.session !== undefined;
    const sessionLost = () => authScoped && scope!.session !== getSessionEpoch();

    const observer = new QueryObserver<T>(queryClient, {
      queryKey: key,
      queryFn: ({ signal }) => {
        if (sessionLost()) throw new DOMException('Auth session changed', 'AbortError');
        return read(signal);
      },
      staleTime: 0,
      refetchInterval: pollWhile<T>(data => !data || active(data), opts.pollMs),
      refetchIntervalInBackground: true,
    });

    let last: T | undefined;
    const subscription: { stop?: () => void } = {};
    let settled = false;

    const finish = () => {
      settled = true;
      clearTimeout(timeout);
      opts.signal?.removeEventListener('abort', abort);
      queueMicrotask(() => subscription.stop?.());
    };
    const abort = () => { finish(); reject(new DOMException('Observation cancelled', 'AbortError')); };
    const timeout = opts.timeoutMs
      ? setTimeout(() => { finish(); reject(new Error('Job timed out')); }, opts.timeoutMs)
      : undefined;
    opts.signal?.addEventListener('abort', abort, { once: true });

    subscription.stop = observer.subscribe(result => {
      if (settled) return;
      if (result.isError && result.fetchStatus === 'idle') {
        if (!isRetryableError(result.error)) { finish(); reject(result.error); }
        return;
      }
      if (result.data === undefined || result.isFetching) return;
      try {
        if (sessionLost()) throw new DOMException('Auth session changed', 'AbortError');
        const repeated = result.data === last;
        if (repeated && !opts.deliverRepeats) return;
        last = result.data;
        onData?.(result.data);
        if (!active(result.data)) { finish(); resolve(result.data); }
      } catch (error) { finish(); reject(error); }
    });
  });
}
