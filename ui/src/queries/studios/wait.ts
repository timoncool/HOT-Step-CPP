import { QueryObserver, type QueryKey } from '@tanstack/react-query';
import { queryClient } from '../client';
import { pollWhile, isRetryableError } from '../contract';

/** Workflow callers await a snapshot; Query owns reads and polling. Detaching
 *  this observer stops observation, never the underlying server job. */
export function waitForSnapshot<T>(key: QueryKey, read: (signal: AbortSignal) => Promise<T>,
  active: (data: T) => boolean, onData?: (data: T) => void,
  opts: { pollMs?: number; timeoutMs?: number; signal?: AbortSignal } = {}): Promise<T> {
  return new Promise((resolve, reject) => {
    if (opts.signal?.aborted) { reject(new DOMException('Observation cancelled', 'AbortError')); return; }
    const observer = new QueryObserver<T>(queryClient, {
      queryKey: key, queryFn: ({ signal }) => read(signal), staleTime: 0,
      refetchInterval: pollWhile<T>(data => !data || active(data), opts.pollMs),
      refetchIntervalInBackground: true,
    });
    let last: T | undefined;
    let unsubscribe: (() => void) | undefined;
    const finish = () => {
      clearTimeout(timeout);
      opts.signal?.removeEventListener('abort', abort);
      queueMicrotask(() => unsubscribe?.());
    };
    const abort = () => { finish(); reject(new DOMException('Observation cancelled', 'AbortError')); };
    const timeout = opts.timeoutMs ? setTimeout(() => { finish(); reject(new Error('Job timed out')); }, opts.timeoutMs) : undefined;
    opts.signal?.addEventListener('abort', abort, { once: true });
    unsubscribe = observer.subscribe(result => {
      if (result.isError && result.fetchStatus === 'idle') {
        if (!isRetryableError(result.error)) { finish(); reject(result.error); }
        return;
      }
      if (result.data === undefined || result.isFetching || result.data === last) return;
      last = result.data;
      onData?.(result.data);
      if (!active(result.data)) { finish(); resolve(result.data); }
    });
  });
}
