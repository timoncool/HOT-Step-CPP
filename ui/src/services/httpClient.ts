// httpClient.ts — one configurable client for every /api/* call.
//
// JSON traffic goes through the shared axios instance (axiosClient.ts): its
// request interceptor injects the bearer from the auth session accessor, and
// its response interceptor is the global catch that normalizes failures to
// ApiError (status + optional currentRevision + reason) — `instanceof
// ApiError` is enough, callers don't need a second error class.
//
// Multipart upload stays on XHR — fetch has no upload-progress event and only
// XHR reports it — but its failures go through the same handleFailure() as the
// axios path, so there is still exactly one place that decides what is logged
// and toasted.
//
// SSE is not here: `streamEvents()` stays on fetch (axios does not stream
// response bodies), and reconnecting streams are owned by sharedEventSource.ts.

import { getAuthToken, handleFailure, http, SLOW_ROUTE_TIMEOUT_MS } from './axiosClient';
import { ApiError, HttpStatus } from './apiError';
export { ApiError, HttpStatus };

/** Trailing options every entrypoint accepts, so a query function can pass
 *  TanStack Query's AbortSignal without changing positional callers. The
 *  bearer is no longer threaded per call — the transport reads it from the
 *  auth session accessor (axiosClient.ts). */
export interface ClientOptions {
  signal?: AbortSignal;
  /** Opt this request out of the global error toast (X-Skip-Global-Error). */
  silent?: boolean;
  /** Override the instance's default timeout for known-slow routes (ms). */
  timeout?: number;
  /** Talk to a different origin than this client's base (remote workers). */
  baseURL?: string;
  /** Return the raw body as a Blob instead of parsing JSON (binary routes). */
  responseType?: 'json' | 'blob';
}

export interface ReadOptions {
  signal?: AbortSignal;
}

export interface ApiClientConfig {
  baseUrl?: string;
  /** Root for `mediaUrl()` — `/audio` and `/references` src attributes.
   *  Defaults to '' (same origin as the page), which is what the Vite dev
   *  proxy and the prod server both want today. A client built for a
   *  separate frontend origin (a future remote/phone client) sets this to
   *  that server's origin instead. */
  mediaRoot?: string;
}

function axiosOpts(opts: ClientOptions): { signal?: AbortSignal; timeout?: number; headers?: Record<string, string>; baseURL?: string; responseType?: 'json' | 'blob' } {
  const out: { signal?: AbortSignal; timeout?: number; headers?: Record<string, string>; baseURL?: string; responseType?: 'json' | 'blob' } = {};
  if (opts.signal) out.signal = opts.signal;
  if (opts.timeout !== undefined) out.timeout = opts.timeout;
  if (opts.baseURL !== undefined) out.baseURL = opts.baseURL;
  if (opts.responseType !== undefined) out.responseType = opts.responseType;
  if (opts.silent) out.headers = { 'X-Skip-Global-Error': '1' };
  return out;
}

/** A non-OK response on a fetch-based SSE route (streamEvents, consumeSSE):
 *  the same ApiError and global catch as any other request — handleFailure
 *  throws, so awaiting this never returns. */
export async function sseFailure(res: Response, method: string, url: string, skip?: boolean): Promise<never> {
  const body = await res.json().catch(() => null);
  handleFailure({ status: res.status, body, method, url, skip });
}

export class ApiClient {
  private readonly base: string;
  private readonly mediaRoot: string;
  constructor(opts: ApiClientConfig = {}) {
    this.base = opts.baseUrl ?? '/api';
    this.mediaRoot = opts.mediaRoot ?? '';
  }

  /** The JSON base this client talks to; query keys carry it as the API origin. */
  get baseUrl(): string {
    return this.base;
  }

  private url(path: string): string {
    return path.startsWith('http') ? path : `${this.base}${path}`;
  }

  /** The one JSON request path every verb below delegates to, and what
   *  domain clients (workflowApi's revisionedRequest) use for a method the
   *  verbs don't name. Failures come back as ApiError from the interceptor. */
  async request<T>(method: string, path: string, body?: unknown, opts: ClientOptions = {}): Promise<T> {
    const { data } = await http.request<T>({
      method: method.toUpperCase(), url: path, data: body, baseURL: this.base, ...axiosOpts(opts),
    });
    return data;
  }

  async get<T>(path: string, opts: ClientOptions = {}): Promise<T> {
    return this.request<T>('GET', path, undefined, opts);
  }

  async post<T>(path: string, body?: unknown, opts: ClientOptions = {}): Promise<T> {
    return this.request<T>('POST', path, body, opts);
  }

  async patch<T>(path: string, body?: unknown, opts: ClientOptions = {}): Promise<T> {
    return this.request<T>('PATCH', path, body, opts);
  }

  async put<T>(path: string, body?: unknown, opts: ClientOptions = {}): Promise<T> {
    return this.request<T>('PUT', path, body, opts);
  }

  async delete<T>(path: string, opts: ClientOptions = {}): Promise<T> {
    return this.request<T>('DELETE', path, undefined, opts);
  }

  /** Multipart upload via XHR, so `onProgress` can report upload fraction —
   *  fetch has no upload-progress event. `fields` are appended before the
   *  files, so a server that reads the form in order sees them first. A
   *  plain Blob (recombined stems) names itself with `name`; a File uses its
   *  own name unless overridden. */
  upload<T>(
    path: string,
    files: { field: string; file: File | Blob; name?: string }[],
    opts: ClientOptions & { fields?: Record<string, string>; onProgress?: (fraction: number) => void } = {},
  ): Promise<T> {
    const form = new FormData();
    for (const [key, value] of Object.entries(opts.fields ?? {})) form.append(key, value);
    for (const { field, file, name } of files) form.append(field, file, name ?? (file instanceof File ? file.name : undefined));

    return new Promise<T>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', this.url(path));
      const token = getAuthToken();
      if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      // A stalled upload would otherwise hang forever (XHR has no default
      // timeout). The limit is on silence, not the whole request, so a big
      // import over a slow link still finishes while bytes keep moving. Once
      // the body is sent, the server's own work is untimed.
      const idleMs = opts.timeout ?? SLOW_ROUTE_TIMEOUT_MS;
      let stalled = false;
      let idle: ReturnType<typeof setTimeout> | undefined;
      const watch = () => {
        clearTimeout(idle);
        idle = setTimeout(() => { stalled = true; xhr.abort(); }, idleMs);
      };
      xhr.upload.onprogress = (e) => {
        watch();
        if (opts.onProgress && e.lengthComputable) opts.onProgress(e.loaded / e.total);
      };
      xhr.upload.onload = () => clearTimeout(idle);
      if (opts.signal) {
        if (opts.signal.aborted) { reject(new DOMException('Aborted', 'AbortError')); return; }
        opts.signal.addEventListener('abort', () => xhr.abort());
      }
      // handleFailure throws (it is written for the axios interceptor's promise
      // chain); inside an event handler that throw would be uncaught and this
      // promise would hang, so route it through reject instead.
      const fail = (info: Parameters<typeof handleFailure>[0]) => {
        try { handleFailure(info); } catch (err) { reject(err); }
      };
      xhr.onload = () => {
        clearTimeout(idle);
        let parsed: unknown;
        try { parsed = JSON.parse(xhr.responseText); } catch { /* non-JSON error page */ }
        if (xhr.status >= 200 && xhr.status < 300) { resolve(parsed as T); return; }
        fail({ status: xhr.status, body: parsed ?? null, method: 'POST', url: this.url(path), skip: opts.silent });
      };
      xhr.onerror = () => { clearTimeout(idle); fail({ status: 0, body: null, method: 'POST', url: this.url(path), skip: opts.silent }); };
      xhr.onabort = () => {
        clearTimeout(idle);
        if (stalled) fail({ status: 0, body: null, method: 'POST', url: this.url(path), skip: opts.silent });
        else reject(new DOMException('Aborted', 'AbortError'));
      };
      xhr.send(form);
      watch();
    });
  }

  /** The URL to pass to EventSource/sharedEventSource for a streaming route
   *  under this client's base. Only for routes that don't check the bearer
   *  token: EventSource cannot set a header, and the server reads auth from
   *  `Authorization` only (`server/src/routes/auth.ts` getUserId) — a token
   *  query param here would silently 401 against any authenticated SSE
   *  route. An authenticated stream needs `streamEvents()` below instead. */
  eventSourceUrl(path: string): string {
    return this.url(path);
  }

  /** Read a Server-Sent-Events route that needs the bearer token, via fetch
   *  rather than EventSource (which cannot send one). Splits frames the same
   *  way `workflowApi.ts`'s `followJob` does; `onData` gets each frame's raw
   *  `data:` payload to parse. Resolves when the server ends the stream;
   *  rejects on an aborted `signal` or a non-OK response (after throwing the
   *  same ApiError a JSON call would). Cancellation closes the underlying
   *  connection, unlike sharedEventSource's auto-reconnect streams. Stays on
   *  fetch deliberately — axios does not stream response bodies, and no
   *  default timeout applies to an open stream. */
  async streamEvents(path: string, onData: (data: string) => void, opts: ClientOptions = {}): Promise<void> {
    const token = getAuthToken();
    const res = await fetch(this.url(path), { headers: token ? { Authorization: `Bearer ${token}` } : {}, signal: opts.signal });
    if (!res.ok) await sseFailure(res, 'GET', this.url(path), opts.silent);
    if (!res.body) return;
    const reader = res.body.getReader();
    // A throwing onData (or a read error) must not leave the body locked and
    // the connection open — cancel() first (closes the underlying stream;
    // a no-op if it already ended on its own), then release the lock so a
    // caller that wraps the body elsewhere isn't stuck behind this reader.
    try {
      const decoder = new TextDecoder();
      let buffer = '';
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let at;
        while ((at = buffer.indexOf('\n\n')) >= 0) {
          const frame = buffer.slice(0, at);
          buffer = buffer.slice(at + 2);
          const line = frame.split('\n').find(l => l.startsWith('data: '));
          if (line) onData(line.slice(6));
        }
      }
    } finally {
      await reader.cancel().catch(() => {});
      reader.releaseLock();
    }
  }

  /** A `/audio` or `/references` URL for `<audio>`/`<img>` src. Relative to
   *  `mediaRoot` (default '', same origin as the page) rather than this
   *  client's JSON `base` — a remote frontend origin configures `mediaRoot`
   *  separately, since media is served outside `/api`. */
  mediaUrl(path: string): string {
    if (path.startsWith('http')) return path;
    return `${this.mediaRoot}${path.startsWith('/') ? path : `/${path}`}`;
  }
}

/** The default client, base `/api` — what api.ts and most callers want. */
export const apiClient = new ApiClient();
