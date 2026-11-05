// Same-origin guard for state-changing update calls. The API is open to any
// page (cors()), so a POST must come from this app: the page's own origin, or
// the Vite dev server on :3000, which proxies with changeOrigin and so sends
// Host 127.0.0.1:3001 beside Origin http://localhost:3000. Only a loopback
// Origin on that port, aimed at a loopback Host, is accepted that way.
const DEV_UI_PORT = '3000';
const LOOPBACK = new Set(['localhost', '127.0.0.1', '[::1]']);

export function trustedOrigin(origin: string | undefined, host: string | undefined): boolean {
  if (!origin) return true;
  try {
    const o = new URL(origin);
    if (o.host === host) return true;
    const h = host ? new URL(`http://${host}`) : null;
    return o.protocol === 'http:' && o.port === DEV_UI_PORT && LOOPBACK.has(o.hostname) && !!h && LOOPBACK.has(h.hostname);
  } catch { return false; }
}
