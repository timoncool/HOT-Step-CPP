// Preload for the managed production launcher. Keeps promotion admission
// control outside the application sources owned by the server slice.
import http from 'node:http';

export function maintenanceHandler(handler, token) {
  let paused = false;
  let lease;
  const activeWrites = new Set();
  return function (req, res) {
    const trusted = req.headers['x-hotstep-prod-token'] === token;
    if (req.url === '/__prod/maintenance' && req.method === 'POST') {
      if (!trusted) { res.writeHead(403).end(); return; }
      paused = req.headers['x-hotstep-prod-action'] !== 'resume';
      clearTimeout(lease);
      if (paused) lease = setTimeout(() => { paused = false; }, 60_000).unref();
      const deadline = Date.now() + 15_000;
      const drain = () => {
        if (!paused || !activeWrites.size) {
          res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ paused, drained: !activeWrites.size }));
        } else if (Date.now() >= deadline) {
          paused = false;
          res.writeHead(409).end('Requests still active; production resumed');
        } else setTimeout(drain, 50);
      };
      drain(); return;
    }
    // Browser reads, including preview audio and SSE, can continue. New
    // submissions and cancellations are rejected once the lease is acquired.
    if (paused && !['GET', 'HEAD', 'OPTIONS'].includes(req.method) && !trusted) {
      res.writeHead(503, { 'retry-after': '30' }).end('Production promotion in progress'); return;
    }
    if (!['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
      activeWrites.add(req);
      const done = () => activeWrites.delete(req);
      res.once('finish', done); res.once('close', done);
    }
    return handler.call(this, req, res);
  };
}

if (process.env.HOTSTEP_PROD_TOKEN) {
  const original = http.createServer;
  http.createServer = function (...args) {
    const index = typeof args[0] === 'function' ? 0 : 1;
    if (typeof args[index] === 'function') args[index] = maintenanceHandler(args[index], process.env.HOTSTEP_PROD_TOKEN);
    return original.apply(this, args);
  };
}
