// Release gate: boot a freshly built server.mjs and require it to reach ready.
// Usage: node release/smoke-boot.mjs <server.mjs> [timeoutSeconds]
// Run it with the same node the release ships. Exits 0 only if /api/health answers.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';

const [bundleArg, secs = '60'] = process.argv.slice(2);
const bundle = bundleArg && path.resolve(bundleArg);
if (!bundle || !fs.existsSync(bundle)) { console.error(`smoke-boot: bundle not found: ${bundle}`); process.exit(2); }

const freePort = () => new Promise((res, rej) => {
  const s = net.createServer().once('error', rej).listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => res(port)); });
});

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hotstep-smoke-'));
// The release root holds server/data, so boot from it; data and profile go to tmp.
const root = path.dirname(path.dirname(bundle));
const dataRoot = path.join(tmp, 'data');
// No engine: a missing exe is skipped, and a cpu marker beside it skips the CUDA runtime download.
const engineDir = path.join(tmp, 'engine');
fs.mkdirSync(dataRoot);
fs.mkdirSync(engineDir);
fs.writeFileSync(path.join(engineDir, '.variant'), 'cpu');
const [port, enginePort] = [await freePort(), await freePort()];
const child = spawn(process.execPath, [bundle], {
  cwd: root,
  stdio: ['ignore', 'pipe', 'pipe'],
  env: {
    ...process.env, HOT_STEP_ROOT: root, HOT_STEP_DATA_ROOT: dataRoot, HOT_STEP_PROFILE_DIR: path.join(tmp, 'profile'),
    ACESTEPCPP_EXE: path.join(engineDir, 'ace-server-absent'), SERVER_PORT: String(port), ACESTEPCPP_PORT: String(enginePort),
  },
});
let log = '';
for (const s of [child.stdout, child.stderr]) s.on('data', d => { log += d; });

const exited = new Promise(res => child.once('exit', (code, sig) => res({ code, sig })));
const timeout = new Promise(res => setTimeout(() => res('timeout'), Number(secs) * 1000));
const ready = (async () => {
  for (;;) {
    try { if ((await fetch(`http://127.0.0.1:${port}/api/health`)).ok) return 'ready'; } catch { /* not up yet */ }
    await new Promise(r => setTimeout(r, 500));
  }
})();

const result = await Promise.race([ready, exited, timeout]);
if (result === 'ready') {
  console.log(`smoke-boot: PASS - ${bundle} answered /api/health on port ${port}`);
} else {
  const why = result === 'timeout' ? `no /api/health answer within ${secs}s`
    : `exited before ready (code ${result.code}, signal ${result.sig})`;
  console.error(`smoke-boot: FAIL - ${why}\n--- server output (last 40 lines) ---\n${log.split('\n').slice(-40).join('\n')}`);
}
child.kill(); // our own child PID only
await Promise.race([exited, new Promise(r => setTimeout(r, 5000))]);
fs.rmSync(tmp, { recursive: true, force: true });
process.exit(result === 'ready' ? 0 : 1);
