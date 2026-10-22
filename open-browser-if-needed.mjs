// open-browser-if-needed.mjs
// Opens the given URL in the default browser ONLY if no browser tab already has
// the app loaded. Waits for the server to come up, then asks /api/health how
// many tabs are connected (each open tab holds an SSE stream, so tabs left over
// from a previous run re-register within a few seconds of the server returning).
//
// Runs on Node rather than PowerShell: a launcher starting
// `powershell -ExecutionPolicy Bypass` gets the whole app tree quarantined by
// behaviour-based antivirus.
//
// Usage: node open-browser-if-needed.mjs <url>
import { spawn } from 'node:child_process';

const url = process.argv[2];
if (!url) {
  console.error('Usage: node open-browser-if-needed.mjs <url>');
  process.exit(2);
}
const health = url.replace(/\/+$/, '') + '/api/health';
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function getClients() {
  try {
    const res = await fetch(health, { signal: AbortSignal.timeout(5000) });
    return res.ok ? Number((await res.json()).clients) || 0 : null;
  } catch { return null; }
}

// Wait for the server (DB open takes a few seconds; a first-run UI build under
// LAUNCH.bat can take longer).
let deadline = Date.now() + 5 * 60_000;
let clients = null;
while (clients === null && Date.now() < deadline) {
  await sleep(1000);
  clients = await getClients();
}
if (clients === null) {
  console.log(`[HOT-Step] Server did not come up at ${health} - not opening a browser.`);
  process.exit(1);
}

// Give existing tabs a moment to reconnect (SSE retry is 2 s).
deadline = Date.now() + 8000;
while (clients === 0 && Date.now() < deadline) {
  await sleep(1000);
  clients = (await getClients()) ?? 0;
}

if (clients > 0) {
  console.log(`[HOT-Step] ${clients} browser tab(s) already connected - skipping.`);
} else {
  console.log(`[HOT-Step] Opening browser: ${url}`);
  // explorer.exe hands a URL to the default browser without a shell in between.
  spawn('explorer.exe', [url], { detached: true, stdio: 'ignore' }).unref();
}
