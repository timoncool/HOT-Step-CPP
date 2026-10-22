/**
 * Headless check of the mobile Sidebar drawer and the GlobalParamBar settings sheet:
 *   (1) focus moves inside on open, (2) Escape closes, (3) focus returns to the opener,
 *   (4) open at 412x915, resize to 1400x900 (desktop media query), the overlay closes.
 *
 * Usage (test-only server, never the live app):
 *   npx tsx scripts/check-drawer-focus.ts --server-dir ../server --port 3021 [--engine-port 8199] [--label new]
 *   npx tsx scripts/check-drawer-focus.ts --url http://127.0.0.1:3021      (server already running)
 * With --server-dir the script spawns `node --import tsx src/index.ts` itself, with DATA_DIR and a
 * nonexistent engine exe inside <server-dir>/../_testdata, and stops that child when done.
 * Exit 1 if any check fails.
 */
import type { Page } from 'playwright';
import { createRequire } from 'node:module';
import { execSync } from 'node:child_process';
import { spawn, type ChildProcess } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';

const args = process.argv.slice(2);
const arg = (k: string, d = '') => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const port = arg('--port', '3021');
const enginePort = arg('--engine-port', '8199');
const label = arg('--label', 'run');
const serverDir = arg('--server-dir');
let url = arg('--url', `http://127.0.0.1:${port}`);
for (const p of ['3001', '3011', '8085', '8095']) {
  if (port === p || enginePort === p || url.endsWith(`:${p}`)) throw new Error(`refusing protected port ${p}`);
}

async function waitUp(u: string, ms = 90000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    try { const r = await fetch(u); if (r.ok) return; } catch { /* not yet */ }
    await new Promise(r => setTimeout(r, 500));
  }
  throw new Error(`server at ${u} did not come up`);
}

// Playwright is not a ui devDependency: use a local copy if present, else the global npm one.
function loadPlaywright(): typeof import('playwright') {
  const req = createRequire(import.meta.url);
  try { return req('playwright'); } catch {
    const env = { ...process.env }; delete env.npm_config_prefix; delete env.NPM_CONFIG_PREFIX;
    const g = execSync('npm root -g', { encoding: 'utf-8', env }).trim();
    return req(path.join(g, 'playwright'));
  }
}
const { chromium } = loadPlaywright();

let child: ChildProcess | null = null;
async function startServer() {
  if (!serverDir) return;
  const sd = path.resolve(serverDir);
  const data = path.resolve(sd, '..', '_testdata');
  fs.mkdirSync(data, { recursive: true });
  const log = fs.openSync(path.join(data, `server_${port}.log`), 'a');
  child = spawn(process.execPath, ['--import', 'tsx', 'src/index.ts'], {
    cwd: sd, stdio: ['ignore', log, log],
    env: {
      ...process.env, SERVER_PORT: port, DATA_DIR: data,
      ACESTEPCPP_EXE: path.join(data, 'no-such-ace-server.exe'), ACESTEPCPP_PORT: enginePort,
      ACESTEPCPP_WARM_ON_STARTUP: '0', CUDA_VISIBLE_DEVICES: '0',
    },
  });
  url = `http://127.0.0.1:${port}`;
  await waitUp(url + '/');
}

const describe = (page: Page) => page.evaluate(() => {
  const a = document.activeElement as HTMLElement | null;
  if (!a) return 'null';
  const bits = [a.tagName.toLowerCase()];
  for (const k of ['role', 'aria-label']) { const v = a.getAttribute(k); if (v) bits.push(`${k}=${v}`); }
  if (a === document.body) return 'body';
  return bits.join(' ');
});

type Target = {
  name: string; opener: string;
  isOpen: (p: Page) => Promise<boolean>;
  focusInside: (p: Page) => Promise<boolean>;
};
const drawer: Target = {
  name: 'Sidebar drawer', opener: 'button[aria-label="Open navigation"]',
  isOpen: p => p.locator('button[aria-label="Close navigation"]').count().then(n => n > 0),
  focusInside: p => p.evaluate(() => {
    const c = document.querySelector('button[aria-label="Close navigation"]')?.closest('.fixed');
    return !!c && c.contains(document.activeElement) ;
  }),
};
const sheet: Target = {
  name: 'Settings sheet', opener: 'button[aria-label="Settings"]:visible',
  isOpen: p => p.locator('[role="dialog"][aria-label="Settings"]').count().then(n => n > 0),
  focusInside: p => p.evaluate(() => {
    const d = document.querySelector('[role="dialog"][aria-label="Settings"]');
    return !!d && d.contains(document.activeElement);
  }),
};

const rows: { t: string; vp: string; check: string; res: string; val: string }[] = [];
const add = (t: string, vp: string, check: string, ok: boolean | null, val: string) =>
  rows.push({ t, vp, check, res: ok === null ? 'N/A' : ok ? 'PASS' : 'FAIL', val });

async function run() {
  await startServer();
  const browser = await chromium.launch();
  try {
    for (const vp of [{ width: 412, height: 915 }, { width: 1400, height: 900 }]) {
      const vpn = `${vp.width}x${vp.height}`;
      for (const t of [drawer, sheet]) {
        const page = await browser.newPage({ viewport: vp });
        await page.goto(url, { waitUntil: 'networkidle' });
        const op = page.locator(t.opener).first();
        if (!(await op.count()) || !(await op.isVisible())) {
          for (const c of ['1 focus in', '2 Esc closes', '3 focus back']) add(t.name, vpn, c, null, 'opener hidden');
          await page.close(); continue;
        }
        await op.focus();
        await op.press('Enter');
        await page.waitForTimeout(300);
        const open1 = await t.isOpen(page);
        const inside = await t.focusInside(page);
        add(t.name, vpn, '1 focus in', open1 && inside, `open=${open1} active=${await describe(page)}`);
        await page.keyboard.press('Escape');
        await page.waitForTimeout(300);
        const open2 = await t.isOpen(page);
        add(t.name, vpn, '2 Esc closes', !open2, `open=${open2}`);
        const back = await op.evaluate(el => el === document.activeElement);
        add(t.name, vpn, '3 focus back', !open2 && back, `active=${await describe(page)}`);
        await page.close();
      }
    }
    for (const t of [drawer, sheet]) {
      const page = await browser.newPage({ viewport: { width: 412, height: 915 } });
      await page.goto(url, { waitUntil: 'networkidle' });
      await page.locator(t.opener).first().click();
      await page.waitForTimeout(300);
      const before = await t.isOpen(page);
      await page.setViewportSize({ width: 1400, height: 900 });
      await page.waitForTimeout(500);
      const after = await t.isOpen(page);
      await page.setViewportSize({ width: 412, height: 915 });
      await page.waitForTimeout(300);
      const back = await t.isOpen(page);
      add(t.name, '412->1400', '4 closes on desktop', before && !after && !back,
        `open@412=${before} open@1400=${after} open@back412=${back}`);
      await page.close();
    }
  } finally {
    await browser.close();
  }
  console.log(`\n[${label}] ${url}`);
  console.log('| target | viewport | check | result | values |\n|---|---|---|---|---|');
  for (const r of rows) console.log(`| ${r.t} | ${r.vp} | ${r.check} | ${r.res} | ${r.val} |`);
  return rows.some(r => r.res === 'FAIL') ? 1 : 0;
}

run().then(c => { child?.kill(); process.exit(c); }, e => { console.error(e); child?.kill(); process.exit(2); });
