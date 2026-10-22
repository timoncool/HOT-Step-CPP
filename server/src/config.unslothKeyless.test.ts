import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { config, reloadEnvConfig } from './config.js';

// UNSLOTH_KEYLESS: only an explicit "1" turns it on, at startup and on reload.
const CASES: Array<[label: string, value: string | undefined, on: boolean]> = [
  ['unset', undefined, false],
  ['empty', '', false],
  ['0', '0', false],
  ['1', '1', true],
];

for (const [label, value, on] of CASES) {
  test(`startup: UNSLOTH_KEYLESS ${label} -> ${on ? 'on' : 'off'}`, () => {
    const env = { ...process.env };
    delete env.UNSLOTH_KEYLESS;
    if (value !== undefined) env.UNSLOTH_KEYLESS = value;
    const out = execFileSync(process.execPath, ['--import', 'tsx', '-e',
      "import('./src/config.ts').then(m => console.log('KEYLESS=' + m.config.lireek.unslothKeyless))"],
    { env, encoding: 'utf8', cwd: process.cwd() });
    assert.match(out, new RegExp(`KEYLESS=${on}`));
  });

  test(`reload: UNSLOTH_KEYLESS ${label} -> ${on ? 'on' : 'off'}`, () => {
    const saved = config.lireek.unslothKeyless;
    try {
      // Flip first so a no-op reload can't pass by accident.
      config.lireek.unslothKeyless = !on;
      reloadEnvConfig(value === undefined ? '' : `UNSLOTH_KEYLESS=${value}\n`);
      assert.equal(config.lireek.unslothKeyless, on);
    } finally { config.lireek.unslothKeyless = saved; }
  });
}
