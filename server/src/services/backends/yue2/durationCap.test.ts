// What this guards: a YuE2 render longer than the requested duration is
// trimmed to that duration with an audible fade-out, and auto length (<= 0)
// leaves the file alone. Run: npx tsx src/services/backends/yue2/durationCap.test.ts
import fs from 'fs';
import os from 'os';
import path from 'path';
import { applyYue2DurationCap } from './durationCap.js';
import { wavDurationSec } from '../../audioCrop.js';

function writeTone(file: string, seconds: number, sr = 48000): void {
  const frames = Math.round(seconds * sr);
  const buf = Buffer.alloc(44 + frames * 4);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + frames * 4, 4); buf.write('WAVE', 8);
  buf.write('fmt ', 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(2, 22);
  buf.writeUInt32LE(sr, 24); buf.writeUInt32LE(sr * 4, 28); buf.writeUInt16LE(4, 32); buf.writeUInt16LE(16, 34);
  buf.write('data', 36); buf.writeUInt32LE(frames * 4, 40);
  for (let i = 0; i < frames; i++) {
    const v = Math.round(16000 * Math.sin(2 * Math.PI * 440 * i / sr));
    buf.writeInt16LE(v, 44 + i * 4); buf.writeInt16LE(v, 46 + i * 4);
  }
  fs.writeFileSync(file, buf);
}
function rmsDb(file: string, fromSec: number, toSec: number, sr = 48000): number {
  const b = fs.readFileSync(file); let acc = 0; let n = 0;
  for (let i = Math.floor(fromSec * sr); i < Math.floor(toSec * sr); i++) { const v = b.readInt16LE(44 + i * 4) / 32768; acc += v * v; n++; }
  return 10 * Math.log10(acc / n + 1e-12);
}

let failed = 0;
const check = (name: string, ok: boolean, detail: string) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${name}: ${detail}`); if (!ok) failed++; };
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2cap-'));
try {
  const a = path.join(dir, 'a.wav'); writeTone(a, 42);
  const r = applyYue2DurationCap(a, { duration: 15 });
  const d = wavDurationSec(a);
  check('42s render capped to 15s', Math.abs(d - 15) < 0.01 && r.trimmed, `duration=${d.toFixed(3)}s trimmed=${r.trimmed}`);
  const body = rmsDb(a, 10, 12), tail = rmsDb(a, 14.9, 15);
  check('fade-out drops tail >= 20 dB', body - tail >= 20, `body=${body.toFixed(1)} dB tail=${tail.toFixed(1)} dB drop=${(body - tail).toFixed(1)} dB`);

  const b = path.join(dir, 'b.wav'); writeTone(b, 42);
  const rb = applyYue2DurationCap(b, { duration: -1 });
  check('auto length untouched', !rb.trimmed && Math.abs(wavDurationSec(b) - 42) < 0.01, `duration=${wavDurationSec(b).toFixed(3)}s`);

  const c = path.join(dir, 'c.wav'); writeTone(c, 12);
  const rc = applyYue2DurationCap(c, { duration: 15 });
  check('shorter render untouched', !rc.trimmed && Math.abs(wavDurationSec(c) - 12) < 0.01, `duration=${wavDurationSec(c).toFixed(3)}s`);
} finally {
  fs.rmSync(dir, { recursive: true, force: true });
}
if (failed) { console.log(`${failed} check(s) failed`); process.exit(1); }
console.log('all checks passed');
