import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { hasRuntimeFile } from './runtimeFiles.js';

test('runtime DLL in data-folder tools/runtime counts as present with none beside the exe', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hs-rt-'));
  try {
    const exeDir = path.join(root, 'engine');
    const dataRuntime = path.join(root, 'HOT-Step-Data', 'tools', 'runtime');
    fs.mkdirSync(exeDir, { recursive: true });
    fs.mkdirSync(dataRuntime, { recursive: true });
    fs.writeFileSync(path.join(dataRuntime, 'cublas64_13.dll'), '');
    assert.equal(hasRuntimeFile([dataRuntime, exeDir], 'cublas64_13.dll'), true);
    assert.equal(hasRuntimeFile([dataRuntime, exeDir], 'cudart64_13.dll'), false);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
