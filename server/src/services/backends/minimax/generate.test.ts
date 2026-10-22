import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mapMinimaxParams } from './generate.js';

test('MM3 sends the shared runtime DoRA opt-in without changing legacy requests', () => {
  const params = { prompt: 'Instrumental ambient', seed: 42 };
  const legacy = mapMinimaxParams(params).req;
  assert.equal(Object.hasOwn(legacy, 'adapter_runtime_dora'), false);
  for (const adapterRuntimeDora of [false, 'true', 1]) {
    assert.deepEqual(mapMinimaxParams({ ...params, adapterRuntimeDora }).req, legacy);
  }
  const { adapter_runtime_dora, ...rest } = mapMinimaxParams({ ...params, adapterRuntimeDora: true }).req;
  assert.equal(adapter_runtime_dora, true);
  assert.deepEqual(rest, legacy);
});
