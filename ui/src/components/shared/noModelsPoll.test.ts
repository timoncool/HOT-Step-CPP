import assert from 'node:assert/strict';
import test from 'node:test';
import { noModelsRefetchInterval, NO_MODELS_POLL_MS } from './noModelsPoll';

const snap = (ready: boolean, noModels: boolean) => ({
  state: { data: { engine: { ready, bootStatus: '', noModels } }, status: 'success', errorUpdateCount: 0 },
}) as any;

test('keeps polling through ready=true so a later noModels=true is seen', () => {
  assert.equal(noModelsRefetchInterval(snap(true, false)), NO_MODELS_POLL_MS);   // startup snapshot
  assert.equal(noModelsRefetchInterval(snap(false, true)), NO_MODELS_POLL_MS);   // after the scan
});
