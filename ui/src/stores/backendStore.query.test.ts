import assert from 'node:assert/strict';
import test, { afterEach } from 'node:test';
import { useBackendStore } from './backendStore';
import { queryClient } from '../queries/client';
import { capabilitiesOptions } from '../queries/global-models/options';

const originalFetch = globalThis.fetch;
afterEach(() => { queryClient.clear(); globalThis.fetch = originalFetch; });
const json = (data: unknown) => new Response(JSON.stringify(data));

test('backend switch keeps B selected when A capabilities finish late', async () => {
  let release!: () => void;
  const held = new Promise<void>(resolve => { release = resolve; });
  globalThis.fetch = async input => {
    const path = String(input);
    if (path === '/api/backends/active') return json({ activeId: 'B' });
    if (path.includes('capabilities?backend=A')) { await held; return json({ up: true, displayName: 'A' }); }
    if (path.includes('capabilities?backend=B')) return json({ up: true, displayName: 'B' });
    if (path.includes('/backends/models?backend=B')) return json({ selectable: false, buckets: {} });
    throw new Error(`Unexpected request ${path}`);
  };
  const a = useBackendStore.getState().fetchCapabilities('A');
  await useBackendStore.getState().switchBackend('B');
  release(); await a;
  const state = useBackendStore.getState();
  assert.equal(state.activeBackendId, 'B');
  assert.equal(state.capabilities.B.displayName, 'B');
  assert.equal(state.capabilities.A.displayName, 'A');
});

test('down recovery polls the raw manifest while compatibility readers retain the last up manifest', async () => {
  const options = capabilitiesOptions('recovery');
  let up = false;
  globalThis.fetch = async () => json({ up, core: { modelsMissing: !up } });
  await useBackendStore.getState().fetchCapabilities('recovery');
  assert.equal(useBackendStore.getState().capabilities.recovery.up, false);
  up = true;
  await queryClient.invalidateQueries({ queryKey: options.queryKey });
  await useBackendStore.getState().fetchCapabilities('recovery');
  const usable = useBackendStore.getState().capabilities.recovery;
  assert.equal(usable.up, true);
  up = false;
  await queryClient.invalidateQueries({ queryKey: options.queryKey });
  await useBackendStore.getState().fetchCapabilities('recovery');
  assert.equal(queryClient.getQueryData(options.queryKey)?.up, false);
  assert.equal(useBackendStore.getState().capabilities.recovery, usable);
  const query = queryClient.getQueryCache().find({ queryKey: options.queryKey })!;
  assert.equal(options.refetchInterval(query as never), 10_000);
});
