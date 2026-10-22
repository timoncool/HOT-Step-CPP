import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';

test('clearing a paused track resets Media Session caches before another paused selection', () => {
  const source = readFileSync(new URL('./playbackStore.ts', import.meta.url), 'utf8');
  const mediaSource = source.slice(source.indexOf('let _msTrackId:'));
  let sync: () => void;
  const positions: any[] = [];
  const ms = {
    metadata: null as any,
    playbackState: 'none',
    setActionHandler() {},
    setPositionState(position: any) { positions.push(position); },
  };
  const state: any = { currentTrack: { id: 'a', title: 'First' }, isPlaying: false, currentTime: 0, duration: 30, playbackRate: 1 };
  vm.runInNewContext(ts.transpileModule(mediaSource, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText, {
    navigator: { mediaSession: ms }, _state: state,
    MediaMetadata: class { constructor(data: any) { Object.assign(this, data); } },
    i18n: { t: (key: string) => key }, subscribe: (fn: () => void) => { sync = fn; },
  });
  assert.equal(ms.playbackState, 'paused');
  assert.equal(positions.length, 1);
  state.currentTrack = null; sync!();
  assert.equal(ms.playbackState, 'none');
  assert.equal(ms.metadata, null);
  assert.equal(positions.length, 2, 'clearing a track resets the native position state');
  assert.equal(positions[1], undefined);
  state.currentTrack = { id: 'b', title: 'Second' }; sync!();
  assert.equal(ms.playbackState, 'paused');
  assert.equal(ms.metadata.title, 'Second');
  assert.equal(positions.length, 3);
  state.currentTrack = null; sync!();
  state.currentTrack = { id: 'b', title: 'Second' }; sync!();
  assert.equal(ms.playbackState, 'paused');
  assert.equal(positions.length, 5, 'reselecting the same paused track also republishes its position');
});
