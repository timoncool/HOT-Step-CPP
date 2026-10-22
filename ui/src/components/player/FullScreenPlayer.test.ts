import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';

function fixture() {
  const source = readFileSync(new URL('./FullScreenPlayer.tsx', import.meta.url), 'utf8');
  const listeners = new Set<() => void>();
  const stack: any[] = [null];
  const pending: (() => void)[] = [];
  let index = 0;
  let effect: () => (() => void);
  let refs: any[] = [];
  let refIndex = 0;
  let closed = 0;
  const body = {};
  let rendered: any;
  const window = {
    history: {
      get state() { return stack[index]; },
      pushState(state: any) { stack.splice(++index); stack[index] = state; },
      replaceState(state: any) { stack[index] = state; },
      back() { pending.push(() => { index = Math.max(0, index - 1); for (const fn of [...listeners]) fn(); }); },
    },
    addEventListener(type: string, fn: () => void, options?: any) {
      if (type !== 'popstate') return;
      if (options?.once) { const once = () => { listeners.delete(once); fn(); }; listeners.add(once); }
      else listeners.add(fn);
    },
    removeEventListener(type: string, fn: () => void) { if (type === 'popstate') listeners.delete(fn); },
  };
  const exports: any = {};
  vm.runInNewContext(ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText, {
    exports, window, document: { body }, queueMicrotask,
    require: (id: string) => {
      if (id === 'react') return {
        useEffect: (fn: any) => { effect = fn; },
        useRef: (value: any) => { const i = refIndex++; return refs[i] ??= { current: value }; },
      };
      if (id === 'react-dom') return { createPortal: (children: any, container: any) => ({ children, container }) };
      if (id === 'react/jsx-runtime') return { jsx: (type: any, props: any) => ({ type, props }), jsxs: (type: any, props: any) => ({ type, props }) };
      if (id === 'react-i18next') return { useTranslation: () => ({ t: (s: string) => s }) };
      if (id.includes('useDisguiseMode')) return { useDisguiseMode: () => ({ disguiseTitle: (title: string) => title }) };
      return {};
    },
  });
  const render = (fresh = false) => {
    if (fresh) refs = [];
    refIndex = 0;
    rendered = exports.FullScreenPlayer({ song: { id: 'track' }, onClose: () => { closed++; } });
    return effect!;
  };
  return { window, render, pending, body, get rendered() { return rendered; }, get closed() { return closed; }, get index() { return index; } };
}
const settle = async () => { await Promise.resolve(); await Promise.resolve(); };

test('phone dialog renders into the body outside player transform and stacking ancestors', () => {
  const f = fixture();
  f.render();
  assert.equal(f.rendered.container, f.body);
  assert.equal(f.rendered.children.props.role, 'dialog');
});

test('StrictMode replay keeps one entry; Back closes once and cleanup never navigates again', async () => {
  const f = fixture();
  const setup = f.render();
  setup()(); const cleanup = setup();
  await settle();
  assert.equal(f.pending.length, 0);
  assert.equal(f.index, 1);
  f.window.history.back(); f.pending.shift()!();
  assert.equal(f.closed, 1);
  cleanup(); await settle();
  assert.equal(f.pending.length, 0);
});

test('close and reopen before cleanup reclaims the entry; final close removes it', async () => {
  const f = fixture();
  f.render()()();
  const cleanup = f.render(true)();
  await settle();
  assert.equal(f.index, 1);
  assert.equal(f.pending.length, 0);
  cleanup(); await settle();
  assert.equal(f.pending.length, 1);
  f.pending.shift()!(); await settle();
  assert.equal(f.index, 0);
});

test('reopening waits for an already queued Back before acquiring a fresh entry', async () => {
  const f = fixture();
  f.render()()(); await settle();
  assert.equal(f.pending.length, 1);
  const cleanup = f.render(true)();
  f.pending.shift()!(); await settle();
  assert.equal(f.closed, 0);
  assert.equal(f.index, 1);
  cleanup(); await settle(); f.pending.shift()!();
  assert.equal(f.index, 0);
});

test('cleanup after studio navigation cannot pop the new studio entry', async () => {
  const f = fixture();
  const cleanup = f.render()();
  f.window.history.pushState({ studio: 'library' });
  cleanup(); await settle();
  assert.equal(f.pending.length, 0);
  assert.equal(f.window.history.state.studio, 'library');
});
