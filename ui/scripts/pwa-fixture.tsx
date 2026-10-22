import { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { createPortal } from 'react-dom';
import { BottomSheet } from '../src/components/shared/BottomSheet';
import '../src/index.css';

function Child({ close }: { close: () => void }) {
  useEffect(() => {
    const previous = document.activeElement as HTMLElement;
    document.getElementById('child-first')?.focus();
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') close(); };
    document.addEventListener('keydown', key);
    return () => { document.removeEventListener('keydown', key); previous?.focus(); };
  }, [close]);
  return createPortal(<div data-portal-layer role="dialog" aria-label="Child" className="fixed inset-0 z-[70] bg-white">
    <button id="child-first" onClick={close}>Close child</button><button id="child-last">Child last</button>
  </div>, document.body);
}

function Fixture() {
  const [open, setOpen] = useState(false);
  const [child, setChild] = useState(false);
  return <><button id="opener" onClick={() => setOpen(true)}>Open sheet</button>
    <BottomSheet open={open} onClose={() => setOpen(false)} ariaLabel="Parent">
      <button id="sheet-first" onClick={() => setChild(true)}>Open child</button>
      <button id="sheet-last">Sheet last</button>
      {child && <Child close={() => setChild(false)} />}
    </BottomSheet>
  </>;
}
createRoot(document.getElementById('root')!).render(<Fixture />);
