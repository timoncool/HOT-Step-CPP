// Waveform.tsx - canvas waveform from analysis peaks, with a live playhead,
// loop overlay and drag-to-set-loop. A short tap seeks instead.

import React, { useEffect, useRef, useState } from 'react';
import type { AbEngine, LoopRegion } from './useAbEngine';

interface Props {
  peaks: { min: number[]; max: number[] } | null;
  duration: number;
  engine: AbEngine;
  active: boolean;
  height?: number;
  /** Called on a tap (not a drag), with the tapped time in seconds. */
  onTap?: (sec: number) => void;
}

/** Playhead line, isolated so only this re-renders every frame. */
export const Playhead: React.FC<{ engine: AbEngine; duration: number; className?: string }> = ({ engine, duration, className }) => {
  const t = engine.useCurrentTime();
  if (!duration) return null;
  const pct = Math.min(Math.max(t / duration, 0), 1) * 100;
  return <div className={`absolute top-0 bottom-0 w-0.5 pointer-events-none ${className ?? 'bg-pink-400'}`} style={{ left: `${pct}%` }} />;
};

export const LoopOverlay: React.FC<{ loop: LoopRegion | null; duration: number }> = ({ loop, duration }) => {
  if (!loop || !duration) return null;
  const l = (loop.start / duration) * 100;
  const w = ((loop.end - loop.start) / duration) * 100;
  return <div className="absolute top-0 bottom-0 bg-sky-400/20 border-x border-sky-400/60 pointer-events-none" style={{ left: `${l}%`, width: `${w}%` }} />;
};

export const Waveform: React.FC<Props> = ({ peaks, duration, engine, active, height = 56, onTap }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);
  const drag = useRef<{ x0: number; t0: number; moved: boolean } | null>(null);
  const [preview, setPreview] = useState<LoopRegion | null>(null);

  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !width) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.floor(width * dpr);
    c.height = Math.floor(height * dpr);
    const g = c.getContext('2d');
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, width, height);
    if (!peaks || !peaks.max.length) return;
    const n = peaks.max.length;
    const mid = height / 2;
    g.fillStyle = active ? 'rgba(244,114,182,0.85)' : 'rgba(161,161,170,0.6)';
    for (let x = 0; x < width; x++) {
      const i0 = Math.floor((x / width) * n);
      const i1 = Math.max(i0 + 1, Math.floor(((x + 1) / width) * n));
      let lo = 0, hi = 0;
      for (let i = i0; i < i1 && i < n; i++) {
        if (peaks.min[i] < lo) lo = peaks.min[i];
        if (peaks.max[i] > hi) hi = peaks.max[i];
      }
      const y0 = mid - hi * mid;
      const y1 = mid - lo * mid;
      g.fillRect(x, y0, 1, Math.max(1, y1 - y0));
    }
  }, [peaks, width, height, active]);

  const timeAt = (clientX: number): number => {
    const r = boxRef.current?.getBoundingClientRect();
    if (!r || !duration) return 0;
    return Math.min(Math.max((clientX - r.left) / r.width, 0), 1) * duration;
  };

  const onDown = (e: React.PointerEvent) => {
    e.stopPropagation();
    (e.target as Element).setPointerCapture?.(e.pointerId);
    drag.current = { x0: e.clientX, t0: timeAt(e.clientX), moved: false };
  };
  const onMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    if (Math.abs(e.clientX - d.x0) > 6) d.moved = true;
    if (d.moved) {
      const t1 = timeAt(e.clientX);
      setPreview({ start: Math.min(d.t0, t1), end: Math.max(d.t0, t1) });
    }
  };
  const onUp = (e: React.PointerEvent) => {
    e.stopPropagation();
    const d = drag.current;
    drag.current = null;
    if (!d) return;
    if (d.moved) {
      const t1 = timeAt(e.clientX);
      const lp = { start: Math.min(d.t0, t1), end: Math.max(d.t0, t1) };
      setPreview(null);
      if (lp.end - lp.start > 0.1) engine.setLoop(lp);
    } else {
      onTap?.(d.t0);
    }
  };

  return (
    <div
      ref={boxRef}
      className="relative w-full select-none touch-none cursor-crosshair rounded-md bg-zinc-100 dark:bg-zinc-900/60 overflow-hidden"
      style={{ height }}
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerCancel={() => { drag.current = null; setPreview(null); }}
    >
      <canvas ref={canvasRef} style={{ width: '100%', height }} />
      <LoopOverlay loop={preview ?? engine.loop} duration={duration} />
      <Playhead engine={engine} duration={duration} className={active ? 'bg-pink-400' : 'bg-zinc-400/70'} />
    </div>
  );
};
