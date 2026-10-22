// LyricsBar.tsx — Bottom bar showing synced lyrics one line at a time
// Displayed between the waveform and transport controls when playing.
// Ported from hot-step-9000.

import React, { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ChevronDown, ChevronUp, Music } from 'lucide-react';
import { parseLrc, type LrcLine } from '../../utils/lrcUtils';
import { findCurrentLineIndex, findActiveWordIndex } from '../../utils/wordLrcUtils';
import { lrcOptions, lyricsJsonOptions } from '../../queries/shell-queue/lyrics';
import { WordHighlighter } from './WordHighlighter';

interface LyricsBarProps {
    audioUrl?: string;
    currentTime: number;
    isPlaying: boolean;
}

function findCurrentIndex(lines: LrcLine[], time: number): number {
    if (lines.length === 0) return -1;
    let lo = 0, hi = lines.length - 1, result = -1;
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (lines[mid].time <= time) {
            result = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return result;
}

export const LyricsBar: React.FC<LyricsBarProps> = ({ audioUrl, currentTime }) => {
    const [expanded, setExpanded] = useState(true);

    // Word-level .lyrics.json takes priority; .lrc is the fallback. Both are
    // shared sidecar queries, joined with SectionMarkers' reads of the same track.
    const wordData = useQuery(lyricsJsonOptions(audioUrl ?? '')).data ?? null;
    const lrc = useQuery(lrcOptions(audioUrl ?? '')).data ?? null;
    const fetchedLrc = wordData ? null : lrc;

    const lines = useMemo(() => fetchedLrc ? parseLrc(fetchedLrc) : [], [fetchedLrc]);
    const currentIdx = findCurrentIndex(lines, currentTime);

    // Word-level indices
    const wordLineIdx = wordData ? findCurrentLineIndex(wordData.lines, currentTime) : -1;
    const wordLine = wordData && wordLineIdx >= 0 ? wordData.lines[wordLineIdx] : null;
    const activeWordIdx = wordLine ? findActiveWordIndex(wordLine.words, currentTime) : -1;

    // Section marker — show when the current line's section differs from the previous
    const currentSection = wordLine?.section || null;
    const prevSection = wordData && wordLineIdx > 0 ? wordData.lines[wordLineIdx - 1]?.section : null;
    const showSection = currentSection && currentSection !== prevSection;

    // Derive display text for LRC fallback
    const displayText = currentIdx >= 0 ? lines[currentIdx]?.text ?? '' : '';

    // Don't render if no lyrics data at all
    if (lines.length === 0 && !wordData) return null;

    return (
        <div className="flex-shrink-0 border-t border-zinc-200 dark:border-white/5 bg-black/80 backdrop-blur-sm z-30 transition-all duration-300">
            {/* Collapse/Expand toggle tab */}
            <button
                onClick={() => setExpanded(!expanded)}
                className="w-full flex items-center justify-center gap-2 py-1.5 text-xs text-zinc-500 hover:text-zinc-700 dark:text-zinc-300 transition-colors"
            >
                <Music size={11} className="text-pink-500/60" />
                <span className="font-medium tracking-wide uppercase text-[10px]">Lyrics</span>
                {wordData && (
                    <span className="px-1.5 py-0.5 rounded text-[8px] font-bold tracking-wider bg-pink-500/20 text-pink-400 uppercase">
                        Whisper
                    </span>
                )}
                {expanded ? <ChevronDown size={11} /> : <ChevronUp size={11} />}
            </button>

            {/* Lyrics content */}
            <div
                className="overflow-hidden transition-all duration-300 ease-out"
                style={{ maxHeight: expanded ? '60px' : '0px', opacity: expanded ? 1 : 0 }}
            >
                <div className="px-8 pb-3 flex items-center justify-center gap-3">
                    {/* Section badge */}
                    {showSection && (
                        <span
                            className="flex-shrink-0 px-2.5 py-0.5 rounded-full text-[10px] font-semibold tracking-wider uppercase bg-gradient-to-r from-violet-500/25 to-pink-500/25 text-violet-300 border border-violet-500/20"
                            style={{ animation: 'fade-in 0.3s ease-out' }}
                        >
                            {currentSection}
                        </span>
                    )}
                    <span
                        className="text-lg md:text-xl font-bold text-white tracking-wide text-center transition-opacity duration-300"
                        style={{
                            textShadow: wordData ? undefined : '0 0 30px rgba(236, 72, 153, 0.4), 0 2px 8px rgba(0,0,0,0.5)',
                        }}
                    >
                        {wordData && wordLine ? (
                            <WordHighlighter words={wordLine.words} activeWordIndex={activeWordIdx} />
                        ) : (
                            displayText || '♪ ♪ ♪'
                        )}
                    </span>
                </div>
            </div>
        </div>
    );
};
