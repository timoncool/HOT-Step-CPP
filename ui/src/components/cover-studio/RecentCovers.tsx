// RecentCovers.tsx — Recent covers sub-component for Cover Studio

import { useQuery } from '@tanstack/react-query';
import { coverSongsOptions, deleteCovers } from '../../queries/cover-lyrics/songs';
import { queryClient } from '../../queries/client';
import { queryManager } from '../../queries/invalidation';
import React, { useState, useEffect, useRef } from 'react';
import { Music, Play, Pause, Loader2, Trash2 } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import type { Song } from '../../types';
import { playFromList, togglePlay, songToTrack, usePlaybackSelector } from '../../stores/playbackStore';

interface RecentCoversProps {
  refreshTrigger: number;
}

export const RecentCovers: React.FC<RecentCoversProps> = ({ refreshTrigger }) => {
  const { token } = useAuth();
  const currentTrackId = usePlaybackSelector(s => s.currentTrack?.id ?? null);
  const isPlaying = usePlaybackSelector(s => s.isPlaying);
  const options = coverSongsOptions();
  const snapshot = useQuery({ ...options, enabled: !!token });
  const covers = (snapshot.data?.songs ?? []).slice().sort((a, b) =>
    new Date(b.createdAt ?? 0).getTime() - new Date(a.createdAt ?? 0).getTime()).slice(0, 20);
  const loading = !!token && snapshot.isPending;
  const [clearing, setClearing] = useState(false);
  const previousRefresh = useRef(refreshTrigger);
  useEffect(() => {
    if (previousRefresh.current !== refreshTrigger) {
      previousRefresh.current = refreshTrigger;
      void queryManager.keys(queryClient, options.queryKey);
    }
  }, [refreshTrigger, options.queryKey]);

  const handleClearAll = async () => {
    if (!token || covers.length === 0) return;
    if (!window.confirm(`Delete all ${covers.length} covers?`)) return;
    setClearing(true);
    try {
      await deleteCovers(covers.map(c => c.id));
    } finally {
      setClearing(false);
    }
  };

  const handlePlay = (cover: Song) => {
    // The current track pauses/resumes in place; reloading it would restart it.
    if (cover.id === currentTrackId) { togglePlay(); return; }
    playFromList(songToTrack(cover), covers.map(songToTrack), 'cover-studio');
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
          <Music className="w-4 h-4 text-cyan-400" />
          Recent Covers
          {covers.length > 0 && <span className="text-[10px] text-zinc-500 font-normal">({covers.length})</span>}
        </div>
        {covers.length > 0 && (
          <button
            onClick={handleClearAll}
            disabled={clearing}
            className="p-1 rounded-md hover:bg-red-900/30 text-zinc-500 hover:text-red-400 transition-colors"
            title="Clear all covers"
          >
            {clearing ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />}
          </button>
        )}
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-4">
          <Loader2 className="w-4 h-4 text-zinc-500 animate-spin" />
        </div>
      ) : covers.length === 0 ? (
        <p className="text-[10px] text-zinc-500 text-center py-4">No covers yet. Generate your first one!</p>
      ) : (
        <div className="space-y-1">
          {covers.map(cover => {
            const isCurrent = currentTrackId === cover.id;
            return (
              <div
                key={cover.id}
                onClick={() => handlePlay(cover)}
                className={`
                  w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all duration-200 text-left cursor-pointer group relative
                  ${isCurrent ? 'bg-cyan-500/20 ring-1 ring-cyan-400/50' : 'hover:bg-white/5'}
                `}
              >
                <div className="w-6 h-6 rounded-full bg-cyan-500/20 flex items-center justify-center flex-shrink-0">
                  {isCurrent && isPlaying ? (
                    <Pause className="w-3 h-3 text-cyan-400" />
                  ) : (
                    <Play className="w-3 h-3 text-cyan-400" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-medium text-zinc-700 dark:text-zinc-300 truncate">{cover.title}</div>
                  <div className="text-[10px] text-zinc-500">{cover.duration}</div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
