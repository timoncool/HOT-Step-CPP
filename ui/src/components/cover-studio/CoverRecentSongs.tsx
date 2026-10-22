/**
 * CoverRecentSongs.tsx — Shows recently generated covers in Cover Studio.
 *
 * Fetches songs with source=cover-studio, provides play/download/delete.
 * Download uses cover-specific filename: Target Artist - Track Name (Source Artist Cover)
 */

import { useQuery } from '@tanstack/react-query';
import { coverSongsOptions, deleteCovers } from '../../queries/cover-lyrics/songs';
import { queryClient } from '../../queries/client';
import { queryManager } from '../../queries/invalidation';
import React, { useEffect, useCallback, useRef } from 'react';
import { Play, Loader2, Music, ListPlus, Check } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import type { Song } from '../../types';
import { downloadTrack } from '../../utils/downloadTrack';
import { usePlaylist } from '../lyric-studio/playlistStore';
import { playFromList, togglePlay, songToTrack, usePlaybackSelector } from '../../stores/playbackStore';
import { useDisguiseMode } from '../../hooks/useDisguiseMode';
import { SongActionsMenu } from '../shared/SongActionsMenu';

interface CoverRecentSongsProps {
  showToast: (msg: string, type?: 'success' | 'error') => void;
  refreshKey?: number;
}


export const CoverRecentSongs: React.FC<CoverRecentSongsProps> = ({ showToast, refreshKey = 0 }) => {
  const { token } = useAuth();
  const currentTrackId = usePlaybackSelector(s => s.currentTrack?.id ?? null);
  const options = coverSongsOptions();
  const snapshot = useQuery({ ...options, enabled: !!token });
  const songs = (snapshot.data?.songs ?? []).filter(s => !!s.audioUrl).slice().sort((a, b) =>
    new Date(b.createdAt ?? 0).getTime() - new Date(a.createdAt ?? 0).getTime()).slice(0, 50);
  const loading = !!token && snapshot.isPending;
  const { disguiseArtist, disguiseTitle } = useDisguiseMode();
  const previousRefresh = useRef(refreshKey);
  useEffect(() => {
    if (previousRefresh.current !== refreshKey) {
      previousRefresh.current = refreshKey;
      void queryManager.keys(queryClient, options.queryKey);
    }
  }, [refreshKey, options.queryKey]);
  useEffect(() => {
    const refresh = () => { void queryManager.keys(queryClient, options.queryKey); };
    window.addEventListener('song-postprocessed', refresh);
    window.addEventListener('song-postprocess-reverted', refresh);
    return () => {
      window.removeEventListener('song-postprocessed', refresh);
      window.removeEventListener('song-postprocess-reverted', refresh);
    };
  }, [token]);

  const handlePlay = useCallback((song: Song) => {
    // The current track pauses/resumes in place; reloading it would restart it.
    if (song.id === currentTrackId) { togglePlay(); return; }
    playFromList(songToTrack(song), songs.map(songToTrack), 'cover-studio');
  }, [songs, currentTrackId]);

  const handleDelete = useCallback(async (song: Song) => {
    if (!token) return;
    try {
      await deleteCovers([song.id]);
      showToast('Cover deleted');
    } catch {
      showToast('Failed to delete', 'error');
    }
  }, [token, showToast]);

  const handleDownload = useCallback((song: Song) => {
    const gp = song.generationParams as any;
    const targetArtist = gp?.artistName || song.artistName || '';
    downloadTrack(song, { artistName: targetArtist });
  }, []);

  if (loading && songs.length === 0) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="w-4 h-4 text-zinc-500 animate-spin" />
      </div>
    );
  }

  if (songs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-center px-4">
        <Music className="w-5 h-5 text-zinc-600 mb-2" />
        <p className="text-xs text-zinc-500">No covers yet</p>
      </div>
    );
  }

  return (
    <>
      <div className="grid grid-cols-1 auto-rows-[4.5rem] gap-1 px-2 py-1.5 overflow-y-auto scrollbar-hide" style={{ maxHeight: '100%' }}>
        {songs.map((song) => {
          const dur = typeof song.duration === 'number' ? song.duration : 0;
          const mins = Math.floor(dur / 60);
          const secs = String(Math.floor(dur % 60)).padStart(2, '0');
          const isCurrent = currentTrackId === song.id;
          const gp = song.generationParams as any;
          const targetArtist = gp?.artistName || song.artistName || '';

          return (
            <div key={song.id}
              className={`flex items-center gap-2.5 rounded-lg hover:bg-white/[0.06] transition-colors text-left group px-2 overflow-hidden relative cursor-pointer ${isCurrent ? 'bg-cyan-500/10 ring-1 ring-cyan-500/30' : ''}`}
              onClick={() => handlePlay(song)}>
              <div className="w-14 h-14 rounded-md flex-shrink-0 overflow-hidden bg-zinc-100 dark:bg-zinc-800 relative">
                <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-cyan-900/40 to-purple-900/40">
                  <Music className="w-5 h-5 text-cyan-500/60" />
                </div>
                <div className="absolute inset-0 bg-black/20 dark:bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                  <Play className="w-4 h-4 text-white ml-0.5" />
                </div>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-zinc-800 dark:text-zinc-200 truncate leading-snug">
                  {disguiseTitle(song.title || 'Untitled Cover')}
                </p>
                {targetArtist && (
                  <p className="text-[10px] text-zinc-500 truncate leading-snug">{disguiseArtist(targetArtist)}</p>
                )}
                {dur > 0 && (
                  <p className="text-[10px] text-zinc-600 font-mono mt-0.5">{mins}:{secs}</p>
                )}
              </div>
              <div className="absolute right-1 top-1/2 -translate-y-1/2 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                <CoverAddToPlaylistBtn song={song} />
                <SongActionsMenu
                  song={song}
                  size={14}
                  className="bg-zinc-100/80 dark:bg-zinc-800/80 rounded-md"
                  onDownload={() => handleDownload(song)}
                  onDelete={() => handleDelete(song)}
                />
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
};

// ── Add-to-playlist helper ───────────────────────────────────────────────────

const CoverAddToPlaylistBtn: React.FC<{ song: Song }> = ({ song }) => {
  const playlist = usePlaylist();
  const inPlaylist = playlist.isIn(song.id);
  const gp = song.generationParams as any;
  const targetArtist = gp?.artistName || song.artistName || '';

  const toggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (inPlaylist) {
      playlist.remove(song.id);
    } else {
      playlist.add({
        id: song.id,
        title: song.title || 'Untitled Cover',
        audioUrl: song.audioUrl || '',
        masteredAudioUrl: song.masteredAudioUrl || '',
        noAdapterAudioUrl: song.noAdapterAudioUrl || '',
        artistName: targetArtist,
        coverUrl: '',
        duration: typeof song.duration === 'number' ? song.duration : 0,
      });
    }
  };

  return (
    <button onClick={toggle}
      className={`p-1.5 rounded-md transition-colors ${
        inPlaylist ? 'bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30'
          : 'bg-zinc-100/80 dark:bg-zinc-800/80 hover:bg-zinc-300 dark:hover:bg-zinc-700 text-zinc-600 dark:text-zinc-400 hover:text-cyan-400'
      }`}
      title={inPlaylist ? 'Remove from playlist' : 'Add to playlist'}>
      {inPlaylist ? <Check className="w-3 h-3" /> : <ListPlus className="w-3 h-3" />}
    </button>
  );
};
