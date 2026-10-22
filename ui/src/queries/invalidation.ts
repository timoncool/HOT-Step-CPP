// invalidation.ts — the one global queryManager: the only place in the app
// that invalidates cache entries (P2.4, docs/dev/tanstack-query.md).
//
// Every refetch-after-write goes through this object: store writes declare
// what they dirtied via writeAndInvalidate (writes.ts), the SSE bridge flushes
// its coalesced keys here, and refresh buttons call a named group below. No
// other file calls QueryClient.invalidateQueries — grepping for it finds
// exactly this one (tests excepted).
//
// P2 shape: the groups wrap today's per-area key lists; when P6 folds each
// entity into its own composer folder, a group closes over that composer and
// the old list/helper is deleted. Parameterized invalidations (a write knows
// its worker/id) call `keys`/`filter` with the captured scope — the manager
// still owns the actual invalidateQueries call.

import type { InvalidateQueryFilters, QueryClient, QueryKey } from '@tanstack/react-query';
import { resourceKey, SHARED, SONG_VIEWS } from './keys';

/** The one primitive: the only QueryClient.invalidateQueries call in the app. */
export function invalidateKeys(client: QueryClient, keys: QueryKey[]): Promise<unknown> {
  return Promise.all(keys.map(key => client.invalidateQueries({ queryKey: key })));
}

/** For filters that need more than a key prefix (predicates, type narrowing). */
export function invalidateQuery(client: QueryClient, filter: InvalidateQueryFilters): Promise<unknown> {
  return client.invalidateQueries(filter);
}

export const queryManager = {
  /** Every cached view of song rows, across areas (keys.ts's SONG_VIEWS). A
   *  song write, delete, completion or reconnect dirties all of them. */
  songs: {
    views: (c: QueryClient) => invalidateKeys(c, SONG_VIEWS),
  },
  /** A track's lyric sidecars, after a crop or retranscription rewrote them. */
  sidecars: {
    all: (c: QueryClient) => invalidateKeys(c, [resourceKey(SHARED.lrc), resourceKey(SHARED.lyricsJson)]),
  },
  /** Exact keys or prefixes, with the captured scope — parameterized writes
   *  and refresh buttons. */
  keys: (c: QueryClient, ...keys: QueryKey[]) => invalidateKeys(c, keys),
  /** A full filter (predicate-narrowed sets, e.g. one worker's reads). */
  filter: (c: QueryClient, filter: InvalidateQueryFilters) => invalidateQuery(c, filter),

  /** Event map: writes declare what happened; the manager resolves it to the
   *  groups above. Parameterized methods are called directly when the write
   *  knows the id (e.g. a patch). */
  events: {
    'song:written': (c: QueryClient) => queryManager.songs.views(c),
    'sidecars:changed': (c: QueryClient) => queryManager.sidecars.all(c),
  },
};

export type EventRef = keyof typeof queryManager.events;

/** Resolve a list of named events to their invalidations. */
export function invalidateEvents(client: QueryClient, events: readonly EventRef[]): Promise<unknown> {
  return Promise.all(events.map(e => queryManager.events[e](client)));
}

/** P2.4 transitional re-exports for keys.ts's old per-area helpers; P6 deletes
 *  them once every caller moves to the groups above. */
export const invalidateSongViews = (c: QueryClient) => queryManager.songs.views(c);
export const invalidateSidecars = (c: QueryClient) => queryManager.sidecars.all(c);
