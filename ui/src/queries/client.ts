// client.ts — the one QueryClient for the app.
//
// main.tsx mounts it above AuthProvider and App. It survives Vite HMR through
// import.meta.hot.data, so editing a module never builds a second cache with
// its own observers and polls.

import { QueryClient } from '@tanstack/react-query';
import { QUERY_GC_MS, QUERY_STALE_MS, retryDelay, retryQuery } from './contract';

/** A client with the project defaults (contract.ts). Tests build their own. */
export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: QUERY_STALE_MS,
        gcTime: QUERY_GC_MS,
        refetchOnWindowFocus: false,
        refetchOnReconnect: true,
        retry: retryQuery,
        retryDelay,
      },
      // A submit is never repeated behind the user's back.
      mutations: { retry: false },
    },
  });
}

export const queryClient: QueryClient =
  (import.meta.hot?.data.queryClient as QueryClient | undefined) ?? createQueryClient();
if (import.meta.hot) import.meta.hot.data.queryClient = queryClient;
