import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';
import { referencesOptions, vramOptions } from './options';
import { queryManager } from '../invalidation';

export const useVram = () => useQuery(vramOptions());
export function useReferences() {
  const client = useQueryClient();
  const query = useQuery(referencesOptions());
  const refresh = useCallback(() => queryManager.keys(client, referencesOptions().queryKey), [client]);
  return { references: query.data?.references ?? [], refresh };
}
