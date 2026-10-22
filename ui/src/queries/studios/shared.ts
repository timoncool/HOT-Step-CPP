import { queryOptions, useQuery } from '@tanstack/react-query';
import { useAuth } from '../../context/AuthContext';
import { songApi } from '../../services/api';
import { apiClient } from '../../services/httpClient';
import type { PluginRegistry } from '../../types/pluginTypes';
import { sharedKeys } from '../keys';

// Shared reads retain their canonical wire shapes. The transport injects the
// bearer from the session accessor; observers own selection and polling cadence.
export const songsOptions = () => queryOptions({
  queryKey: sharedKeys.songs(),
  queryFn: ({ signal }) => songApi.list({ signal }),
});
export function useStudioSongs(enabled = true) {
  const { token, sessionEpoch } = useAuth();
  void sessionEpoch; // re-render on login/logout so the key's epoch stays current
  return useQuery({ ...songsOptions(), enabled: enabled && !!token });
}
export const studioPluginsOptions = () => queryOptions({
  queryKey: sharedKeys.plugins(),
  queryFn: ({ signal }) => apiClient.get<PluginRegistry>('/plugins', { signal }),
});
export { aceModelsOptions as modelsOptions } from '../global-models/options';
