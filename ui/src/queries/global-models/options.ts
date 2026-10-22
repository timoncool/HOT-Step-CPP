import { queryOptions, type QueryClient } from '@tanstack/react-query';
import { ApiClient } from '../../services/httpClient';
import { adapterApi } from '../../services/api';
import { serverKey, sharedKeys, scopeOf, resourceKey, type ServerKey } from '../keys';
import { pollWhile } from '../contract';
import { queryManager } from '../invalidation';
import type { PluginRegistry } from '../../types/pluginTypes';
import type { ModelRegistry, AceModels } from '../../types';
import type { BackendCapabilities, BackendInfo, BackendModelCatalogue } from '../../stores/backendStore';

export const adapterScanOptions = (folder: string) => queryOptions({
  queryKey: sharedKeys.adapterScan(folder),
  queryFn: ({ signal }) => adapterApi.scan(folder, { signal }),
  staleTime: 0,
});

// Each request uses the API origin and backend captured in its key.
function read<T>(key: ServerKey, path: string) {
  return queryOptions({
    queryKey: key,
    queryFn: ({ queryKey, signal }) => new ApiClient({ baseUrl: scopeOf(queryKey)!.api }).get<T>(path, { signal }),
  });
}

export function localOptions<T>(name: string, path: string, filters = {}) {
  return read<T>(serverKey(`global-models/${name}`, { filters }), path);
}

export const pluginsOptions = () => read<PluginRegistry>(sharedKeys.plugins(), '/plugins');
export const registryOptions = () => localOptions<ModelRegistry>('registry', '/model-manager/registry');
export const backendsOptions = () => localOptions<{ backends: BackendInfo[]; activeId: string }>('backends', '/backends');
export const backendModelsOptions = (backend: string) => read<BackendModelCatalogue>(
  serverKey('global-models/backend-models', { backend }), `/backends/models?backend=${encodeURIComponent(backend)}`,
);
export const capabilitiesOptions = (backend: string) => ({
  ...read<BackendCapabilities>(sharedKeys.capabilities(backend), `/capabilities?backend=${encodeURIComponent(backend)}`),
  refetchInterval: pollWhile<BackendCapabilities>(data => !data?.up, 10_000),
});
export function usableCapabilities(previous: BackendCapabilities | undefined, next: BackendCapabilities) {
  return next.up === false && previous?.up ? previous : next;
}
export interface VramInfo { used_mb: number; total_mb: number; free_mb: number }
export interface LoadedModel { label: string; mb: number; in_use: boolean }
export const vramOptions = () => ({
  ...read<VramInfo>(sharedKeys.vram(), '/logs/vram'),
  refetchInterval: pollWhile<VramInfo>(() => true),
});
export const loadedModelsOptions = () => ({
  ...read<{ loaded: LoadedModel[] }>(sharedKeys.loadedModels(), '/logs/models-loaded'),
  refetchInterval: pollWhile<{ loaded: LoadedModel[] }>(() => true, 2000),
});
export const aceModelsOptions = () => ({
  ...localOptions<AceModels>('ace-models', '/models'),
  refetchInterval: pollWhile<AceModels>(data => !data?.models?.dit?.length, 3000),
});
export interface ReferenceTrack { name: string; size: number; url: string }
export const referencesOptions = () => localOptions<{ references: ReferenceTrack[] }>('references', '/mastering/references');
export interface MonitorStatus { running: boolean; paused: boolean; position: number; duration: number }
export const monitorOptions = () => ({
  ...localOptions<MonitorStatus>('monitor', '/vst/monitor/status'),
  staleTime: 0,
  refetchInterval: (query: Parameters<ReturnType<typeof pollWhile<MonitorStatus>>>[0]) =>
    pollWhile<MonitorStatus>(() => true, query.state.data?.running ? 300 : 2000)(query),
});

// File installation affects every backend and each availability probe. Prefixes
// include inactive backend caches so returning to a backend cannot reuse them.
export const modelChangeKeys = () => [
  resourceKey('global-models/registry'), resourceKey('global-models/backend-models'),
  resourceKey('global-models/ace-models'), resourceKey('capabilities'),
  resourceKey('loaded-models'), resourceKey('vram'),
  resourceKey('global-models/pp-vae'), resourceKey('global-models/stablestep'),
  resourceKey('global-models/sa3-adapters'), resourceKey('global-models/mm3-adapters'),
];
export async function invalidateModels(client: QueryClient) {
  await queryManager.keys(client, ...modelChangeKeys());
}
