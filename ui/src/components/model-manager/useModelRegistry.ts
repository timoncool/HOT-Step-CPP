import { useQuery, useQueryClient } from '@tanstack/react-query';
import { registryOptions } from '../../queries/global-models/options';
import { queryManager } from '../../queries/invalidation';
// useModelRegistry.ts — Hook for fetching and querying the model registry

import { useCallback, useMemo } from 'react';
import type { RegistryFile } from '../../types';

export interface PackStatus {
  total: number;
  installed: number;
  missing: number;
  totalSize: number;
  remainingSize: number;
}

export function useModelRegistry() {
  const client = useQueryClient();
  const query = useQuery(registryOptions());
  const registry = query.data ?? null;
  const loading = query.isPending;
  const error = query.error?.message ?? null;
  const fetchRegistry = useCallback(async (_silent = false) => {
    await queryManager.keys(client, registryOptions().queryKey);
  }, [client]);
  const silentRefresh = useCallback(() => fetchRegistry(true), [fetchRegistry]);

  const installedFiles = useMemo(() => {
    if (!registry) return new Set<string>();
    return new Set(registry.files.filter(f => f.installed).map(f => f.filename));
  }, [registry]);

  const getFile = useCallback((id: string): RegistryFile | undefined => {
    return registry?.files.find(f => f.id === id);
  }, [registry]);

  const getPackFiles = useCallback((packId: string): RegistryFile[] => {
    const pack = registry?.packs.find(p => p.id === packId);
    if (!pack || !registry) return [];
    // fileIds is data from model-registry.json, so a hand-edited catalogue can
    // ship a pack without it. That used to throw on undefined.map during
    // render and take the whole Model Manager down with it, which is a very
    // large blast radius for one bad entry (issue #120). One empty pack is a
    // better failure than a blank page.
    if (!Array.isArray(pack.fileIds)) {
      console.warn(`[ModelRegistry] Pack "${packId}" has no fileIds array — nothing to download.`);
      return [];
    }
    return pack.fileIds.map(id => registry.files.find(f => f.id === id)).filter(Boolean) as RegistryFile[];
  }, [registry]);

  const packStatus = useCallback((packId: string): PackStatus => {
    const files = getPackFiles(packId);
    const installed = files.filter(f => f.installed).length;
    return {
      total: files.length,
      installed,
      missing: files.length - installed,
      totalSize: files.reduce((a, f) => a + f.sizeBytes, 0),
      remainingSize: files.filter(f => !f.installed).reduce((a, f) => a + f.sizeBytes, 0),
    };
  }, [getPackFiles]);

  return {
    registry,
    loading,
    error,
    refresh: fetchRegistry,
    silentRefresh,
    installedFiles,
    getFile,
    getPackFiles,
    packStatus,
  };
}
