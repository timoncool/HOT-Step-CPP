import { apiClient, type ReadOptions } from './httpClient';

export type DataFolderOperation = 'move' | 'repoint' | 'create' | 'import';
export interface DataFolderPending {
  kind: DataFolderOperation | 'migrate';
  source: string | null;
  destination: string;
}
export interface DataFolderLocation {
  mode: 'dev' | 'release';
  appRoot: string | null;
  dataRoot: string | null;
  override: string | null;
  pending: DataFolderPending | null;
  externals: Array<{ key: string; path: string }>;
  blockers: string[];
}
export interface DataFolderPreview {
  ok: boolean;
  errors: string[];
  kind: DataFolderOperation;
  target: string;
  mappings: Array<{ name: string; from: string; to: string }>;
  bytes: number;
  crossVolume: boolean;
  needBytes: number;
  freeBytes: number | null;
  blockers: string[];
}
async function request<T>(suffix = '', body?: unknown, opts: ReadOptions = {}): Promise<T> {
  return body === undefined
    ? apiClient.get<T>(`/settings/data-folder${suffix}`, opts)
    : apiClient.post<T>(`/settings/data-folder${suffix}`, body, opts);
}
export const dataFolderApi = {
  location: (opts?: ReadOptions) => request<DataFolderLocation>('', undefined, opts),
  preview: (kind: DataFolderOperation, target: string) => request<DataFolderPreview>('/preview', { kind, target }),
  schedule: (kind: DataFolderOperation, target: string) => request<{ pending: DataFolderPending; restartRequested: boolean }>('/schedule', { kind, target }),
  cancel: () => request<{ cancelled: boolean }>('/cancel', {}),
};
