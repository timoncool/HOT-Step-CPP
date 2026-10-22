import { queryOptions } from '@tanstack/react-query';
import { serverKey } from '../keys';
import { getProviders } from '../../services/assistantApi';
import { fetchInspireProviders, fetchInstagenPrompt } from '../../services/inspireApi';
import { listJobs, getExtractResult, getExtractProgress } from '../../services/stemStudioApi';
import { getProgress, getStemList } from '../../services/supersepApi';
import { getStreamSession } from '../../services/streamSessionsApi';
import { builderApi } from '../../services/api';

export const assistantProvidersOptions = () => queryOptions({
  queryKey: serverKey('studios/assistant-providers'), queryFn: ({ signal }) => getProviders({ signal }),
});
export const inspireProvidersOptions = () => queryOptions({
  queryKey: serverKey('studios/inspire-providers'), queryFn: ({ signal }) => fetchInspireProviders({ signal }),
});
export const inspirePromptOptions = () => queryOptions({
  queryKey: serverKey('studios/inspire-prompt'), queryFn: ({ signal }) => fetchInstagenPrompt({ signal }),
});
export const extractionsOptions = () => queryOptions({
  queryKey: serverKey('studios/extractions'), queryFn: ({ signal }) => listJobs({ signal }),
});
export const extractResultOptions = (id: string) => queryOptions({
  queryKey: serverKey('studios/extract-result', { id }), queryFn: ({ signal }) => getExtractResult(id, { signal }),
});
export const extractProgressOptions = (id: string) => queryOptions({
  queryKey: serverKey('studios/extract-progress', { id }), queryFn: ({ signal }) => getExtractProgress(id, { signal }),
  staleTime: 0,
});
export const separationProgressOptions = (id: string) => queryOptions({
  queryKey: serverKey('studios/separation-progress', { id }), queryFn: ({ signal }) => getProgress(id, { signal }),
  staleTime: 0,
});
export const separationResultOptions = (id: string) => queryOptions({
  queryKey: serverKey('studios/separation-result', { id }), queryFn: ({ signal }) => getStemList(id, { signal }),
});
export const streamSessionOptions = (id: string) => queryOptions({
  queryKey: serverKey('studios/stream-session', { id }), queryFn: ({ signal }) => getStreamSession(id, { signal }),
});
export const projectsOptions = () => queryOptions({
  queryKey: serverKey('studios/projects', { auth: true }), queryFn: ({ signal }) => builderApi.listProjects({ signal }),
});
export const projectOptions = (id: string) => queryOptions({
  queryKey: serverKey('studios/project', { auth: true, id }), queryFn: ({ signal }) => builderApi.getProject(id, { signal }),
  staleTime: 0,
});
