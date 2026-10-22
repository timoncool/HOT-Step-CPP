import { queryOptions } from '@tanstack/react-query';
import { fetchLrc } from '../../utils/lrcUtils';
import { sidecarUrl } from '../../utils/wordLrcUtils';
import { sharedKeys } from '../keys';

export const lyricsOptions = (audioUrl: string) => queryOptions({
  queryKey: sharedKeys.lrc(sidecarUrl(audioUrl, '.lrc')),
  queryFn: ({ signal }) => fetchLrc(audioUrl, { signal }),
  enabled: !!audioUrl,
});
