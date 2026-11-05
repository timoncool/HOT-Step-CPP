import { useQuery } from '@tanstack/react-query';
import { useAuth } from '../../context/AuthContext';
import { songsOptions } from '../shell-queue/songs';

// Shared reads retain their canonical wire shapes. The transport injects the
// bearer from the session accessor; observers own selection and polling cadence.
//
// Each resource has one owner; this slice only re-exports the factory so studio
// consumers keep their existing import surface.
export { songsOptions } from '../shell-queue/songs';
export { pluginsOptions as studioPluginsOptions } from '../global-models/options';

export function useStudioSongs(enabled = true) {
  const { token, sessionEpoch } = useAuth();
  void sessionEpoch; // re-render on login/logout so the key's epoch stays current
  return useQuery({ ...songsOptions(), enabled: enabled && !!token });
}

export { aceModelsOptions as modelsOptions } from '../global-models/options';
