/**
 * NoModelsBanner.tsx — first run with an empty models folder.
 *
 * The engine exits at its model scan when nothing is installed. That is not a
 * crash and not a missing DLL, so say what is actually needed and open the Model
 * Manager. The server restarts the engine by itself when a download finishes or a
 * model file appears in the models folder. The shared health query is polled
 * (Query owns the timer and its cancellation) for as long as the app is open,
 * because ready flips true before the engine's model scan decides otherwise.
 */

import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { PackageOpen } from 'lucide-react';
import { noModelsRefetchInterval } from './noModelsPoll';
import { healthOptions } from '../../queries/shell-queue/settings';
import { ModelManagerModal } from '../model-manager/ModelManagerModal';

export const NoModelsBanner: React.FC = () => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const { data } = useQuery({
    ...healthOptions(),
    refetchInterval: noModelsRefetchInterval,
  });

  if (data?.engine?.noModels !== true) return null;

  return (
    <>
      <div className="flex items-center gap-3 px-4 py-2 border-b border-amber-500/30 bg-amber-500/10">
        <PackageOpen size={18} className="flex-shrink-0 text-amber-400" />
        <div className="min-w-0 flex-1 text-sm text-zinc-800 dark:text-zinc-100">
          {t('noModels.message')}
        </div>
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="flex-shrink-0 px-3 py-1.5 rounded-lg text-xs font-semibold
                     bg-amber-500/20 text-amber-300 hover:bg-amber-500/30 transition-colors"
        >
          {t('noModels.open')}
        </button>
      </div>
      {open && <ModelManagerModal onClose={() => setOpen(false)} />}
    </>
  );
};
