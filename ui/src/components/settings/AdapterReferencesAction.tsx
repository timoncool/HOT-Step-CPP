import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { lireekApi } from '../../services/lireekApi';

export function AdapterReferencesAction() {
  const { t } = useTranslation();
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<Awaited<ReturnType<typeof lireekApi.updateAdapterReferences>> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const update = async () => {
    if (running) return;
    setRunning(true);
    setResult(null);
    setError(null);
    try { setResult(await lireekApi.updateAdapterReferences()); }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setRunning(false); }
  };
  return (
    <div className="px-4 py-3 space-y-2">
      <p className="setting-description">{t('settings.adapters.updateInfo')}</p>
      <button className="placement-button" disabled={running} onClick={update}>
        {t(running ? 'settings.adapters.updating' : 'settings.adapters.update')}
      </button>
      <div role="status" aria-live="polite" className="text-sm text-zinc-700 dark:text-zinc-300">
        {error && <p className="text-red-600 dark:text-red-400">{t('settings.adapters.failed', { error })}</p>}
        {result && <>
          <p>{t('settings.adapters.result', { scanned: result.scanned, updated: result.updated, count: result.changes.length })}</p>
          {result.changes.length > 0 && <ul className="mt-2 space-y-1 break-all">
            {result.changes.map(change => <li key={`${change.lyricsSetId}:${change.field}`}>
              {t('settings.adapters.dropped', { preset: change.presetName, adapter: change.adapter })}
            </li>)}
          </ul>}
        </>}
      </div>
    </div>
  );
}
