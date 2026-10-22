import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { dataFolderApi, type DataFolderLocation, type DataFolderOperation, type DataFolderPreview } from '../../services/dataFolderApi';
import { ParamLabel } from '../shared/ParamLabel';
import { dataFolderOptions } from '../../queries/shell-queue/settings';
import { StyledSelect } from '../shared/StyledSelect';

export function DataFolderPanel() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const locationQuery = useQuery(dataFolderOptions());
  const location: DataFolderLocation | null = locationQuery.data ?? null;
  const [kind, setKind] = useState<DataFolderOperation>('move');
  const [target, setTarget] = useState('');
  const [preview, setPreview] = useState<DataFolderPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const revision = useRef(0);
  const reload = () => queryClient.refetchQueries({ queryKey: dataFolderOptions().queryKey });
  useEffect(() => { if (locationQuery.error) setError(locationQuery.error.message); }, [locationQuery.error]);
  const change = () => { revision.current++; setPreview(null); setError(''); };
  const run = async (action: () => Promise<void>) => {
    setBusy(true); setError('');
    try { await action(); } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      await reload().catch(() => {});
    } finally { setBusy(false); }
  };
  const bytes = (value: number) => `${(value / 1e9).toFixed(2)} GB`;
  const blocked = busy || !location || !!location.override || !!location.pending;
  return <section className="settings-section">
    <div className="settings-section-header"><span className="settings-section-title">{t('settings.dataFolder.title')}</span></div>
    <div className="space-y-3 text-sm text-zinc-700 dark:text-zinc-300">
      {error && <p role="alert" className="text-red-600 dark:text-red-400">{error}</p>}
      {!location && !error && <p>{t('settings.dataFolder.loading')}</p>}
      {location && <>
        <ParamLabel label={t('settings.dataFolder.current')} info={t('settings.dataFolder.currentInfo')} />
        <p className="break-all">{location.dataRoot || t('settings.dataFolder.dev')}</p>
        {location.override && <p className="break-all">{t('settings.dataFolder.override', { path: location.override })}</p>}
        {location.externals.length > 0 && <div>
          <ParamLabel label={t('settings.dataFolder.externals')} info={t('settings.dataFolder.externalsInfo')} />
          <ul>{location.externals.map(item => <li className="break-all" key={item.key}>{item.key}: {item.path}</li>)}</ul>
        </div>}
        {location.blockers.length > 0 && <div><p>{t('settings.dataFolder.wait')}</p><ul>{location.blockers.map((item, i) => <li key={i}>{item}</li>)}</ul></div>}
        {location.pending && <div className="space-y-2">
          <p>{t('settings.dataFolder.pending', { action: t(`settings.dataFolder.${location.pending.kind}`) })}</p>
          <p className="break-all">{location.pending.source || location.dataRoot} → {location.pending.destination}</p>
          <button className="settings-save-btn" disabled={busy} onClick={() => void run(async () => {
            const response = await fetch('/api/shutdown/restart', { method: 'POST' });
            if (!response.ok) throw new Error(t('settings.dataFolder.restartFailed'));
            window.dispatchEvent(new CustomEvent('hotstep:restarting'));
          })}>{t('settings.dataFolder.restart')}</button>
          <button className="settings-save-btn" disabled={busy} onClick={() => void run(async () => { await dataFolderApi.cancel(); await reload(); })}>{t('settings.dataFolder.cancel')}</button>
        </div>}
        <ParamLabel label={t('settings.dataFolder.action')} info={t('settings.dataFolder.actionInfo')} />
        <StyledSelect value={kind} disabled={blocked} accent="emerald" onChange={value => { change(); setKind(value as DataFolderOperation); }} options={(['move', 'repoint', 'create', 'import'] as const).map(value => ({ value, label: t(`settings.dataFolder.${value}`) }))} />
        <p>{t(`settings.dataFolder.${kind}Info`)}</p>
        <label className="flex flex-col gap-1">
          <ParamLabel label={t(kind === 'import' ? 'settings.dataFolder.previousInstall' : 'settings.dataFolder.target')} info={t('settings.dataFolder.targetInfo')} />
          <input className="w-full rounded border border-zinc-300 bg-white p-2 dark:border-zinc-700 dark:bg-zinc-900" disabled={blocked} value={target} onChange={e => { change(); setTarget(e.target.value); }} />
        </label>
        <button className="settings-save-btn" disabled={blocked || !target.trim()} onClick={() => void run(async () => {
          const currentRevision = revision.current;
          const result = await dataFolderApi.preview(kind, target.trim());
          if (currentRevision === revision.current) setPreview(result);
          await reload();
        })}>{t('settings.dataFolder.preview')}</button>
        {preview && <div className="space-y-2">
          <p className="break-all">{t('settings.dataFolder.current')}: {location.dataRoot || t('settings.dataFolder.dev')}</p>
          <p className="break-all">{t(kind === 'import' ? 'settings.dataFolder.previousInstall' : 'settings.dataFolder.target')}: {preview.target}</p>
          <p>{t('settings.dataFolder.size', { size: bytes(preview.bytes) })}</p>
          {preview.crossVolume && <p>{t('settings.dataFolder.space', { needed: bytes(preview.needBytes), free: preview.freeBytes === null ? t('settings.dataFolder.unknown') : bytes(preview.freeBytes) })}</p>}
          {preview.mappings.map(item => <p key={item.name} className="break-all">{item.name}: {item.from} → {item.to}</p>)}
          {preview.errors.map((item, i) => <p role="alert" key={i} className="text-red-600 dark:text-red-400">{item}</p>)}
          {preview.blockers.map((item, i) => <p key={i}>{item}</p>)}
          <button className="settings-save-btn" disabled={blocked || !preview.ok || preview.blockers.length > 0 || location.blockers.length > 0} onClick={() => void run(async () => {
            await dataFolderApi.schedule(preview.kind, preview.target);
            window.dispatchEvent(new CustomEvent('hotstep:restarting'));
          })}>{t('settings.dataFolder.confirm')}</button>
        </div>}
      </>}
    </div>
  </section>;
}
