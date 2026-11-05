import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { updatesApi, type UpdateChannel, type UpdatesStatus } from '../../services/updatesApi';
import { updatesStatusOptions } from '../../queries/shell-queue/settings';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';

const short = (sha: string | null) => (sha ? sha.slice(0, 8) : null);

/** Plain-language notice for states that can't be compared; null when nothing special applies. */
function noticeKey(s: UpdatesStatus): string | null {
  if (s.installed.git?.dirty) return 'dirty';
  if (s.installed.kind === 'unknown' || (s.installed.kind === 'portable' && !s.installed.tag && !s.installed.sha)) return 'unknownIdentity';
  if (s.status === 'unknown') return 'cannotCompare';
  return null;
}

export function UpdatesPanel() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const query = useQuery({
    ...updatesStatusOptions(),
    // A check started elsewhere (startup, another tab) reports checking:true; follow it until it ends.
    refetchInterval: q => (q.state.data?.checking ? 2000 : false),
  });
  const check = useMutation({
    mutationFn: (channel?: UpdateChannel) => updatesApi.check(channel),
    onSuccess: result => queryClient.setQueryData(updatesStatusOptions().queryKey, result),
  });
  const s = query.data;
  const busy = check.isPending || !!s?.checking;
  const error = check.error?.message ?? query.error?.message ?? '';
  const when = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : t('settings.updates.never'));
  const notice = s ? noticeKey(s) : null;
  const latest = s?.channel === 'release' ? s.latest.release?.tag : short(s?.latest.master?.sha ?? null);
  const installedLabel = s && [s.installed.tag ?? short(s.installed.sha) ?? t('settings.updates.unknown'), s.installed.variant].filter(Boolean).join(' · ');

  return <section className="settings-section">
    <div className="settings-section-header"><span className="settings-section-title">{t('settings.updates.title')}</span></div>
    <div className="space-y-3 text-sm text-zinc-700 dark:text-zinc-300">
      {error && <p role="alert" className="text-red-600 dark:text-red-400">{error}</p>}
      {!s && !error && <p>{t('settings.updates.loading')}</p>}
      {s && <>
        <ParamLabel label={t('settings.updates.installed')} info={t('settings.updates.installedInfo')} />
        <p className="break-all">{installedLabel}</p>
        <ParamLabel label={t('settings.updates.channel')} info={t('settings.updates.channelInfo')} />
        <StyledSelect value={s.channel} disabled={busy} accent="emerald"
          onChange={value => check.mutate(value as UpdateChannel)}
          options={(['release', 'master'] as const).map(value => ({ value, label: t(`settings.updates.${value}`) }))} />
        <p>{t('settings.updates.lastCheck', { time: when(s.lastSuccessAt) })}</p>
        <button className="settings-save-btn" disabled={busy} onClick={() => check.mutate(undefined)}>
          {busy ? t('settings.updates.checking') : t('settings.updates.checkNow')}
        </button>
        {check.data?.throttled && <p>{t('settings.updates.throttled')}</p>}
        {s.retryAfter && <p>{t('settings.updates.retryAfter', { time: when(s.retryAfter) })}</p>}
        {s.status === 'could-not-check' && <p role="alert" className="text-red-600 dark:text-red-400">{t('settings.updates.couldNotCheck', { error: s.error ?? '' })}</p>}
        {notice && <p role="status">{t(`settings.updates.${notice}`)}</p>}
        {s.status === 'up-to-date' && <p role="status">{t('settings.updates.upToDate')}</p>}
        {s.status === 'ahead' && <p role="status">{t('settings.updates.ahead')}</p>}
        {s.status === 'update-available' && <div role="status" className="rounded border border-emerald-500/40 bg-emerald-500/10 p-3">
          <p>{t('settings.updates.available', { version: latest ?? '' })}</p>
          <p className="mt-1">{t('settings.updates.noInstall')}</p>
        </div>}
        {s.notesUrl && <a className="block underline" href={s.notesUrl} target="_blank" rel="noreferrer">{t(s.channel === 'release' ? 'settings.updates.notes' : 'settings.updates.commit')}</a>}
      </>}
    </div>
  </section>;
}
