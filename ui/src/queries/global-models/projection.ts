import { hashKey, type QueryClient, type QueryKey } from '@tanstack/react-query';

// Compatibility readers keep their public store shape, but only this exact
// captured key can supply a server snapshot. No fetch is started here.
export function projectQuery<T>(client: QueryClient, key: QueryKey, apply: (data: T) => void) {
  const hash = hashKey(key);
  const current = client.getQueryData<T>(key);
  if (current !== undefined) apply(current);
  return client.getQueryCache().subscribe(event => {
    if (event.type === 'updated' && event.action.type === 'success' && hashKey(event.query.queryKey) === hash) {
      apply(event.query.state.data as T);
    }
  });
}
