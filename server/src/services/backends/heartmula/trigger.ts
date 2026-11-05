/** `tags` led by the adapter's trigger word, as every training sample was, unless the user already typed it. */
export function withHeartmulaTrigger(tags: string, trigger: string | undefined): string {
  const word = trigger?.trim();
  if (!word) return tags;
  const has = tags.split(',').some(t => t.trim().toLowerCase() === word.toLowerCase());
  return has ? tags : tags.trim() ? `${word}, ${tags.trim()}` : word;
}
