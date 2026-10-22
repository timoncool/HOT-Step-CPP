// Fake ace-midi for midiStudio tests: prints JSONL chunk progress lines.
// argv: <chunks> <intervalMs> [stallAfter]  (stallAfter: go silent after N chunks)
const [chunks, intervalMs, stallAfter] = process.argv.slice(2).map(Number);
let done = 0;
const tick = () => {
  if (stallAfter && done >= stallAfter) { setInterval(() => {}, 1000); return; }
  done++;
  process.stdout.write(JSON.stringify({ type: 'progress', completed: done, total: chunks }) + '\n');
  if (done < chunks) setTimeout(tick, intervalMs); else process.exit(0);
};
setTimeout(tick, intervalMs);
