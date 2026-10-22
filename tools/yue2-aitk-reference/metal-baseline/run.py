"""Replay archived worker argv with fresh state; keep private inputs in probe storage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.request


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source_log', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--steps', type=int, default=7)
    p.add_argument('--binary', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    line = next(x.split(' | > $ ', 1)[1] for x in args.source_log.read_text().splitlines()
                if ' | > $ ' in x and ' yue2-joint-train ' in x)
    # The worker log prints JSON values without shell quoting. Parse flag
    # boundaries, not shell words, to retain the exact style-norms argument.
    parts = re.split(r' --([a-z0-9-]+) ', line)
    original = [parts[0].split(' yue2-joint-train')[0], 'yue2-joint-train']
    for flag, value in zip(parts[1::2], parts[2::2]):
        original.extend(['--' + flag, value])
    flags = dict(zip(original[2::2], original[3::2]))
    dataset = Path(flags['--dataset'])
    provenance = {'original_argv': original, 'source_log_sha256': sha256(args.source_log),
                  'parent': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  'ggml': subprocess.check_output(['git', '-C', 'engine/ggml', 'rev-parse', 'HEAD'], text=True).strip(),
                  'binary_sha256': sha256(args.binary or original[0]), 'files': []}
    files = {dataset, Path(flags['--checkpoint']), Path(flags['--companion'])}
    files.update(x.resolve() for x in Path('engine/build').glob('libggml*.dylib'))
    files.update(x for x in dataset.parent.rglob('*') if x.is_file())
    def referenced(value):
        if isinstance(value, dict):
            for child in value.values():
                referenced(child)
        elif isinstance(value, list):
            for child in value:
                referenced(child)
        elif isinstance(value, str):
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = dataset.parent / candidate
            try:
                if candidate.is_file():
                    files.add(candidate)
            except OSError:
                pass  # Captions and other text fields are not filesystem paths.
    referenced(json.loads(dataset.read_text()))
    for f in sorted(files):
        provenance['files'].append({'path': str(f), 'bytes': f.stat().st_size, 'sha256': sha256(f)})
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    env = os.environ.copy()
    overrides = {k: v for k, v in env.items() if k.startswith(('GGML_', 'YUE2_', 'HS340_'))}
    if overrides:
        raise RuntimeError('Benchmark requires no inherited diagnostic/backend overrides: ' + repr(overrides))
    for i in range(1, args.runs + 1):
        with urllib.request.urlopen('http://localhost:3000/api/training/worker/status', timeout=10) as response:
            worker = json.load(response)
        with urllib.request.urlopen('http://localhost:3000/api/generate/queue', timeout=10) as response:
            queue = json.load(response)
        if not worker.get('idle') or queue.get('running') or queue.get('pending'):
            raise RuntimeError('Worker or generation queue is busy')
        run = args.output / ('run-%d' % i)
        if run.exists():
            raise RuntimeError('Refusing to reuse training state')
        adjusted = dict(flags)
        adjusted.update({'--steps': str(args.steps), '--save-every': '100000', '--output': str(run)})
        adjusted.pop('--pause-at', None)
        command = [str(args.binary) if args.binary else original[0], original[1]] + [v for item in adjusted.items() for v in item]
        (args.output / ('run-%d.argv.json' % i)).write_text(json.dumps(command, indent=2))
        start = time.monotonic()
        print('START', i, flush=True)
        with (args.output / ('run-%d.log' % i)).open('w') as log:
            process = subprocess.Popen(['/usr/bin/time', '-l'] + command, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            try:
                code = process.wait(timeout=1500)
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        record = {'exit_code': code, 'wall_seconds': time.monotonic() - start}
        (args.output / ('run-%d.exit.json' % i)).write_text(json.dumps(record, indent=2))
        print('END', i, json.dumps(record), flush=True)
        if code:
            raise SystemExit(code)


if __name__ == '__main__':
    main()
