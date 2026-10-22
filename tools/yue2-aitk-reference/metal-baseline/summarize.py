"""Summarize matched JSONL runs without disclosing private dataset paths."""
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directories', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    report = {}
    for directory in args.directories:
        runs = []
        for logfile in sorted(directory.glob('run-*.log')):
            phases = [[]]
            compilations = [0]
            rss = None
            for line in logfile.read_text().splitlines():
                if line == 'HS342_PHASE measured_fresh_state':
                    phases.append([])
                    compilations.append(0)
                if 'compiling pipeline:' in line:
                    compilations[-1] += 1
                if 'maximum resident set size' in line:
                    rss = int(line.split()[0])
                if line.startswith('{'):
                    row = json.loads(line)
                    if row.get('stage') == 'joint':
                        for key in ('ar_ce', 'ar_kl', 'nar_mse', 'gradient_norm', 'step_ms'):
                            if not math.isfinite(row[key]):
                                raise ValueError('Nonfinite ' + key)
                        phases[-1].append(row)
            exitfile = logfile.with_suffix('.exit.json')
            exit_record = json.loads(exitfile.read_text()) if exitfile.exists() else None
            argv = json.loads(logfile.with_suffix('.argv.json').read_text())
            expected_steps = int(argv[argv.index('--steps') + 1])
            complete = (exit_record is not None and exit_record['exit_code'] == 0
                        and [x['step'] for x in phases[-1]] == list(range(1, expected_steps + 1)))
            runs.append({'name': logfile.stem, 'exit': exit_record,
                         'complete': complete, 'expected_steps': expected_steps,
                         'max_rss_bytes': rss, 'pipeline_compilations': compilations,
                         'warmup_steps': phases[0] if len(phases) > 1 else [],
                         'steps': phases[-1]})
        metrics = [run['steps'][0]['step_ms'] / 1000 for run in runs if run['complete']]
        comparisons = []
        fields = ('ar_ce', 'ar_kl', 'nar_mse', 'gradient_norm')
        for a, b in itertools.combinations([run for run in runs if run['complete']], 2):
            if len(a['steps']) != len(b['steps']):
                raise ValueError('Cannot compare different step counts')
            diffs = {key: 0.0 for key in fields}
            for left, right in zip(a['steps'], b['steps']):
                assert left['step'] == right['step']
                for key in fields:
                    if not math.isfinite(left[key]) or not math.isfinite(right[key]):
                        raise ValueError('Nonfinite ' + key)
                    diffs[key] = max(diffs[key], abs(left[key] - right[key]))
            comparisons.append({'pair': [a['name'], b['name']], 'max_abs_difference': diffs})
        report[directory.name] = {'runs': runs, 'step1_seconds': metrics,
                                  'median_seconds': statistics.median(metrics) if metrics else None,
                                  'range_seconds': max(metrics) - min(metrics) if metrics else None,
                                  'repeatability': comparisons}
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
