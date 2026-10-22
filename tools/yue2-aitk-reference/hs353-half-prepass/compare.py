"""HOTSTEP-353: compare run.py arms with the frozen warm baseline (HOTSTEP-342), as H's collect.py.

usage: python3 compare.py <arm-dir> [<arm-dir> ...] [--output results.json]
Each arm dir holds run-1.log, run-1.exit.json and run-1/checkpoint-step7/. Exit 1 unless every arm
matches all 28 scalars bit for bit and all arms export the same adapter hashes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

FIELDS = ('ar_ce', 'nar_mse', 'ar_kl', 'gradient_norm')
REFERENCE = Path('.probe-logs/hs342/warm-baseline-full/run-1.log')


def parse(path):
    text = path.read_text()
    assert 'HS342_PHASE measured_fresh_state' in text, path
    rows = []
    for line in text.split('HS342_PHASE measured_fresh_state')[-1].splitlines():
        if line.startswith('{'):
            row = json.loads(line)
            if row.get('stage') == 'joint':
                assert all(math.isfinite(row[k]) for k in FIELDS + ('step_ms',)), path
                rows.append(row)
    assert [r['step'] for r in rows] == list(range(1, 8)), path
    mem = {}
    for line in text.splitlines():
        for key in ('maximum resident set size', 'peak memory footprint'):
            if key in line:
                mem[key] = int(line.split()[0])
    hp = sum('kernel_mul_mm_bf16_f32_hp' in line and 'compiling pipeline' in line for line in text.splitlines())
    return rows, mem, hp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('arms', nargs='+', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    reference, _, _ = parse(REFERENCE)
    results, ok = [], True
    for d in args.arms:
        exit_code = json.loads((d / 'run-1.exit.json').read_text())['exit_code']
        rows, mem, hp = parse(d / 'run-1.log')
        exact = all(struct.pack('>d', r[k]) == struct.pack('>d', ref[k]) for r, ref in zip(rows, reference) for k in FIELDS)
        hashes = {}
        for f in sorted((d / 'run-1/checkpoint-step7').glob('*.safetensors')):
            kind = 'combined' if f.name == 'adapter.safetensors' else ('ar' if '_ar_' in f.name else 'nar')
            hashes[kind] = hashlib.sha256(f.read_bytes()).hexdigest()
        ok &= exit_code == 0 and exact and len(hashes) == 3
        results.append({'arm': str(d), 'exit_code': exit_code, 'exact_28': exact, 'hp_pipelines_compiled': hp,
                        'step_s': [r['step_ms'] / 1000 for r in rows], 'warm_s': rows[0]['step_ms'] / 1000,
                        'max_rss_gb': mem.get('maximum resident set size', 0) / 1e9,
                        'peak_footprint_gb': mem.get('peak memory footprint', 0) / 1e9, 'adapters': hashes})
        print('%s: exit %d, 28 scalars %s, hp pipelines %d, warm %.3f s, seven-step %.3f s, rss %.3f GB, footprint %.3f GB'
              % (d, exit_code, 'EXACT' if exact else 'DIFFER', hp, rows[0]['step_ms'] / 1000,
                 sum(r['step_ms'] for r in rows) / 1000, results[-1]['max_rss_gb'], results[-1]['peak_footprint_gb']))
    same = all(r['adapters'] == results[0]['adapters'] for r in results)
    ok &= same
    print('adapter hashes %s across %d arms' % ('EQUAL' if same else 'DIFFER', len(results)))
    if args.output:
        args.output.write_text(json.dumps({'all_exact': ok, 'adapters_equal': same, 'arms': results}, indent=2))
    print('GATE', 'PASS' if ok else 'FAIL')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
