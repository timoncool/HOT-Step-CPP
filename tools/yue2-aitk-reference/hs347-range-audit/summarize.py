"""Summarise range-audit.h output: one row per GEMM family, plus group totals.

usage: python summarize.py audit.json
"""
import json
import sys


def changed(c):
    return c['sub_rounded'] + c['flushed'] + c['rounded_normal'] + c['overflow']


def group(key):
    kernel, direction, name, _graph = key.split('|')
    if kernel == 'f32':
        return 'F32 companion ' + direction
    if name == 'output.weight':
        return 'head logits fwd (no-grads graph)'
    if name == 'lm_head_transposed':
        return 'head dX (no-grads graph)'
    trainable = 'LoRA' if '(trainable)' in name else 'base'
    return '%s %s' % (trainable, direction)


def pct(a, b):
    return '%.4g%%' % (100.0 * a / b) if b else '-'


def main():
    d = json.load(open(sys.argv[1]))
    fams = d['families']
    total_flops = sum(f['flops'] for f in fams.values())
    print('mul_mm nodes audited: %d, graph calls: %d, skipped: %d' % (d['audited_mm_nodes'], d['calls'], d['skipped']))
    print()
    print('| family | nodes | FLOP share | src1 max abs | src1 min nonzero | src1 changed | src1 overflow / nonfinite | src0 changed | tile steps falling back (any / src0 / src1) |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---|')
    groups = {}
    for key in sorted(fams):
        f = fams[key]
        s0, s1 = f['src0'], f['src1']
        g = groups.setdefault(group(key), {'flops': 0, 'nodes': 0, 'steps': 0, 'any': 0, 's0': 0, 's1': 0,
                                           'e1': 0, 'c1': 0, 'e0': 0, 'c0': 0, 'ovf': 0, 'nf': 0, 'max1': 0})
        g['flops'] += f['flops']; g['nodes'] += f['mm_nodes']; g['steps'] += f['steps']
        g['any'] += f['steps_bad_any']; g['s0'] += f['steps_bad_src0']; g['s1'] += f['steps_bad_src1']
        g['e1'] += s1['elems']; g['c1'] += changed(s1); g['e0'] += s0['elems']; g['c0'] += changed(s0)
        g['ovf'] += s1['overflow'] + s0['overflow']; g['nf'] += s1['nonfinite'] + s0['nonfinite']
        g['max1'] = max(g['max1'], s1['max_abs'])
        print('| `%s` | %d | %s | %.3g | %.3g | %s | %d / %d | %s | %s / %s / %s |' % (
            key, f['mm_nodes'], pct(f['flops'], total_flops), s1['max_abs'], s1['min_nonzero'],
            pct(changed(s1), s1['elems']), s1['overflow'], s1['nonfinite'],
            pct(changed(s0), s0['elems']) if s0['elems'] else 'same as fwd',
            pct(f['steps_bad_any'], f['steps']), pct(f['steps_bad_src0'], f['steps']), pct(f['steps_bad_src1'], f['steps'])))
    print()
    print('| group | nodes | FLOP share | src1 max abs | src1 changed | src0 changed | overflow / nonfinite | tile steps falling back (any / src0 / src1) |')
    print('|---|---:|---:|---:|---:|---:|---:|---|')
    for name, g in sorted(groups.items(), key=lambda kv: -kv[1]['flops']):
        print('| %s | %d | %s | %.3g | %s | %s | %d / %d | %s / %s / %s |' % (
            name, g['nodes'], pct(g['flops'], total_flops), g['max1'], pct(g['c1'], g['e1']),
            pct(g['c0'], g['e0']) if g['e0'] else 'same as fwd', g['ovf'], g['nf'],
            pct(g['any'], g['steps']), pct(g['s0'], g['steps']), pct(g['s1'], g['steps'])))


if __name__ == '__main__':
    main()
