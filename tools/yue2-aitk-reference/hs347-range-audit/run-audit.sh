#!/bin/zsh
# HOTSTEP-347: one fresh seven-update audit run with A's frozen argv (HOTSTEP-342),
# then the exact-bits gate against the frozen warm baseline. Run from the mirror root.
# usage: run-audit.sh <audit-binary> <outdir>
set -eu -o pipefail
bin=$1; out=$2
python3 - <<'PY'
import json, urllib.request
w = json.load(urllib.request.urlopen('http://localhost:3000/api/training/worker/status', timeout=10)); assert w['idle'], w
q = json.load(urllib.request.urlopen('http://localhost:3000/api/generate/queue', timeout=10)); assert not q['running'] and not q['pending'], q
PY
[[ -e $out ]] && { echo "$out exists"; exit 2; }
mkdir -p $out
targs=("${(@f)$(python3 -c 'import json,sys; a=json.load(open(".probe-logs/hs343/b-full-1/run-1.argv.json")); i=a.index("--output"); a[i+1]=sys.argv[1]; j=a.index("--steps"); a[j+1]="7"; print("\n".join(a[1:]))' $out/run-1)}")
# Chunked execution hides later consumers from ggml-metal's single-use fusion
# check, so fusion is off; HOTSTEP-345 showed fused and unfused are bit-identical.
vars=(HS347_AUDIT_OUT=$out/audit.json GGML_METAL_FUSION_DISABLE=1 GGML_METAL_NO_BF16_ROUND_PRODUCER=1)
print -r -- "env ${vars[*]} $bin ${targs[*]}" > $out/run-1.cmd
# bits.py reads the updates after this marker (the warm probe's measured phase).
echo 'HS342_PHASE measured_fresh_state' > $out/run-1.log
set +e
env "${vars[@]}" /usr/bin/time -l $bin "${targs[@]}" >> $out/run-1.log 2>&1
rc=$?
echo "exit $rc" | tee $out/exit.txt
python3 .probe-logs/hs346/bits.py .probe-logs/hs342/warm-baseline-full/run-1.log $out/run-1.log | tee $out/bits.txt
gate=$?
# A trainer failure outranks the gate result; pipefail keeps the checker's status through tee.
(( rc == 0 )) || exit $rc
exit $gate
