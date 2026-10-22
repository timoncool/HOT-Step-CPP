#!/bin/zsh
# HOTSTEP-348: build the embedded mul_mm source the way ggml-metal's CMake does, then run the
# guarded half staging microbenchmark. Run from the mirror root.
# usage: run-bench.sh <ggml-src-dir> <outdir> [iters]
set -eu -o pipefail
ggml=$1; out=$2; iters=${3:-10}
here=${0:A:h}
[[ -e $out ]] && { echo "$out exists"; exit 2; }
mkdir -p $out
k=$ggml/ggml-metal/kernels
cat $k/common.h $k/dequantize.h $k/mul_mm.metal \
  | sed -e '/#include "common.h"/d' -e '/#include "dequantize.h"/d' -e '/#pragma once/d' \
  | sed -e "/__embed_ggml-common.h__/r $ggml/ggml-common.h" -e '/__embed_ggml-common.h__/d' \
  | sed -e "/#include \"ggml-metal-impl.h\"/r $ggml/ggml-metal/ggml-metal-impl.h" -e '/#include "ggml-metal-impl.h"/d' \
  > $out/mul_mm-embed.metal
clang++ -std=c++17 -O2 -fobjc-arc -I $ggml/ggml-metal $here/bench.mm -framework Metal -framework Foundation -o $out/bench
print -r -- "$out/bench $out/mul_mm-embed.metal $iters" > $out/bench.cmd
$out/bench $out/mul_mm-embed.metal $iters 2>&1 | tee $out/bench.log
