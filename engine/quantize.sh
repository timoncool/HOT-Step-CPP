#!/bin/bash

set -eu

Q="./build/quantize"
if [ ! -x "$Q" ] && [ -x ./build/Release/quantize.exe ]; then
    Q="./build/Release/quantize.exe"
fi

# Optional single-model ladder, including the calibrated low-bit variants.
# Run from engine/: quantize.sh <model-BF16.gguf> [<imatrix.gguf>]
# QUANTIZE_JOBS controls bounded CPU batches, defaulting to one conversion.
if [ "$#" -gt 0 ]; then
    if [ "$#" -gt 2 ] || [ ! -f "$1" ] || { [ "$#" -eq 2 ] && [ ! -f "$2" ]; }; then
        echo "Usage: quantize.sh <existing model-BF16.gguf> [<existing imatrix.gguf>]" >&2
        exit 1
    fi
    source="$1"
    case "$source" in
        *-BF16.gguf) prefix="${source%-BF16.gguf}" ;;
        *) echo "Expected a -BF16.gguf source" >&2; exit 1 ;;
    esac
    max_jobs="${QUANTIZE_JOBS:-1}"
    case "$max_jobs" in
        ''|*[!0-9]*|0) echo "QUANTIZE_JOBS must be a positive integer" >&2; exit 1 ;;
    esac
    pids=()
    wait_quantizers() {
        local failed=0 pid
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then failed=1; fi
        done
        pids=()
        return "$failed"
    }
    quantize_one() {
        local out="$1" type="$2"
        shift 2
        "$Q" "$source" "${out}.partial" "$type" "$@"
        mv -- "${out}.partial" "$out"
    }
    queue_quantizer() {
        quantize_one "$@" &
        pids+=("$!")
        if [ "${#pids[@]}" -ge "$max_jobs" ]; then wait_quantizers; fi
    }
    for type in Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K Q8_0 NVFP4 MXFP4; do
        out="${prefix}-${type}.gguf"
        if [ ! -f "$out" ]; then
            queue_quantizer "$out" "$type"
        fi
    done
    wait_quantizers
    if [ "$#" -gt 1 ]; then
        for type in Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K Q8_0 IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ3_XXS IQ4_XS; do
            out="${prefix}-${type}-imat.gguf"
            if [ ! -f "$out" ]; then
                queue_quantizer "$out" "$type" --imatrix "$2"
            fi
        done
        wait_quantizers
    fi
    exit 0
fi

quantize() {
    local bf16="$1" type="$2"
    local out="${bf16/-BF16.gguf/-${type}.gguf}"
    if [ -f "$out" ]; then
        echo "[Skip] $out"
    else
        $Q "$bf16" "$out" "$type"
    fi
}

# Embedding: Q8_0 only (single-shot, precision matters)
quantize models/Qwen3-Embedding-0.6B-BF16.gguf Q8_0

# Small/medium LM: Q8_0 only (too small to survive aggressive quant)
quantize models/acestep-5Hz-lm-0.6B-BF16.gguf Q8_0
quantize models/acestep-5Hz-lm-1.7B-BF16.gguf Q8_0

# Large LM: full range (Q4_K_M confirmed broken for audio codes)
for type in Q5_K_M Q6_K Q8_0; do
    quantize models/acestep-5Hz-lm-4B-BF16.gguf "$type"
done

# DiT models: full range
for bf16 in models/acestep-v15-*-BF16.gguf; do
    for type in Q4_K_M Q5_K_M Q6_K Q8_0; do
        quantize "$bf16" "$type"
    done
done

# VAE: never quantized (stays BF16)
