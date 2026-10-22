// monitor-chain.h: ping-pong processing of a plugin chain over fixed-size chunks.
// Kept header-only and free of VST3 types so it can be unit tested.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstring>

// Runs n_plugins stages over n frames in chunks of at most max_chunk.
// process(pi, in_L, in_R, out_L, out_R, chunk, offset) must write chunk frames
// to out_*. src_* holds the input (may be clobbered), scr_* is scratch (>= n).
// The output of the last stage (or src for an empty chain) goes to out_*.
template <typename ProcessFn>
static void monitor_run_chain(float * src_L, float * src_R, float * scr_L, float * scr_R,
                              float * out_L, float * out_R, int n, int max_chunk,
                              size_t n_plugins, ProcessFn && process) {
    int processed = 0;
    while (processed < n) {
        int chunk = std::min(n - processed, max_chunk);
        float * cur_L = src_L + processed, * cur_R = src_R + processed;
        float * dst_L = scr_L + processed, * dst_R = scr_R + processed;
        for (size_t pi = 0; pi < n_plugins; pi++) {
            process(pi, cur_L, cur_R, dst_L, dst_R, chunk, processed);
            std::swap(cur_L, dst_L);
            std::swap(cur_R, dst_R);
        }
        if (cur_L != out_L + processed) memmove(out_L + processed, cur_L, (size_t)chunk * sizeof(float));
        if (cur_R != out_R + processed) memmove(out_R + processed, cur_R, (size_t)chunk * sizeof(float));
        processed += chunk;
    }
}
