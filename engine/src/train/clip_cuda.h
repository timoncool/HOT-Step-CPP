#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace yue2_clip_cuda {

// One block's share of one gradient: `count` floats from `data`.
struct Chunk {
    float * data;
    int     count;
};

// Global-norm clip over every chunk, the device graph's maths in four
// launches on `stream`: denom = max|pre_scale*g| + eps, sum of
// (pre_scale*g/denom)^2 (per-chunk partials summed in a fixed order, so the
// result is deterministic), then every g *= min(ratio, 1) * pre_scale with
// ratio = (max_norm/denom) / (sqrt(sum) + eps/denom). `chunks` and `scratch`
// are device buffers (scratch holds n_chunks + 4 floats). Writes denom and
// sqrt(sum) to host and synchronizes `stream`. Returns cudaSuccess or the error.
cudaError_t clip(const Chunk * chunks, int n_chunks, float pre_scale, float max_norm, float eps,
                 float * scratch, float * host_denom, float * host_sqrt_sum, cudaStream_t stream);

constexpr int kChunkElements = 16384;

} // namespace yue2_clip_cuda
