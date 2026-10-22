#include "joint_loss_cuda.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <climits>
#include <cmath>
#include <cfloat>
#include <limits>

namespace yue2_joint_loss_cuda {
namespace {
constexpr int kThreads = 256;

__device__ __forceinline__ float reduce_max(float value, float *shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        __syncthreads();
    }
    const float result = shared[0];
    __syncthreads(); // All lanes must read the result before scratch is reused.
    return result;
}

__device__ __forceinline__ float reduce_sum(float value, float *shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const float result = shared[0];
    __syncthreads();
    return result;
}

__device__ __forceinline__ float load(float x) { return x; }
__device__ __forceinline__ float load(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ void store(float * p, float x) { *p = x; }
// Round to nearest even, as ggml_bf16_round() does to the F32 gradient.
__device__ __forceinline__ void store(__nv_bfloat16 * p, float x) { *p = __float2bfloat16(x); }

// T: logit storage (F32, or BF16 holding the same BF16-rounded values); G:
// gradient storage (F32, or BF16 when the caller would round it anyway).
template <class T, class G>
__global__ void loss_chunk_kernel(const T *adapted, const T *base, const uint32_t *targets, G *gradient,
                                  float *per_ce, float *per_kl, std::size_t offset, std::size_t count,
                                  std::size_t total, std::size_t vocab, float kl_weight) {
    const std::size_t local = blockIdx.x;
    if (local >= count) return;
    const std::size_t row = local, global_row = offset + local;
    const T *a = adapted + row * vocab, *b = base + row * vocab;
    __shared__ float max_a[kThreads], max_b[kThreads], sum_a[kThreads], sum_b[kThreads];
    float amax = -FLT_MAX, bmax = -FLT_MAX;
    for (std::size_t j = threadIdx.x; j < vocab; j += blockDim.x) {
        amax = fmaxf(amax, load(a[j])); bmax = fmaxf(bmax, load(b[j]));
    }
    amax = reduce_max(amax, max_a);
    bmax = reduce_max(bmax, max_b);
    float asum = 0.0f, bsum = 0.0f;
    for (std::size_t j = threadIdx.x; j < vocab; j += blockDim.x) {
        asum += expf(load(a[j]) - amax); bsum += expf(load(b[j]) - bmax);
    }
    asum = reduce_sum(asum, sum_a); bsum = reduce_sum(bsum, sum_b);
    const float alogsum = logf(asum), blogsum = logf(bsum);
    float ce = 0.0f, kl = 0.0f;
    const uint32_t target = targets[row];
    const bool valid_target = target < vocab;
    if (threadIdx.x == 0 && valid_target) ce = alogsum - (load(a[target]) - amax);
    for (std::size_t j = threadIdx.x; j < vocab; j += blockDim.x) {
        const float logp = (load(a[j]) - amax) - alogsum, logq = (load(b[j]) - bmax) - blogsum;
        const float p = expf(logp), q = expf(logq);
        kl += q * (logq - logp);
        store(gradient + row * vocab + j, valid_target ? (p - (j == target ? 1.0f : 0.0f) + kl_weight * (p - q)) / static_cast<float>(total) : 0.0f);
    }
    kl = reduce_sum(kl, sum_b);
    if (threadIdx.x == 0) { const float nan = nanf(""); per_ce[global_row] = valid_target ? ce : nan; per_kl[global_row] = valid_target ? kl : nan; }
}

__global__ void reduce_kernel(const float *per_ce, const float *per_kl, std::size_t n, float *ce, float *kl) {
    __shared__ float a[kThreads], b[kThreads];
    float x = 0.0f, y = 0.0f;
    for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) { x += per_ce[i]; y += per_kl[i]; }
    x = reduce_sum(x, a); y = reduce_sum(y, b);
    if (threadIdx.x == 0) { *ce = x / static_cast<float>(n); *kl = y / static_cast<float>(n); }
}
} // namespace

Status forward_chunk(const float *adapted, const float *base, const uint32_t *targets, float *gradient,
                     float *per_ce, float *per_kl, std::size_t offset, std::size_t count, std::size_t total,
                     std::size_t vocab, float kl_weight, cudaStream_t stream) {
    if (!adapted || !base || !targets || !gradient || !per_ce || !per_kl || !count || !total || !vocab ||
        !std::isfinite(kl_weight) || kl_weight < 0.0f || count > static_cast<std::size_t>(INT_MAX) ||
        offset > total || count > total - offset || vocab > static_cast<std::size_t>(INT_MAX)) return Status::invalid_argument;
    loss_chunk_kernel<<<static_cast<unsigned>(count), kThreads, 0, stream>>>(adapted, base, targets, gradient,
        per_ce, per_kl, offset, count, total, vocab, kl_weight);
    return cudaGetLastError() == cudaSuccess ? Status::success : Status::cuda_error;
}

Status forward_chunk_bf16(const void *adapted, const void *base, const uint32_t *targets, void *gradient,
                          float *per_ce, float *per_kl, std::size_t offset, std::size_t count, std::size_t total,
                          std::size_t vocab, float kl_weight, cudaStream_t stream) {
    if (!adapted || !base || !targets || !gradient || !per_ce || !per_kl || !count || !total || !vocab ||
        !std::isfinite(kl_weight) || kl_weight < 0.0f || count > static_cast<std::size_t>(INT_MAX) ||
        offset > total || count > total - offset || vocab > static_cast<std::size_t>(INT_MAX)) return Status::invalid_argument;
    loss_chunk_kernel<<<static_cast<unsigned>(count), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(adapted), static_cast<const __nv_bfloat16 *>(base), targets,
        static_cast<__nv_bfloat16 *>(gradient), per_ce, per_kl, offset, count, total, vocab, kl_weight);
    return cudaGetLastError() == cudaSuccess ? Status::success : Status::cuda_error;
}

Status reduce(const float *per_ce, const float *per_kl, std::size_t total, float *ce, float *kl, cudaStream_t stream) {
    if (!per_ce || !per_kl || !ce || !kl || !total) return Status::invalid_argument;
    reduce_kernel<<<1, kThreads, 0, stream>>>(per_ce, per_kl, total, ce, kl);
    return cudaGetLastError() == cudaSuccess ? Status::success : Status::cuda_error;
}

const char *status_string(Status status) {
    switch (status) { case Status::success: return "success"; case Status::invalid_argument: return "invalid_argument"; case Status::cuda_error: return "cuda_error"; }
    return "unknown";
}
} // namespace yue2_joint_loss_cuda
