#include "clip_cuda.h"

#include <cuda_runtime.h>

namespace yue2_clip_cuda {
namespace {
constexpr int kThreads = 256;

// |x| as ordered bits: for non-negative floats the unsigned bit patterns sort
// like the values, and a NaN sorts above infinity, so atomicMax on them is the
// float max with NaN winning (a non-finite gradient then fails the check).
__global__ void absmax_kernel(const Chunk * chunks, float pre_scale, unsigned int * out) {
    const Chunk c = chunks[blockIdx.x];
    unsigned int m = 0;
    for (int i = threadIdx.x; i < c.count; i += kThreads) {
        m = max(m, __float_as_uint(pre_scale * c.data[i]) & 0x7fffffffu);
    }
    __shared__ unsigned int s[kThreads];
    s[threadIdx.x] = m;
    __syncthreads();
    for (int stride = kThreads / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] = max(s[threadIdx.x], s[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(out, s[0]);
}

__global__ void sumsq_kernel(const Chunk * chunks, float pre_scale, float eps, const unsigned int * absmax, float * partials) {
    const Chunk c = chunks[blockIdx.x];
    const float denom = __uint_as_float(*absmax) + eps;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < c.count; i += kThreads) {
        const float v = pre_scale * c.data[i] / denom;
        acc += v * v;
    }
    __shared__ float s[kThreads];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = kThreads / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = s[0];
}

// One block: sums the partials in a fixed order, then the clip factor.
// out[0] denom, out[1] sqrt(sum), out[2] combined factor.
__global__ void finalize_kernel(const float * partials, int n, const unsigned int * absmax, float pre_scale,
                                float max_norm, float eps, float * out) {
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += kThreads) acc += partials[i];
    __shared__ float s[kThreads];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = kThreads / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float denom = __uint_as_float(*absmax) + eps;
        const float sqrt_sum = sqrtf(s[0]);
        const float ratio = (max_norm / denom) / (sqrt_sum + eps / denom);
        out[0] = denom;
        out[1] = sqrt_sum;
        out[2] = fminf(ratio, 1.0f) * pre_scale;
    }
}

__global__ void apply_kernel(const Chunk * chunks, const float * factor) {
    const Chunk c = chunks[blockIdx.x];
    const float f = factor[2];
    for (int i = threadIdx.x; i < c.count; i += kThreads) c.data[i] *= f;
}
} // namespace

cudaError_t clip(const Chunk * chunks, int n_chunks, float pre_scale, float max_norm, float eps,
                 float * scratch, float * host_denom, float * host_sqrt_sum, cudaStream_t stream) {
    if (n_chunks <= 0) return cudaErrorInvalidValue;
    unsigned int * absmax = reinterpret_cast<unsigned int *>(scratch);
    float * out = scratch + 1;
    float * partials = scratch + 4;
    cudaError_t err = cudaMemsetAsync(absmax, 0, sizeof(unsigned int), stream);
    if (err != cudaSuccess) return err;
    absmax_kernel<<<n_chunks, kThreads, 0, stream>>>(chunks, pre_scale, absmax);
    sumsq_kernel<<<n_chunks, kThreads, 0, stream>>>(chunks, pre_scale, eps, absmax, partials);
    finalize_kernel<<<1, kThreads, 0, stream>>>(partials, n_chunks, absmax, pre_scale, max_norm, eps, out);
    apply_kernel<<<n_chunks, kThreads, 0, stream>>>(chunks, out);
    if ((err = cudaGetLastError()) != cudaSuccess) return err;
    float host[2];
    if ((err = cudaMemcpyAsync(host, out, sizeof(host), cudaMemcpyDeviceToHost, stream)) != cudaSuccess) return err;
    if ((err = cudaStreamSynchronize(stream)) != cudaSuccess) return err;
    *host_denom = host[0];
    *host_sqrt_sum = host[1];
    return cudaSuccess;
}

} // namespace yue2_clip_cuda
