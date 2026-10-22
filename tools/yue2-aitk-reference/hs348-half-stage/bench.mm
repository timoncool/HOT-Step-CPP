// HOTSTEP-348: guarded half staging microbenchmark for kernel_mul_mm_bf16_f32 (fix 2).
// Compiles the fork's embedded mul_mm source as ggml-metal does, then runs today's BF16 kernel
// (hg=0), the guarded kernel (hg=1) and the F16-weight kernel (plain half staging, the ceiling)
// on the base-forward shapes with the fallback mix the HOTSTEP-347 audit measured.
// Checks hg=1 against hg=0 bit for bit on every input.
//   usage: bench <embedded-mul_mm.metal> [iters]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml-metal-impl.h"

static uint16_t f2bf(float f) { // RNE, finite inputs only
    uint32_t u; memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}
static float bf2f(uint16_t b) { uint32_t u = (uint32_t) b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2h(float f) { __fp16 h = (__fp16) f; uint16_t r; memcpy(&r, &h, 2); return r; }

struct Shape { const char * name; int K, M, N; double wfail, afail; };
enum Mix { MIX_AUDIT, MIX_PASS, MIX_FAIL, MIX_SUBNORMAL };
static const char * mix_name[] = {"audit", "all-half", "all-bfloat", "subnormal"};

static id<MTLDevice> dev;
static id<MTLLibrary> lib;
static id<MTLCommandQueue> queue;

static id<MTLComputePipelineState> pipeline(const char * base, bool bc_out, bool hg, bool rnd) {
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    bool bc_inp = false; int16_t one = 1;
    [cv setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:FC_MUL_MM + 0];
    [cv setConstantValue:&bc_out type:MTLDataTypeBool atIndex:FC_MUL_MM + 1];
    for (int i = 2; i <= 5; i++) [cv setConstantValue:&one type:MTLDataTypeShort atIndex:FC_MUL_MM + i];
    [cv setConstantValue:&rnd type:MTLDataTypeBool atIndex:FC_MUL_MM + 7];
    [cv setConstantValue:&hg  type:MTLDataTypeBool atIndex:FC_MUL_MM + 8];
    NSError * err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:base] constantValues:cv error:&err];
    if (!fn) { fprintf(stderr, "function %s: %s\n", base, err.localizedDescription.UTF8String); exit(2); }
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) { fprintf(stderr, "pipeline %s: %s\n", base, err.localizedDescription.UTF8String); exit(2); }
    return p;
}

// Weights [M][K] BF16 and activations [N][K] F32. Normal-range values are exact in half after the
// BF16 round; a planted 1e-9 makes its tile (64 weight rows or 32 activation rows by 32 K) fall back.
static void fill(const Shape & s, Mix mix, std::vector<uint16_t> & w, std::vector<float> & a, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    w.resize((size_t) s.M * s.K);
    a.resize((size_t) s.N * s.K);
    auto mag = [&](float lo, float hi) { return std::exp2(lo + (hi - lo) * u(rng)) * (u(rng) < 0.5f ? -1.f : 1.f); };
    if (mix == MIX_SUBNORMAL) {
        // exact half subnormals and near-subnormals mixed in (k * 2^-24 with <= 8 significant bits)
        for (auto & v : w) v = f2bf(u(rng) < 0.5f ? (float)((int)(u(rng) * 255) + 1) * std::exp2(-24.f) : mag(-10, -4));
        for (auto & v : a) v = u(rng) < 0.5f ? (float)((int)(u(rng) * 255) + 1) * std::exp2(-24.f) : mag(-4, 2);
        return;
    }
    for (auto & v : w) v = f2bf(mag(-10, -4));
    for (auto & v : a) v = mag(-4, 2);
    const double wf = mix == MIX_AUDIT ? s.wfail : mix == MIX_FAIL ? 1.0 : 0.0;
    const double af = mix == MIX_AUDIT ? s.afail : mix == MIX_FAIL ? 1.0 : 0.0;
    for (int r0 = 0; r0 < s.M; r0 += 64)
        for (int k0 = 0; k0 < s.K; k0 += 32)
            if (u(rng) < wf) {
                const int r = std::min(s.M - 1, r0 + (int)(u(rng) * 64)), k = k0 + (int)(u(rng) * 32);
                w[(size_t) r * s.K + k] = f2bf(1e-9f);
            }
    for (int r1 = 0; r1 < s.N; r1 += 32)
        for (int k0 = 0; k0 < s.K; k0 += 32)
            if (u(rng) < af) {
                const int r = std::min(s.N - 1, r1 + (int)(u(rng) * 32)), k = k0 + (int)(u(rng) * 32);
                a[(size_t) r * s.K + k] = 1e-9f;
            }
}

// median per-dispatch GPU time in ms over `reps` command buffers of `iters` dispatches
static double run(id<MTLComputePipelineState> p, const Shape & s, int elsize0, id<MTLBuffer> b0, id<MTLBuffer> b1,
                  id<MTLBuffer> bd, size_t smem, int iters, int reps) {
    ggml_metal_kargs_mul_mm args = {};
    args.ne00 = s.K; args.ne02 = 1;
    args.nb01 = (uint64_t) s.K * elsize0; args.nb02 = args.nb01 * s.M; args.nb03 = args.nb02;
    args.ne12 = 1;
    args.nb10 = 4; args.nb11 = (uint64_t) s.K * 4; args.nb12 = args.nb11 * s.N; args.nb13 = args.nb12;
    args.ne0 = s.M; args.ne1 = s.N; args.r2 = 1; args.r3 = 1;
    std::vector<double> t;
    for (int rep = 0; rep < reps; rep++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:b0 offset:0 atIndex:1];
        [enc setBuffer:b1 offset:0 atIndex:2];
        [enc setBuffer:bd offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:smem atIndex:0];
        for (int i = 0; i < iters; i++)
            [enc dispatchThreadgroups:MTLSizeMake((s.N + 31) / 32, (s.M + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { fprintf(stderr, "command buffer: %s\n", cb.error.localizedDescription.UTF8String); exit(2); }
        t.push_back((cb.GPUEndTime - cb.GPUStartTime) * 1e3 / iters);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "usage: bench <embedded-mul_mm.metal> [iters]\n"); return 2; }
        const int iters = argc > 2 ? atoi(argv[2]) : 10;
        dev = MTLCreateSystemDefaultDevice();
        queue = [dev newCommandQueue];
        NSError * err = nil;
        NSString * src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "read: %s\n", err.localizedDescription.UTF8String); return 2; }
        MTLCompileOptions * opt = [MTLCompileOptions new]; // ggml-metal's defaults
        opt.preprocessorMacros = @{@"GGML_METAL_HAS_BF16": @"1", @"GGML_METAL_EMBED_LIBRARY": @"1"};
        lib = [dev newLibraryWithSource:src options:opt error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 2; }
        printf("device %s, iters %d\n", dev.name.UTF8String, iters);

        // base-forward families from the HOTSTEP-347 audit (AR length 7668): fallback rates per
        // weight tile and per activation tile
        const Shape shapes[] = {
            {"gate_up", 2048, 12288, 7668, 0.2516, 0.00056},
            {"qkv",     2048,  4096, 7668, 0.2656, 0.0079},
            {"down",    6144,  2048, 7668, 0.2519, 0.2976},
            {"o",       2048,  2048, 7668, 0.2560, 0.0832},
            {"gate_up/N7680", 2048, 12288, 7680, 0.2516, 0.00056},
        };
        // FLOP weights of the families in base forward (audit: gate_up 31.8, qkv 10.6,
        // down 14.4 + 1.5, o 4.8 + 0.5 of 63.6%)
        const double flop_w[] = {31.8, 10.6, 15.9, 5.3, 0.0};

        bool all_ok = true;
        double w_base = 0, w_hg = 0, w_f16 = 0;
        printf("| shape | mix | bf16 ms | guarded ms | f16 ms | guarded/bf16 | bits |\n|---|---|---:|---:|---:|---:|---|\n");
        for (int si = 0; si < (int)(sizeof(shapes) / sizeof(shapes[0])); si++) {
            const Shape & s = shapes[si];
            const bool bc_out = s.M % 64 != 0 || s.N % 32 != 0;
            auto p_base = pipeline("kernel_mul_mm_bf16_f32", bc_out, false, false);
            auto p_hg   = pipeline("kernel_mul_mm_bf16_f32", bc_out, true,  false);
            auto p_f16  = pipeline("kernel_mul_mm_f16_f32",  bc_out, false, false);
            auto p_base_r = pipeline("kernel_mul_mm_bf16_f32", bc_out, false, true);
            auto p_hg_r   = pipeline("kernel_mul_mm_bf16_f32", bc_out, true,  true);
            const size_t smem_base = bc_out ? 8192 : 6144;
            for (int mi = 0; mi < 4; mi++) {
                const Mix mix = (Mix) mi;
                if (si == 4 && mix != MIX_AUDIT) continue;
                std::vector<uint16_t> w; std::vector<float> a;
                fill(s, mix, w, a, 42 + si * 10 + mi);
                std::vector<uint16_t> wh(w.size());
                for (size_t i = 0; i < w.size(); i++) wh[i] = f2h(bf2f(w[i]));
                id<MTLBuffer> b0  = [dev newBufferWithBytes:w.data()  length:w.size() * 2 options:MTLResourceStorageModeShared];
                id<MTLBuffer> b0h = [dev newBufferWithBytes:wh.data() length:wh.size() * 2 options:MTLResourceStorageModeShared];
                id<MTLBuffer> b1  = [dev newBufferWithBytes:a.data()  length:a.size() * 4 options:MTLResourceStorageModeShared];
                const size_t dsz = (size_t) s.M * s.N * 4;
                id<MTLBuffer> d0 = [dev newBufferWithLength:dsz options:MTLResourceStorageModeShared];
                id<MTLBuffer> d1 = [dev newBufferWithLength:dsz options:MTLResourceStorageModeShared];
                id<MTLBuffer> d2 = [dev newBufferWithLength:dsz options:MTLResourceStorageModeShared];

                // parity (one dispatch each), plain and with the BF16_ROUND epilogue
                memset(d0.contents, 0xff, dsz); memset(d1.contents, 0x7f, dsz);
                run(p_base, s, 2, b0, b1, d0, smem_base, 1, 1);
                run(p_hg,   s, 2, b0, b1, d1, 8192,      1, 1);
                const bool eq = memcmp(d0.contents, d1.contents, dsz) == 0;
                memset(d0.contents, 0xff, dsz); memset(d1.contents, 0x7f, dsz);
                run(p_base_r, s, 2, b0, b1, d0, 8192, 1, 1);
                run(p_hg_r,   s, 2, b0, b1, d1, 8192, 1, 1);
                const bool eq_r = memcmp(d0.contents, d1.contents, dsz) == 0;
                all_ok &= eq && eq_r;

                double tb = 0, th = 0, tf = 0;
                if (mix != MIX_SUBNORMAL) {
                    run(p_base, s, 2, b0,  b1, d0, smem_base, 3, 1); // warm
                    // interleave so drift and thermals hit every arm alike
                    std::vector<double> vb, vh, vf;
                    for (int r = 0; r < 5; r++) {
                        vb.push_back(run(p_base, s, 2, b0,  b1, d0, smem_base, iters, 1));
                        vh.push_back(run(p_hg,   s, 2, b0,  b1, d1, 8192,      iters, 1));
                        vf.push_back(run(p_f16,  s, 2, b0h, b1, d2, smem_base, iters, 1));
                    }
                    std::sort(vb.begin(), vb.end()); std::sort(vh.begin(), vh.end()); std::sort(vf.begin(), vf.end());
                    tb = vb[2]; th = vh[2]; tf = vf[2];
                    if (mix == MIX_AUDIT) { w_base += flop_w[si] * 1.0; w_hg += flop_w[si] * th / tb; w_f16 += flop_w[si] * tf / tb; }
                }
                printf("| %s | %s | %.3f | %.3f | %.3f | %.4f | %s%s |\n", s.name, mix_name[mi], tb, th, tf,
                       tb > 0 ? th / tb : 0.0, eq ? "equal" : "DIFFER", eq_r ? "" : ", rnd DIFFER");
                fflush(stdout);
            }
        }
        printf("\nFLOP-weighted base-forward ratio (audit mix): guarded/bf16 %.4f, f16/bf16 %.4f\n", w_hg / w_base, w_f16 / w_base);
        printf("BITS %s\n", all_ok ? "PASS" : "FAIL");
        return all_ok ? 0 : 1;
    }
}
