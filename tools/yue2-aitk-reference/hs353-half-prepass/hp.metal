// HOTSTEP-353 bench: half pre-pass for kernel_mul_mm_bf16_f32 (appended to the embedded mul_mm source).
// A pre-pass writes half copies of the BF16 weights and the BF16-rounded F32 activations plus one flag per
// 8-wide K block of each tile (64 weight rows, 32 activation rows): 1 when every value is exact in half.
// The GEMM stages half on steps where both flags are 1 and bfloat otherwise. Mixed steps are bit-identical
// to all-bfloat (HOTSTEP-347 probe), so the output matches kernel_mul_mm_bf16_f32 bit for bit.

// weights [M][K] BF16 -> [M][K] half, mask [ceil(M/64)][K/8]; grid (K/32, ceil(M/64)), 64 threads
kernel void kernel_hp_cvt_w(
        device const bfloat * src,
        device       half   * dst,
        device       uchar  * mask,
        constant     int2   & KM,
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup bool votes[2][4];
    const int K = KM.x, M = KM.y;
    const int row = tgpig.y*64 + tiitg;
    bool ok[4] = {true, true, true, true};
    if (row < M) {
        device const bfloat4 * s = (device const bfloat4 *)(src + (uint64_t) row*K + tgpig.x*32);
        device       half4   * d = (device       half4   *)(dst + (uint64_t) row*K + tgpig.x*32);
        FOR_UNROLL (short i = 0; i < 8; i++) {
            const float4 f = (float4) s[i];
            const half4  h = (half4) f;
            ok[i/2] &= all((float4) h == f);
            d[i] = h;
        }
    }
    FOR_UNROLL (short j = 0; j < 4; j++) {
        const bool v = simd_all(ok[j]);
        if (tiisg == 0) votes[sgitg][j] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiitg < 4) mask[tgpig.y*(K/8) + 4*tgpig.x + tiitg] = votes[0][tiitg] && votes[1][tiitg];
}

// activations [N][K] F32 -> [N][K] half((bfloat) x), mask [ceil(N/32)][K/8]; grid (K/32, ceil(N/32)), 32 threads
kernel void kernel_hp_cvt_a(
        device const float * src,
        device       half  * dst,
        device       uchar * mask,
        constant     int2  & KN,
        device       bfloat * dstb,
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const int K = KN.x, N = KN.y;
    const int row = tgpig.y*32 + tiisg;
    bool ok[4] = {true, true, true, true};
    if (row < N) {
        device const float4 * s = (device const float4 *)(src + (uint64_t) row*K + tgpig.x*32);
        device       half4  * d = (device       half4  *)(dst + (uint64_t) row*K + tgpig.x*32);
        FOR_UNROLL (short i = 0; i < 8; i++) {
            const bfloat4 bb = (bfloat4) s[i];
            const float4 b = (float4) bb;
            const half4  h = (half4) b;
            ((device bfloat4 *)(dstb + (uint64_t) row*K + tgpig.x*32))[i] = bb;
            ok[i/2] &= all((float4) h == b);
            d[i] = h;
        }
    }
    FOR_UNROLL (short j = 0; j < 4; j++) {
        const bool v = simd_all(ok[j]);
        if (tiisg == 0) mask[tgpig.y*(K/8) + 4*tgpig.x + j] = v;
    }
}

#ifndef HP_FORCE
#define HP_FORCE 0 // bench-only: 1 = always half (bits differ), 2 = always bfloat
#endif
#ifndef HP_VARIANT
#define HP_VARIANT 1 // bench-only: 1 = branch per step, 2 = runs of same-type steps, 4 = type-agnostic staging, 5 = as 4 per 8-wide K block
#endif

// 2-D only, K % 32 == 0. src0/src1 as kernel_mul_mm_bf16_f32; src0h/src1h and masks from the pre-pass.
kernel void kernel_mul_mm_hp(
        constant ggml_metal_kargs_mul_mm & args,
        device const char  * src0,
        device const char  * src1,
        device       char  * dst,
        device const half  * src0h,
        device const half  * src1h,
        device const uchar * mask0,
        device const uchar * mask1,
        device const bfloat * src1b,
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int K  = args.ne00;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    const short iy  = 8*(tiitg % NL1);

    device const bfloat4x4 * x  = (device const bfloat4x4 *)(src0 + args.nb01*(r0 + lr0)) + il0;
    device const float2x4  * y  = (device const float2x4  *)((device const float *)(src1 + args.nb11*(r1 + lr1)) + iy);
    device const half4x4   * xh = (device const half4x4   *)(src0h + (uint64_t) K*(r0 + lr0) + 16*il0);
    device const half2x4   * yh = (device const half2x4   *)(src1h + (uint64_t) K*(r1 + lr1) + iy);

    device const uint * m0 = (device const uint *)(mask0 + tgpig.y*(K/8)); // 4 flags per 32-wide step
    device const uint * m1 = (device const uint *)(mask1 + tgpig.x*(K/8));

    // staging positions (as kernel_mul_mm)
    const short a_sx = 2*il0;
    const short a_sy = (tiitg/NL0)/8;
    const short a_lx = (tiitg/NL0)%8;
    const short b_ib = 4*(tiitg%NL1) + (tiitg/NL1)/8;
    const short b_ly = (tiitg/NL1)%8;

    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    // one 32-wide K step staged and multiplied in type S (half from the pre-pass copies, or bfloat
    // from the originals as kernel_mul_mm_bf16_f32 does)
#define HP_STEP(S, A4x4, AV, B2x4, BV)                                                                  {                                                                                                       threadgroup S * sA = (threadgroup S *)(shmem);                                                      threadgroup S * sB = (threadgroup S *)(shmem + 4096);                                               threadgroup_barrier(mem_flags::mem_threadgroup);                                                    const A4x4 ta = AV;                                                                                 FOR_UNROLL (short i = 0; i < 16; i++) {                                                                 const short ib = 8*(a_sx + i/8) + a_sy;                                                             *(sA + 64*ib + 8*(i%8) + a_lx) = ta[i/4][i%4];                                                  }                                                                                                   *(threadgroup B2x4 *)(sB + 64*b_ib + 8*b_ly) = BV;                                                  threadgroup_barrier(mem_flags::mem_threadgroup);                                                    threadgroup const S * lsma = (sA + 4*64*(sgitg%2));                                                 threadgroup const S * lsmb = (sB + 2*64*(sgitg/2));                                                 simdgroup_matrix<S, 8, 8> ma[4];                                                                    simdgroup_matrix<S, 8, 8> mb[2];                                                                    FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {                                                            simdgroup_barrier(mem_flags::mem_none);                                                             FOR_UNROLL (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);               simdgroup_barrier(mem_flags::mem_none);                                                             FOR_UNROLL (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);               simdgroup_barrier(mem_flags::mem_none);                                                             FOR_UNROLL (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);             lsma += 8*64;                                                                                       lsmb += 4*64;                                                                                   }                                                                                               }

#define HP_MAC(S)                                                                                       {                                                                                                       threadgroup const S * lsma = (threadgroup const S *)(shmem) + 4*64*(sgitg%2);                       threadgroup const S * lsmb = (threadgroup const S *)(shmem + 4096) + 2*64*(sgitg/2);                simdgroup_matrix<S, 8, 8> ma[4];                                                                    simdgroup_matrix<S, 8, 8> mb[2];                                                                    FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {                                                            simdgroup_barrier(mem_flags::mem_none);                                                             FOR_UNROLL (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);               simdgroup_barrier(mem_flags::mem_none);                                                             FOR_UNROLL (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);               simdgroup_barrier(mem_flags::mem_none);                                                             FOR_UNROLL (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);             lsma += 8*64;                                                                                       lsmb += 4*64;                                                                                   }                                                                                               }

#define HP_MAC1(S, ik)                                                                              \
    {                                                                                               \
        threadgroup const S * lsma = (threadgroup const S *)(shmem) + 4*64*(sgitg%2) + 8*64*(ik);   \
        threadgroup const S * lsmb = (threadgroup const S *)(shmem + 4096) + 2*64*(sgitg/2) + 4*64*(ik); \
        simdgroup_matrix<S, 8, 8> ma[4];                                                            \
        simdgroup_matrix<S, 8, 8> mb[2];                                                            \
        simdgroup_barrier(mem_flags::mem_none);                                                     \
        FOR_UNROLL (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);       \
        simdgroup_barrier(mem_flags::mem_none);                                                     \
        FOR_UNROLL (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);       \
        simdgroup_barrier(mem_flags::mem_none);                                                     \
        FOR_UNROLL (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]); \
    }

#define HP_FLAG(k) (HP_FORCE == 1 ? true : HP_FORCE == 2 ? false : (m0[k] & m1[k]) == 0x01010101u)

    const int nks = K/NK;
#if HP_VARIANT == 2
    // runs of same-type steps, each run in its own single-type loop
    for (int ks = 0; ks < nks; ) {
        const bool hp = HP_FLAG(ks);
        int ke = ks + 1;
        while (ke < nks && HP_FLAG(ke) == hp) ke++;
        if (hp) {
            for (; ks < ke; ks++) HP_STEP(half,   half4x4,   xh[2*ks], half2x4,   yh[4*ks])
        } else {
            for (; ks < ke; ks++) HP_STEP(bfloat, bfloat4x4, x[2*ks],  bfloat2x4, (bfloat2x4)(y[4*ks]))
        }
    }
#elif HP_VARIANT == 4
    // staging is a plain 16-bit copy from the half or the bfloat copy, picked per step; only the MAC
    // depends on the type
    {
        device const ushort4 * xw[2] = { (device const ushort4 *)(src0 + args.nb01*(r0 + lr0)) + 4*il0,
                                         (device const ushort4 *)(src0h + (uint64_t) K*(r0 + lr0)) + 4*il0 };
        device const ushort4 * ya[2] = { (device const ushort4 *)(src1b + (uint64_t) K*(r1 + lr1) + iy),
                                         (device const ushort4 *)(src1h + (uint64_t) K*(r1 + lr1) + iy) };
        threadgroup ushort * sA = (threadgroup ushort *)(shmem);
        threadgroup ushort * sB = (threadgroup ushort *)(shmem + 4096);
        for (int ks = 0; ks < nks; ks++) {
            const bool hp = HP_FLAG(ks);
            const short sel = hp;
            ushort4 ta[4];
            FOR_UNROLL (short j = 0; j < 4; j++) ta[j] = xw[sel][8*ks + j];
            const ushort4 tb0 = ya[sel][8*ks + 0];
            const ushort4 tb1 = ya[sel][8*ks + 1];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short ib = 8*(a_sx + i/8) + a_sy;
                *(sA + 64*ib + 8*(i%8) + a_lx) = ta[i/4][i%4];
            }
            *(threadgroup ushort4 *)(sB + 64*b_ib + 8*b_ly + 0) = tb0;
            *(threadgroup ushort4 *)(sB + 64*b_ib + 8*b_ly + 4) = tb1;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (hp) HP_MAC(half) else HP_MAC(bfloat)
        }
    }
#elif HP_VARIANT == 5
    // as 4, with the type picked per 8-wide K block: staging copies each 8-value block from the copy
    // its flag picks, and each 8x8 MAC block runs in that type
    {
        device const ushort4 * xw[2] = { (device const ushort4 *)(src0 + args.nb01*(r0 + lr0)) + 4*il0,
                                         (device const ushort4 *)(src0h + (uint64_t) K*(r0 + lr0)) + 4*il0 };
        device const ushort4 * ya[2] = { (device const ushort4 *)(src1b + (uint64_t) K*(r1 + lr1) + iy),
                                         (device const ushort4 *)(src1h + (uint64_t) K*(r1 + lr1) + iy) };
        threadgroup ushort * sA = (threadgroup ushort *)(shmem);
        threadgroup ushort * sB = (threadgroup ushort *)(shmem + 4096);
        for (int ks = 0; ks < nks; ks++) {
            const uint f = HP_FORCE == 1 ? 0x01010101u : HP_FORCE == 2 ? 0u : (m0[ks] & m1[ks]);
            const short s0 = (f >> (16*il0))     & 1;
            const short s1 = (f >> (16*il0 + 8)) & 1;
            const short sb = (f >> (8*(tiitg%NL1))) & 1;
            ushort4 ta[4];
            ta[0] = xw[s0][8*ks + 0];
            ta[1] = xw[s0][8*ks + 1];
            ta[2] = xw[s1][8*ks + 2];
            ta[3] = xw[s1][8*ks + 3];
            const ushort4 tb0 = ya[sb][8*ks + 0];
            const ushort4 tb1 = ya[sb][8*ks + 1];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short ib = 8*(a_sx + i/8) + a_sy;
                *(sA + 64*ib + 8*(i%8) + a_lx) = ta[i/4][i%4];
            }
            *(threadgroup ushort4 *)(sB + 64*b_ib + 8*b_ly + 0) = tb0;
            *(threadgroup ushort4 *)(sB + 64*b_ib + 8*b_ly + 4) = tb1;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
                if ((f >> (8*ik)) & 1) HP_MAC1(half, ik) else HP_MAC1(bfloat, ik)
            }
        }
    }
#else
    for (int ks = 0; ks < nks; ks++) {
        if (HP_FLAG(ks)) HP_STEP(half,   half4x4,   xh[2*ks], half2x4,   yh[4*ks])
        else             HP_STEP(bfloat, bfloat4x4, x[2*ks],  bfloat2x4, (bfloat2x4)(y[4*ks]))
    }
#endif
#undef HP_STEP
#undef HP_FLAG
#undef HP_MAC
#undef HP_MAC1

    if (FC_mul_mm_rnd || (FC_mul_mm_bc_out && (r0 + NR0 > args.ne0 || r1 + NR1 > args.ne1))) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int j = sgitg; j < nr1; j += N_MM_SIMD_GROUP_X*N_MM_SIMD_GROUP_Y) {
            device float * D = (device float *) dst + r0 + (r1 + j)*args.ne0;
            threadgroup const float * C = (threadgroup const float *) shmem + j*NR0;
            for (int i = tiitg%N_SIMDWIDTH; i < nr0; i += N_SIMDWIDTH) {
                D[i] = FC_mul_mm_rnd ? bf16_round_cast<float>(C[i]) : C[i];
            }
        }
        return;
    }

    device float * C = (device float *) dst + (r0 + 32*(sgitg & 1)) + (r1 + 16*(sgitg >> 1)) * args.ne0;
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], C + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
    }
}
