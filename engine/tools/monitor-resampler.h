#pragma once
// Streaming linear resampler for vst-host monitor mode, used when the WASAPI
// device runs at a different rate from the track. Header-only and free of
// Windows APIs so engine/tests/test-monitor-resampler.cpp can check it.
//
// Every processed source frame pulled from the chain stays buffered until the
// read position has passed it, and the fractional phase carries over, so the
// source advances by exactly n * step frames per n device frames whatever the
// callback sizes are.

#include <cstddef>
#include <vector>

struct MonitorResampler {
    std::vector<float> L, R;  // processed source frames not yet passed; L[0] is at read position 0
    double phase = 0.0;       // read position into L/R, in source frames

    // Drop buffered frames: call on seek, track or rate change and stop, where
    // the buffered frames no longer follow on from the next source frame.
    void reset() { L.clear(); R.clear(); phase = 0.0; }

    // Writes n device-rate frames. step = source frames per device frame.
    // pull(dst_L, dst_R, count) must write the next count source frames.
    template <class Pull>
    void run(float * out_L, float * out_R, int n, double step, Pull && pull) {
        if (n <= 0) return;
        const double last = phase + (double)(n - 1) * step;
        const double next = phase + (double)n * step;
        // Enough for the last output's two taps, and never fewer frames than
        // the read position passes (downsampling can step past both taps).
        size_t need = (size_t)last + 2;
        if ((size_t)next > need) need = (size_t)next;
        if (L.size() < need) {
            const size_t have = L.size();
            L.resize(need);
            R.resize(need);
            pull(L.data() + have, R.data() + have, (int)(need - have));
        }
        for (int i = 0; i < n; i++) {
            const double p = phase + (double)i * step;
            const size_t k = (size_t)p;
            const float f = (float)(p - (double)k);
            out_L[i] = L[k] + (L[k + 1] - L[k]) * f;
            out_R[i] = R[k] + (R[k + 1] - R[k]) * f;
        }
        const size_t drop = (size_t)next;
        L.erase(L.begin(), L.begin() + (std::ptrdiff_t)drop);
        R.erase(R.begin(), R.begin() + (std::ptrdiff_t)drop);
        phase = next - (double)drop;
    }
};
