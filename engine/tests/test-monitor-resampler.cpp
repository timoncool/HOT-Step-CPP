// Self-check: the vst-host monitor resampler keeps every unconsumed source
// frame and its phase across irregular callback sizes. Header-only:
//   cl /std:c++17 /EHsc /I..\tools test-monitor-resampler.cpp
// The source is a ramp (frame j has value j), so linear interpolation must
// give exactly i * step for device frame i; a lost or repeated frame shows
// up as an error of a whole frame.
#include "monitor-resampler.h"
#include <cassert>
#include <cmath>
#include <cstdio>

static void check_rate(int src_sr, int dev_sr) {
    const double step = (double)src_sr / (double)dev_sr;
    MonitorResampler rs;
    long long pulled = 0;
    auto pull = [&](float * L, float * R, int count) {
        for (int j = 0; j < count; j++) { L[j] = (float)pulled; R[j] = -(float)pulled; pulled++; }
    };
    const int blocks[] = { 960, 1, 441, 480, 7, 1024, 2, 882, 333, 4096, 13, 960 };
    std::vector<float> L(4096), R(4096);
    long long done = 0;
    for (int round = 0; round < 3; round++) {
        for (int n : blocks) {
            rs.run(L.data(), R.data(), n, step, pull);
            for (int i = 0; i < n; i++) {
                const double want = (double)(done + i) * step;
                assert(std::fabs(L[i] - want) < 0.02);
                assert(std::fabs(R[i] + want) < 0.02);
            }
            done += n;
            // Source frames pulled = frames passed + frames still buffered.
            const double passed = (double)(pulled - (long long)rs.L.size()) + rs.phase;
            assert(std::fabs(passed - (double)done * step) < 1e-6);
            assert(rs.phase >= 0.0 && rs.phase < 1.0);
        }
    }

    // After reset (seek, track change, stop) output starts at the next pulled frame.
    rs.reset();
    const long long first = pulled;
    rs.run(L.data(), R.data(), 64, step, pull);
    for (int i = 0; i < 64; i++) assert(std::fabs(L[i] - ((double)first + i * step)) < 0.02);
    printf("ok %d -> %d Hz: %lld device frames, %lld source frames pulled\n", src_sr, dev_sr, done, pulled);
}

int main() {
    check_rate(44100, 48000);
    check_rate(48000, 44100);
    check_rate(48000, 48000);
    check_rate(32000, 48000);
    check_rate(96000, 44100);
    printf("monitor resampler: all checks passed\n");
    return 0;
}
