// Unit test for monitor_run_chain: every stage in the chain must be heard.
#include "../tools/monitor-chain.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

static int run_case(const std::vector<float> & gains, int n) {
    std::vector<float> sL(n), sR(n), bL(n), bR(n), oL(n), oR(n);
    for (int i = 0; i < n; i++) { sL[i] = 0.001f * (float)(i % 97 + 1); sR[i] = -sL[i]; }
    std::vector<float> refL = sL;
    float want = 1.0f;
    for (float g : gains) want *= g;
    monitor_run_chain(sL.data(), sR.data(), bL.data(), bR.data(), oL.data(), oR.data(), n, 4096,
        gains.size(), [&](size_t pi, float * iL, float * iR, float * dL, float * dR, int c, int) {
            for (int k = 0; k < c; k++) { dL[k] = iL[k] * gains[pi]; dR[k] = iR[k] * gains[pi]; }
        });
    double worst = 0; float got = 0;
    for (int i = 0; i < n; i++) {
        float r = oL[i] / refL[i];
        double e = std::fabs(r - want);
        if (e > worst) { worst = e; got = r; }
        e = std::fabs(oR[i] / -refL[i] - want);
        if (e > worst) { worst = e; got = oR[i] / -refL[i]; }
    }
    bool ok = worst < 1e-4 * want;
    printf("%s: stages=%zu n=%d want x%g worst got x%g\n", ok ? "PASS" : "FAIL", gains.size(), n, want,
           worst == 0 ? want : got);
    return ok ? 0 : 1;
}

// Non-commutative stages: stage k computes y = x * g[k] + (k + 1), so the
// expected output depends on stage order. With in_place, out aliases src.
static int run_order_case(const std::vector<float> & gains, int n, bool in_place) {
    std::vector<float> sL(n), sR(n), bL(n), bR(n), oL(n), oR(n);
    for (int i = 0; i < n; i++) { sL[i] = 0.001f * (float)(i % 97 + 1); sR[i] = -sL[i]; }
    std::vector<double> wantL(n), wantR(n);
    for (int i = 0; i < n; i++) {
        double l = sL[i], r = sR[i];
        for (size_t k = 0; k < gains.size(); k++) { l = l * gains[k] + (double)(k + 1); r = r * gains[k] + (double)(k + 1); }
        wantL[i] = l; wantR[i] = r;
    }
    float * outL = in_place ? sL.data() : oL.data();
    float * outR = in_place ? sR.data() : oR.data();
    monitor_run_chain(sL.data(), sR.data(), bL.data(), bR.data(), outL, outR, n, 4096,
        gains.size(), [&](size_t pi, float * iL, float * iR, float * dL, float * dR, int c, int) {
            for (int k = 0; k < c; k++) {
                dL[k] = iL[k] * gains[pi] + (float)(pi + 1);
                dR[k] = iR[k] * gains[pi] + (float)(pi + 1);
            }
        });
    double worst = 0; int at = 0; double gotv = 0, wantv = 0;
    for (int i = 0; i < n; i++) {
        double e = std::fabs(outL[i] - wantL[i]);
        if (e > worst) { worst = e; at = i; gotv = outL[i]; wantv = wantL[i]; }
        e = std::fabs(outR[i] - wantR[i]);
        if (e > worst) { worst = e; at = i; gotv = outR[i]; wantv = wantR[i]; }
    }
    bool ok = worst < 1e-3;
    printf("%s: order stages=%zu n=%d in_place=%d worst |err|=%g at %d (got %g want %g)\n", ok ? "PASS" : "FAIL",
           gains.size(), n, in_place ? 1 : 0, worst, at, gotv, wantv);
    return ok ? 0 : 1;
}

// Block sizes: the live monitor feeds whatever block the audio device asks for,
// so run a stateful chain at odd sizes against a one-block reference. Input is
// a 0.5 s decaying burst, then 1 s of silence. Stage 0 is a one-pole lowpass
// whose state crosses block edges; with flush on it zeroes |y| < 1e-20, as a
// well-behaved plugin does (nothing in the host sets FTZ/DAZ). Every size must
// match the reference bit for bit, stay finite, hold no subnormals and end in
// exact silence. The flush-off run must produce subnormals, which proves the
// counter works and shows the cost of a tail that never reaches zero.
static int run_block_case(const std::vector<int> & sizes) {
    const int n = 72000, burst = 24000;
    std::vector<float> in(n, 0.0f);
    for (int i = 0; i < burst; i++) in[i] = 0.5f * std::exp(-i / 4000.0f) * std::sin(i * 0.05f);

    auto run = [&](int chunk, bool flush, std::vector<float> & oL, std::vector<float> & oR, double & ns) {
        std::vector<float> sL = in, sR = in, bL(n), bR(n);
        oL.assign(n, 0.0f); oR.assign(n, 0.0f);
        float yL = 0.0f, yR = 0.0f;
        auto t0 = std::chrono::steady_clock::now();
        monitor_run_chain(sL.data(), sR.data(), bL.data(), bR.data(), oL.data(), oR.data(), n, chunk, 2,
            [&](size_t pi, float * iL, float * iR, float * dL, float * dR, int c, int) {
                for (int k = 0; k < c; k++) {
                    if (pi == 0) {
                        yL += 0.01f * (iL[k] - yL);
                        yR += 0.01f * (-iR[k] - yR);
                        if (flush && std::fabs(yL) < 1e-20f) yL = 0.0f;
                        if (flush && std::fabs(yR) < 1e-20f) yR = 0.0f;
                        dL[k] = yL; dR[k] = yR;
                    } else {
                        dL[k] = iL[k] * 0.5f; dR[k] = iR[k] * 0.5f;
                    }
                }
            });
        ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / n;
    };
    auto count = [&](const std::vector<float> & v, int & nonfin, int & subn) {
        for (float x : v) {
            if (!std::isfinite(x)) nonfin++;
            else if (std::fpclassify(x) == FP_SUBNORMAL) subn++;
        }
    };

    int f = 0;
    double ns;
    std::vector<float> refL, refR, oL, oR;
    run(n, true, refL, refR, ns);
    for (int chunk : sizes) {
        run(chunk, true, oL, oR, ns);
        int nonfin = 0, subn = 0;
        count(oL, nonfin, subn); count(oR, nonfin, subn);
        bool exact = std::memcmp(oL.data(), refL.data(), n * sizeof(float)) == 0 &&
                     std::memcmp(oR.data(), refR.data(), n * sizeof(float)) == 0;
        bool silent = true;
        for (int i = n - 4800; i < n; i++) silent = silent && oL[i] == 0.0f && oR[i] == 0.0f;
        bool ok = exact && nonfin == 0 && subn == 0 && silent;
        printf("%s: block=%d bit-exact=%d nonfinite=%d subnormal=%d tail-silent=%d %.2f ns/sample\n",
               ok ? "PASS" : "FAIL", chunk, exact, nonfin, subn, silent, ns);
        f += ok ? 0 : 1;
    }
    run(441, false, oL, oR, ns);
    int nonfin = 0, subn = 0;
    count(oL, nonfin, subn); count(oR, nonfin, subn);
    bool ok = subn > 0 && nonfin == 0;
    printf("%s: block=441 flush=off subnormal=%d (want > 0, proves the counter) %.2f ns/sample\n",
           ok ? "PASS" : "FAIL", subn, ns);
    return f + (ok ? 0 : 1);
}

int main() {
    int f = 0;
    f += run_block_case({1, 37, 441, 1000, 4096});
    f += run_case({2, 3, 5}, 512);
    f += run_case({2, 3, 5}, 9000);
    f += run_case({2, 3, 5, 7}, 9000);
    f += run_case({4}, 9000);
    f += run_case({}, 9000);
    f += run_order_case({2, 3, 5}, 9000, false);
    f += run_order_case({2, 3, 5, 7}, 9000, true);
    f += run_order_case({}, 9000, true);
    printf("%d failure(s)\n", f);
    return f ? 1 : 0;
}
