#pragma once
// solver-interface.h: universal solver contract for ACE-Step flow matching
//
// Every solver is a pure function of (xt, vt, timesteps, state, model_fn).
// The sampler loop resolves the solver by name and calls it once per step.
//
// Solver interface mirrors the Python registry:
//   solver_step(xt, vt, t_curr, t_prev, n, state, model_fn, vt_buf)
//
// Multi-evaluation solvers (Heun, RK4, RK5, DOPRI5, DOP853, JKASS Quality)
// use model_fn to re-evaluate the DiT at intermediate timesteps.
// Single-evaluation solvers (Euler, DPM, JKASS Fast, STORK) ignore model_fn.

#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

// ---------------------------------------------------------------------------
// Model callback for multi-evaluation solvers
// ---------------------------------------------------------------------------

// Evaluates the DiT model at (xt_in, t_val) and writes the CFG-processed
// velocity into the internal vt buffer owned by the sampler loop.
// After calling model_fn, the caller reads the result from vt_buf (which the
// sampler loop ensures points to the same storage as vt in dit-sampler.h).
//
// The lambda captures all GPU state, encoder tensors, masks, etc. The solver
// never needs to know about ggml, CUDA, attention, or model architecture.
using SolverModelFn = std::function<void(const float *, float)>;

// ---------------------------------------------------------------------------
// Persistent solver state (survives across steps within a single generation)
// ---------------------------------------------------------------------------

struct SolverState {
    int step_index = 0;

    // ── DPM++ 2M / 3M: velocity history ──────────────────────────
    std::vector<float> prev_vt;       // [n] previous step's velocity
    std::vector<float> prev_prev_vt;  // [n] two steps ago (DPM++ 3M only)
    float              prev_dt = 0.0f; // previous step size (DPM++ 2M Adaptive)

    // ── JKASS Fast: momentum blending ────────────────────────────
    std::vector<float> prev_delta;          // [n] previous velocity delta
    float              beat_stability    = 0.0f;
    float              frequency_damping = 0.0f;
    float              temporal_smoothing = 0.0f;

    // ── STORK: velocity history with step sizes ──────────────────
    struct VelocityRecord {
        std::vector<float> vt;
        float              dt;
    };
    std::vector<VelocityRecord> velocity_history;
    int                         stork_substeps = 10;

    // ── SDE: per-batch seeds for Philox re-noising ───────────────
    const int64_t * seeds    = nullptr;
    // Seed breeding (SA3 /sa3-synth): exposed to Lua solvers as globals
    // seed_b / seed_blend so stochastic solvers can mix a lockstep B stream.
    int64_t         seed_b     = -1;
    float           seed_blend = 0.0f;
    int             batch_n  = 1;
    int             n_per    = 0;  // elements per batch item (T * Oc)

    // ── UniPC: data prediction history ──────────────────────────
    struct UniPCRecord {
        std::vector<float> model_output;  // [n] data prediction (x0_hat)
        float              t;             // timestep at which this was computed
    };
    std::vector<UniPCRecord> unipc_history;  // last N data predictions (max order)

    // ── A-FloPS: residual velocity history ──────────────────────
    std::vector<float> aflops_prev_w;      // [n] previous step's residual velocity
    float              aflops_prev_t     = 0.0f; // source timestep of previous w
    float              aflops_prev_t_dst = 0.0f; // destination timestep of previous step

    // ── Scratch buffer for multi-eval solvers ────────────────────
    // Pre-allocated by the sampler loop, sized [n_total].
    // Multi-eval solvers (RK4, Heun, etc.) use this for intermediate xt.
    std::vector<float> xt_scratch;
};

// ---------------------------------------------------------------------------
// Solver step function signature
// ---------------------------------------------------------------------------

// xt:       [n] current latent — modified IN-PLACE to xt_next
// vt:       [n] velocity at (xt, t_curr), already CFG/APG-processed
// t_curr:   current timestep (1.0 = pure noise)
// t_prev:   next timestep (0.0 = clean signal)
// n:        total elements across all batches (batch_n * T * Oc)
// state:    mutable solver state (persists across steps)
// model_fn: callback for re-evaluation (may be empty for 1-NFE solvers)
// vt_buf:   [n] the sampler's vt buffer — model_fn writes results here.
//           After calling model_fn(xt_tmp, t_val), read the result from vt_buf.
using SolverStepFn = void (*)(float *       xt,
                              const float * vt,
                              float         t_curr,
                              float         t_prev,
                              int           n,
                              SolverState & state,
                              SolverModelFn model_fn,
                              float *       vt_buf);
