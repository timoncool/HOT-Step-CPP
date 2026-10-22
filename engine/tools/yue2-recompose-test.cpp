// Issue #234: a semantic redraw of a song that brought its own score must keep
// it, or a lyric_schedule redraw is refused with "needs a supplied score".
// Uses the pipeline's own state, copy and guard. No GPU, no models.
#include "yue2/yue2-pipeline.h"

#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    // songs[0] as yue2_pipeline_run_ar seeds it from a request with an abc.
    Yue2SongState sg;
    sg.src_song   = 2;
    sg.style      = "rock";
    sg.lyrics     = "[verse]";
    sg.abc_text   = "X:1\nK:C\nCDEF|";
    sg.abc_given  = true;
    sg.abc_ids    = { 1, 2, 3 };
    sg.score_abc  = sg.abc_text;
    sg.noise_seed = 7;

    std::vector<Yue2SongState> draws(1);
    yue2_recompose_copy(draws[0], sg);
    const Yue2SongState & ds = draws[0];

    int fail = 0;
    if (!ds.abc_given || ds.abc_text != sg.abc_text) {
        printf("FAIL redraw lost the supplied score (abc_given=%d, abc_text %zu bytes)\n", (int) ds.abc_given,
               ds.abc_text.size());
        fail++;
    }
    if (ds.abc_ids != sg.abc_ids || ds.score_abc != sg.score_abc || ds.style != sg.style || ds.lyrics != sg.lyrics ||
        ds.src_song != sg.src_song || ds.noise_seed != sg.noise_seed) {
        printf("FAIL redraw lost another field\n");
        fail++;
    }

    // The guard the semantic stage applies, on the redraw batch.
    const char * why = yue2_lyric_schedule_refusal(false, true, draws);
    printf("redraw guard: %s\n", why ? why : "ok");
    if (why) fail++;

    // And the guard still refuses a draw with no score.
    std::vector<Yue2SongState> bare(1);
    why = yue2_lyric_schedule_refusal(false, true, bare);
    if (!why || !strstr(why, "supplied score")) {
        printf("FAIL guard let a draw without a score through\n");
        fail++;
    }

    printf(fail ? "yue2-recompose-test: FAIL (%d)\n" : "yue2-recompose-test: PASS\n", fail);
    return fail ? 1 : 0;
}
