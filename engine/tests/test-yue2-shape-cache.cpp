// Self-check for yue2_aitk_shape_cache::fetch under allocation pressure.
// Header-only, no CUDA: cl /std:c++17 /EHsc /I..\src\train test-yue2-shape-cache.cpp
#include "yue2-aitk-shape-cache.h"
#include <cassert>
#include <cstdio>

using yue2_aitk_shape_cache::fetch;

// A fake device: entries hold `bytes` of it until destroyed.
static size_t g_live = 0, g_capacity = 0;
struct Entry {
    size_t bytes = 0, held = 0; int key = 0;
    ~Entry() { g_live -= held; }
};
using Cache = std::vector<std::unique_ptr<Entry>>;

static Entry * get(Cache & cache, int key, size_t need, size_t budget = 1000, size_t cap = 8) {
    return fetch(cache, cap, budget, need,
        [&](const Entry & e) { return e.key == key; },
        [&]() -> std::unique_ptr<Entry> {
            if (g_live + need > g_capacity) return nullptr;  // out of device memory
            auto e = std::make_unique<Entry>(); e->key = key; e->held = need; g_live += need;
            return e;
        });
}

int main() {
    // A hit returns the same entry and makes it the newest.
    { g_capacity = 1000; Cache c;
      Entry * a = get(c, 1, 100); get(c, 2, 100);
      assert(get(c, 1, 100) == a && c.back().get() == a); }
    assert(g_live == 0);
    // The byte budget evicts the oldest before allocating.
    { g_capacity = 1000; Cache c;
      get(c, 1, 300); get(c, 2, 300); get(c, 3, 300);
      assert(get(c, 4, 300, 1000) && c.size() == 3 && c.front()->key == 2 && g_live == 900); }
    // The entry cap holds regardless of bytes.
    { g_capacity = 1000; Cache c;
      for (int k = 0; k < 5; ++k) get(c, k, 10, 1000, 3);
      assert(c.size() == 3 && c.front()->key == 2); }
    // Device memory full of other allocations: the budget alone admits the new
    // shape, the first build fails, the cache is freed and the retry succeeds.
    { g_capacity = 500; Cache c;
      get(c, 1, 200); get(c, 2, 200);
      Entry * big = get(c, 3, 400, 10000);
      assert(big && c.size() == 1 && c.back()->key == 3 && g_live == 400); }
    // A shape that cannot fit even alone fails cleanly with an empty cache.
    { g_capacity = 500; Cache c;
      get(c, 1, 200);
      assert(!get(c, 2, 600, 10000) && c.empty() && g_live == 0); }
    assert(g_live == 0);
    std::puts("test-yue2-shape-cache: ok");
    return 0;
}
