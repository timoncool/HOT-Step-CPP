#pragma once
// Least-recently-used cache of per-shape device buffers (S2b). Backend-free so
// tests/test-yue2-shape-cache.cpp can drive it with a fake allocator.
#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

namespace yue2_aitk_shape_cache {

// E carries `size_t bytes`. Returns the newest entry matching `match`;
// otherwise evicts oldest entries until `need` more bytes fit both the entry
// cap and `budget`, then calls `build` (returns null on failure). If the build
// fails while entries are still cached, every cached entry is freed and the
// build runs once more, so caching never makes a shape fail that would fit on
// its own.
template <class E, class Match, class Build>
E * fetch(std::vector<std::unique_ptr<E>> & cache, size_t max_entries, size_t budget,
          size_t need, Match match, Build build) {
    for (size_t i = 0; i < cache.size(); ++i) {
        if (match(*cache[i])) {
            std::rotate(cache.begin() + long(i), cache.begin() + long(i) + 1, cache.end());
            return cache.back().get();
        }
    }
    size_t used = 0;
    for (const auto & e : cache) used += e->bytes;
    while (!cache.empty() && (cache.size() >= max_entries || used + need > budget)) {
        used -= cache.front()->bytes;
        cache.erase(cache.begin());
    }
    std::unique_ptr<E> e = build();
    if (!e && !cache.empty()) { cache.clear(); e = build(); }
    if (!e) return nullptr;
    e->bytes = need;
    cache.push_back(std::move(e));
    return cache.back().get();
}

} // namespace yue2_aitk_shape_cache
