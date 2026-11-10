#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strix::core {
struct StoredI4Plan {
    std::size_t layers = 0;
    std::size_t free_after = 0;  // excludes the owner's reserve
};
// KV, context, slots and workspace already belong to the owner. A conversion must
// fit BOTH the old layer and its new pack; only completed packs reclaim EXL3.
inline StoredI4Plan plan_stored_i4(std::size_t free_bytes, std::size_t reserve,
                                  std::size_t pack, std::size_t tables,
                                  const std::vector<std::size_t>& reclaim,
                                  std::size_t limit = std::numeric_limits<std::size_t>::max()) {
    if (!pack || tables > std::numeric_limits<std::size_t>::max() - pack)
        throw std::invalid_argument("invalid stored INT4 pack size");
    StoredI4Plan p{0, free_bytes > reserve ? free_bytes - reserve : 0};
    const auto build = pack + tables;
    for (std::size_t i = 0; i < std::min(limit, reclaim.size()); ++i) {
        if (build > p.free_after) break;
        const auto remaining = p.free_after - build;
        if (reclaim[i] > std::numeric_limits<std::size_t>::max() - remaining)
            throw std::overflow_error("stored INT4 reclaimed bytes overflow");
        p.free_after = remaining + reclaim[i];
        ++p.layers;
    }
    return p;
}
}  // namespace strix::core
