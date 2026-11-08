#pragma once
#include "glm/prefill/glm_pf_workspace.hpp"
#include <limits>
#include <vector>

namespace strix::glm::pf {
constexpr std::uint64_t kvshard_round_bytes(std::uint64_t bytes) {
    constexpr std::uint64_t granule = std::uint64_t{2} << 20;
    return (bytes + granule - 1) / granule * granule;
}
constexpr unsigned kvshard_hash_slots(unsigned latents) {
    unsigned slots = 1;
    while (slots < 2 * latents) slots *= 2;
    return slots;
}
// Unified-memory peak, including host storage: three BF16 device images,
// bulk result + frame + retained receive frame, two device/pinned upload slots,
// two pinned selection pages, one live planner and its half-full hash table.
// Every owner rounds separately; 4 MiB covers small control/runtime allocations.
constexpr std::uint64_t kvshard_prefill_bytes(unsigned latents, unsigned rows, unsigned stride) {
    const auto image = std::uint64_t{latents} * 512 * sizeof(std::uint16_t);
    const auto tokens = std::uint64_t{rows} * stride * sizeof(std::uint32_t);
    const auto ids = std::uint64_t{latents} * sizeof(std::uint32_t);
    return 4 * kvshard_round_bytes(image) + 2 * kvshard_round_bytes(image + 9 * sizeof(std::uint64_t)) +
        4 * kvshard_round_bytes(ids + tokens) +
        2 * kvshard_round_bytes(tokens + std::uint64_t{rows} * sizeof(std::uint32_t)) +
        kvshard_round_bytes(ids) + kvshard_round_bytes(tokens) +
        kvshard_round_bytes(std::uint64_t{kvshard_hash_slots(latents)} * sizeof(std::uint32_t)) +
        (std::uint64_t{4} << 20);
}
struct KvShardPrefillLimits { unsigned latents, rows; std::uint64_t bytes; };
inline KvShardPrefillLimits kvshard_prefill_limits(std::uint64_t budget, std::uint64_t capacity,
                                                  unsigned rows, unsigned stride) {
    if (capacity == 0 || capacity > std::numeric_limits<std::uint32_t>::max() || rows == 0 ||
        stride == 0 || stride > 2051)
        throw std::invalid_argument("kvshard: invalid prefill budget geometry");
    budget = std::min(budget, kKvShardPrefillMaximumBytes);
    const unsigned target_rows = std::min(rows, kKvShardPrefillRows);
    unsigned lo = static_cast<unsigned>(std::min<std::uint64_t>(capacity, stride));
    unsigned hi = static_cast<unsigned>(std::min(capacity, std::uint64_t{target_rows} * stride));
    if (kvshard_prefill_bytes(lo, target_rows, stride) > budget)
        throw std::invalid_argument("kvshard: insufficient prefill union byte budget");
    // First fund up to 128 even completely disjoint rows; then spend remaining
    // bytes on more rows. Actual unions stop only at this admitted latent bound.
    while (lo < hi) {
        const unsigned mid = lo + (hi - lo + 1) / 2;
        if (kvshard_prefill_bytes(mid, target_rows, stride) <= budget) lo = mid;
        else hi = mid - 1;
    }
    unsigned latents = lo;
    lo = target_rows; hi = rows;
    while (lo < hi) {
        const unsigned mid = lo + (hi - lo + 1) / 2;
        if (kvshard_prefill_bytes(latents, mid, stride) <= budget) lo = mid;
        else hi = mid - 1;
    }
    const unsigned admitted_rows = lo;
    // The target is not a latent cap either: use any remaining rounded bytes
    // for a larger union after funding the largest row page.
    lo = latents;
    hi = static_cast<unsigned>(std::min({capacity, std::uint64_t{admitted_rows} * stride, budget / 6144}));
    while (lo < hi) {
        const unsigned mid = lo + (hi - lo + 1) / 2;
        if (kvshard_prefill_bytes(mid, admitted_rows, stride) <= budget) lo = mid;
        else hi = mid - 1;
    }
    latents = lo;
    return {latents, admitted_rows, kvshard_prefill_bytes(latents, admitted_rows, stride)};
}
struct KvShardPrefillWindow {
    unsigned rows = 0;
    std::vector<std::uint32_t> latents; // global IDs, first-use order
    std::vector<std::uint32_t> tokens;  // row-major ordinals, original list order
};

// Address-only transformation: no sorting, score arithmetic or rank reduction.
// Half-full byte-admitted table: no capacity-sized lookup or new device kernel.
inline KvShardPrefillWindow kvshard_prefill_window(const std::uint32_t* tokens,
        const std::uint32_t* counts, unsigned rows, unsigned stride, std::uint64_t capacity,
        const KvShardPrefillLimits& limits) {
    if (rows == 0 || rows > limits.rows || stride == 0 || stride > 2051 || limits.latents == 0 ||
        limits.latents > kKvShardPrefillMaximumBytes / (6 * 512 * sizeof(std::uint16_t)))
        throw std::invalid_argument("kvshard: invalid prefill window geometry");
    constexpr auto empty = std::numeric_limits<std::uint32_t>::max();
    const unsigned slots = kvshard_hash_slots(limits.latents);
    std::vector<std::uint32_t> table(slots, empty);
    KvShardPrefillWindow window;
    window.latents.reserve(limits.latents);
    window.tokens.resize(static_cast<std::size_t>(rows) * stride, 0);
    for (unsigned row = 0; row < rows; ++row) {
        if (counts[row] == 0 || counts[row] > stride)
            throw std::invalid_argument("kvshard: invalid prefill selection count");
        const auto saved = window.latents.size();
        for (unsigned item = 0; item < counts[row]; ++item) {
            const auto at = static_cast<std::size_t>(row) * stride + item;
            const auto token = tokens[at];
            if (token >= capacity) throw std::invalid_argument("kvshard: prefill token outside cache");
            unsigned slot = (token * 2654435761U) & (slots - 1);
            while (table[slot] != empty && window.latents[table[slot]] != token)
                slot = (slot + 1) & (slots - 1);
            if (table[slot] == empty) {
                if (window.latents.size() == limits.latents) {
                    // The overflowing row belongs wholly to the next window.
                    if (window.rows == 0)
                        throw std::invalid_argument("kvshard: row exceeds admitted union budget");
                    window.latents.resize(saved);
                    window.tokens.resize(static_cast<std::size_t>(window.rows) * stride);
                    return window;
                }
                table[slot] = static_cast<std::uint32_t>(window.latents.size());
                window.latents.push_back(token);
            }
            window.tokens[at] = table[slot];
        }
        ++window.rows;
    }
    return window;
}
} // namespace strix::glm::pf
