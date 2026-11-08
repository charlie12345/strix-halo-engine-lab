#pragma once

#include <cstdint>
#include <limits>

namespace strix::ple_tile {

// Organization only: one packed input tile, two unchanged projection consumers.
// Keep this admission contract HIP-free so refusals can be checked on a host.
inline bool requested(const char* value) noexcept {
    return value && value[0] == '1' && value[1] == '\0';
}

struct Matrix {
    std::uintptr_t address = 0;
    std::uint32_t columns = 0, stride = 0;
};

struct Range {
    std::uintptr_t first = 0, last = 0;  // Half-open; invalid ranges are empty.
};

inline Range extent(Matrix matrix, std::uint32_t rows, unsigned element_bytes) noexcept {
    if (!matrix.address || !rows || !matrix.columns || matrix.stride < matrix.columns || !element_bytes)
        return {};
    const auto elements = static_cast<std::uint64_t>(rows - 1U) * matrix.stride + matrix.columns;
    if (elements > (std::numeric_limits<std::uintptr_t>::max() - matrix.address) / element_bytes)
        return {};
    return {matrix.address, matrix.address + static_cast<std::uintptr_t>(elements * element_bytes)};
}

inline bool disjoint(Range a, Range b) noexcept {
    return a.last > a.first && b.last > b.first && (a.last <= b.first || b.last <= a.first);
}

// The caller still applies both original lane/batch admissions, including recipe,
// weight alignment, wide/batch enablement and baseline suppression. Bounding pitched
// ranges conservatively refuses aliases, including aliases through row padding.
inline bool admits(bool enabled, Matrix input, Matrix key, Matrix value, std::uint32_t groups) noexcept {
    if (!enabled || !groups || groups > 65535U || input.columns % 512U ||
        input.columns > UINT32_MAX / 64U || (input.address | (std::uint64_t{input.stride} * 2U)) % 16U)
        return false;
    const auto rows = groups * 64U;
    const auto in = extent(input, rows, 2U), k = extent(key, rows, 4U), v = extent(value, rows, 4U);
    return disjoint(in, k) && disjoint(in, v) && disjoint(k, v);
}

}  // namespace strix::ple_tile
