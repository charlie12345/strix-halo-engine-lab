#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace strix::core {

enum class Bf16RowPath { short_rows, fp32_grid, big, big_compact };

struct Bf16RowLimits {
    unsigned short_rows;
    unsigned grid_rows;
    unsigned compact_outputs;
    unsigned tall_input_alignment;
    unsigned tall_output_alignment;
};

// A rejected tall shape can use the same FP32 reduction order on a row grid.
[[nodiscard]] constexpr Bf16RowPath plan_bf16_row_projection(
    unsigned in, unsigned out, unsigned rows, bool tall_mma3, Bf16RowLimits limits) noexcept {
    if (!tall_mma3 || rows <= limits.short_rows || in == 0U || out == 0U)
        return Bf16RowPath::short_rows;
    if (limits.tall_input_alignment != 0U && limits.tall_output_alignment != 0U &&
        in % limits.tall_input_alignment == 0U && out % limits.tall_output_alignment == 0U)
        return out <= limits.compact_outputs ? Bf16RowPath::big_compact : Bf16RowPath::big;
    return rows <= limits.grid_rows ? Bf16RowPath::fp32_grid : Bf16RowPath::short_rows;
}

// Resident BF16 weights [output][input] and tight FP32 output [row][output].
struct Bf16RowProjection {
    const std::uint16_t* weights{};
    unsigned outputs{};
    float* output{};
    bool round_bf16{};  // Round the completed FP32 sum and store it widened to FP32.
    std::size_t row_bytes{};  // Zero uses input_width * sizeof(std::uint16_t).

    constexpr bool contiguous(unsigned input_width) const {
        return row_bytes == 0U ||
            (row_bytes % sizeof(std::uint16_t) == 0U && row_bytes / sizeof(std::uint16_t) == input_width);
    }
};

template <auto Limit, class Size, class Status, class Launch>
Status for_each_dense_chunk(Size total, Status success, const Launch& launch) {
    static_assert(Limit > 0);
    constexpr Size limit = Limit;
    for (Size first = 0; first < total;) {
        const Size count = std::min(total - first, limit);
        if (const auto status = launch(first, count); status != success) return status;
        first += count;
    }
    return success;
}

template <std::size_t Limit, class Status, class Launch>
Status for_each_bf16_projection_group(std::span<const Bf16RowProjection> projections, Status success,
                                      const Launch& launch) {
    return for_each_dense_chunk<Limit>(projections.size(), success, [&](std::size_t first, std::size_t count) {
        return launch(projections.subspan(first, count));
    });
}

}  // namespace strix::core
