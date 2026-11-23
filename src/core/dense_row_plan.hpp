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

// A rejected tall shape can use the same FP32 reduction order on a row grid. `terms` is the activation plane count
// of the tall split (3 exact mma3, 2 / 1 FAST mma2 / mma1). The compact tile is three-term only, so fewer terms
// take plain big.
[[nodiscard]] constexpr Bf16RowPath plan_bf16_row_projection(
    unsigned in, unsigned out, unsigned rows, bool tall_mma, Bf16RowLimits limits, int terms = 3) noexcept {
    if (!tall_mma || terms < 1 || terms > 3 || rows <= limits.short_rows || in == 0U || out == 0U)
        return Bf16RowPath::short_rows;
    if (limits.tall_input_alignment != 0U && limits.tall_output_alignment != 0U &&
        in % limits.tall_input_alignment == 0U && out % limits.tall_output_alignment == 0U)
        return out <= limits.compact_outputs && terms == 3 ? Bf16RowPath::big_compact : Bf16RowPath::big;
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

// Value class of a dense activation operand, declared by the caller that produced it.
// bf16_exact: every element is a finite FP32 value exactly representable in BF16 (the
// producer stored RN_bf16 of its result, or widened a BF16 tensor).
enum class Bf16Operand : unsigned char { fp32, bf16_exact };

// Activation planes the exact BF16 split needs. The split writes hi = RN_bf16(x),
// mid = RN_bf16(x - hi), lo = RN_bf16(x - hi - mid), and each K16 step of the tall
// chain issues lo, mid, hi into one accumulator that starts at +0. For a BF16-exact
// finite x, mid and lo are +0, so the two leading MMAs add only zero products to an
// accumulator that is never -0: one plane (hi) gives the same bits. This is an EXACT
// schedule of the three-term arithmetic, not a FAST operand class.
constexpr int exact_bf16_planes(Bf16Operand operand) {
    return operand == Bf16Operand::bf16_exact ? 1 : 3;
}

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
