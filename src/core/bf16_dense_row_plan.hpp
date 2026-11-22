#pragma once

#include "core/dense_row_plan.hpp"
#include "strix/core/rows.hpp"

#include <limits>

namespace strix::cuda_core {

inline constexpr std::size_t kMaxBf16RowProjections = 4;

struct Bf16DenseRowGroup {
    const float* input;
    unsigned input_width;
    std::span<const core::Bf16RowProjection> projections;
};

namespace detail {

inline bool bf16_mma3_tile(unsigned in, unsigned out) {
    return in % core::rows::kBlockIn == 0 && out % core::rows::kTileOut == 0;
}

inline bool valid_bf16_dense_row_group(const Bf16DenseRowGroup& group, unsigned rows, bool allow_round_bf16) {
    const auto& [input, input_width, projections] = group;
    const auto aligned = [](const void* pointer) {
        return pointer && reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0;
    };
    constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!aligned(input) || input_width == 0 || input_width % 8U != 0 || rows == 0 ||
        input_width > max_floats / rows || projections.empty() || projections.size() > kMaxBf16RowProjections)
        return false;

    const bool round_bf16 = projections.front().round_bf16;
    if (round_bf16 && !allow_round_bf16) return false;
    std::uint64_t total_outputs = 0;
    for (const auto& projection : projections) {
        if (!projection.contiguous(input_width) || projection.round_bf16 != round_bf16 ||
            (round_bf16 && !bf16_mma3_tile(input_width, projection.outputs)) ||
            projection.outputs == 0 || projection.outputs > max_floats / rows ||
            !aligned(projection.weights) || !aligned(projection.output) || projection.output == input ||
            projection.outputs > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) / input_width)
            return false;
        total_outputs += projection.outputs;
    }
    return total_outputs <= std::numeric_limits<unsigned>::max();
}

}  // namespace detail

}  // namespace strix::cuda_core
