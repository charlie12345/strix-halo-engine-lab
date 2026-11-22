#pragma once

#include "ops/head/screened_head.hpp"

#include <array>
#include <string>
#include <string_view>
#include <utility>

namespace b7_fixture {
namespace head = strix::ops::screened_head;

struct Case {
    const char* name;
    head::Plan plan;
    std::vector<std::uint16_t> weights;
    std::vector<float> activation;
};

inline std::uint16_t bf16(float value) {
    return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16U);
}

inline Case make_case(const char* name, unsigned width, unsigned vocab, unsigned first, float x = 1.0F) {
    const auto plan = head::Plan::make(width, vocab, first).value();
    return {name, plan, std::vector<std::uint16_t>(plan.weight_count()), std::vector<float>(width, x)};
}

inline void fill_row(Case& test, unsigned row, float value) {
    std::fill_n(test.weights.data() + std::size_t{row} * test.plan.width(), test.plan.width(), bf16(value));
}

inline std::vector<Case> cases() {
    const auto inf = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    auto separated = make_case("pruning-ties", 264, 67, 13);
    for (unsigned row = 0; row < separated.plan.vocab(); ++row) {
        fill_row(separated, row, static_cast<float>(row % 5U) - 3.0F);
    }
    for (const unsigned row : {2U, 12U, 34U, 66U}) fill_row(separated, row, 8.0F);
    fill_row(separated, 33, 7.0F);

    auto order = make_case("incumbent-order", 264, 3, 49);
    order.weights[0] = bf16(0x1p+24F);
    order.weights[8] = bf16(1.0F);
    order.weights[256] = bf16(-0x1p+24F);
    order.weights[264] = bf16(0.5F);

    auto cancellation = make_case("cancellation", 520, 79, 103);
    for (unsigned col = 0; col < cancellation.plan.width(); ++col) {
        cancellation.activation[col] = std::ldexp(col % 2U == 0 ? 1.0F : -1.0F,
                                                 static_cast<int>(col % 49U) - 24);
    }
    for (unsigned row = 0; row < cancellation.plan.vocab(); ++row) {
        for (unsigned col = 0; col < cancellation.plan.width(); ++col) {
            auto bits = (row + 1U) * 0x45d9f3bU ^ (col + 1U) * 0x9e3779b9U;
            bits ^= bits >> 16U;
            const auto n = static_cast<int>(bits % 511U) - 255;
            cancellation.weights[std::size_t{row} * cancellation.plan.width() + col] =
                bf16(std::ldexp(static_cast<float>(n), static_cast<int>(bits % 35U) - 25));
        }
    }

    auto subnormal = make_case("subnormals", 264, 35, 23, std::numeric_limits<float>::denorm_min());
    fill_row(subnormal, 2, 0.5F);
    fill_row(subnormal, 4, -0.5F);
    fill_row(subnormal, 8, 2.0F);
    fill_row(subnormal, 34, 1.0F);
    subnormal.weights[3] = 1U;  // smallest positive BF16 subnormal

    auto overflow = make_case("overflow-and-nonfinite", 264, 65, 1, 0x1p+122F);
    fill_row(overflow, 1, 0x1p+11F);  // product overflows
    fill_row(overflow, 2, 0x1p+1F);   // products fit; the butterfly sum overflows
    fill_row(overflow, 11, 0x1p-3F);  // finite winner
    fill_row(overflow, 60, 0x1p-4F);  // finite runner-up from a different tile
    overflow.weights[264U * 3U] = bf16(inf);
    overflow.weights[264U * 64U] = bf16(nan);

    auto all_bad = make_case("nonfinite-activation", 8, 3, 4);
    all_bad.activation[3] = nan;
    all_bad.activation[7] = -inf;

    auto singleton = make_case("singleton-and-sentinel", 8, 2, head::ha::kNone - 2U);
    fill_row(singleton, 0, inf);
    fill_row(singleton, 1, -0.0F);

    std::vector<Case> result;
    result.push_back(std::move(separated));
    result.push_back(std::move(order));
    result.push_back(std::move(cancellation));
    result.push_back(std::move(subnormal));
    result.push_back(std::move(overflow));
    result.push_back(std::move(all_bad));
    result.push_back(std::move(singleton));
    return result;
}

// Independent full-score CPU reference following the published incumbent
// lane assignment, non-fused rounding and XOR tree.
inline float incumbent_score(std::span<const std::uint16_t> weights, std::span<const float> x) {
    std::array<float, 32> lanes{};
    for (std::size_t col = 0; col < x.size(); ++col) {
        const float product = head::widen_bf16(weights[col]) * x[col];
        lanes[(col / 8) % lanes.size()] += product;
    }
    for (std::size_t offset = 16; offset != 0; offset /= 2) {
        for (std::size_t lane = 0; lane < offset; ++lane) lanes[lane] += lanes[lane + offset];
    }
    return lanes[0];
}

inline bool same(head::Result lhs, head::Result rhs) {
    return lhs.id == rhs.id && lhs.runner_up == rhs.runner_up && lhs.nonfinite == rhs.nonfinite &&
           std::bit_cast<std::uint32_t>(lhs.score) == std::bit_cast<std::uint32_t>(rhs.score);
}

inline void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

}  // namespace b7_fixture
