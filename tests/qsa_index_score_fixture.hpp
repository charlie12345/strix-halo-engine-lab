#pragma once
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace index_fixture {
inline constexpr unsigned dim = 128U, heads = 4U, ratio = 4U, budget = 1536U, topk = budget / ratio;
// The exact scorer's range prologue reads every query in an active group;
// candidate-only dense rows deliberately retain null sentinels.
inline const float* query_for_arm(unsigned arm, unsigned complete, const float* query) {
    return arm == 0U || complete > topk ? query : nullptr;
}
inline void require(bool b, const char* what) { if (!b) throw std::runtime_error(what); }
inline double median(std::vector<float> values) {
    require(!values.empty(), "empty timing sample");
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2U;
    const double upper = values[middle];
    return values.size() % 2U != 0U ? upper : std::midpoint(static_cast<double>(values[middle - 1U]), upper);
}
inline float bf16(float x) {
    auto b = std::bit_cast<std::uint32_t>(x);
    b = (b + 0x7fffU + ((b >> 16U) & 1U)) & 0xffff0000U;
    return std::bit_cast<float>(b);
}
inline float sample(std::uint32_t i) {
    i = (i ^ (i >> 16U)) * 0x7feb352dU;
    i = (i ^ (i >> 15U)) * 0x846ca68bU;
    i ^= i >> 16U;
    return bf16(static_cast<float>(static_cast<int>(i & 65535U) - 32768) / 18918.6F);
}
// Independent host expression of the ordered FP32 production tree. Disable FMA
// contraction when compiling this fixture, as for the engine reference.
inline float fp32_score(const float* q, const float* k) {
    float total = 0.0F;
    for (unsigned h = 0U; h < heads; ++h) {
        float leaf[32];
        for (unsigned l = 0U; l < 32U; ++l) {
            const float p0 = q[h * dim + l] * k[l], p1 = q[h * dim + l + 32U] * k[l + 32U];
            const float p2 = q[h * dim + l + 64U] * k[l + 64U], p3 = q[h * dim + l + 96U] * k[l + 96U];
            leaf[l] = (p0 + p2) + (p1 + p3);
        }
        for (unsigned s = 16U; s; s /= 2U)
            for (unsigned l = 0U; l < s; ++l) leaf[l] += leaf[l + s];
        total += std::fmax(leaf[0], 0.0F);
    }
    return total / std::sqrt(static_cast<float>(dim));
}
// FP64 panel oracle: tests indexing/semantics, not native WMMA instruction bits.
inline float panel_score(const float* q, const float* k) {
    float total = 0.0F;
    for (unsigned h = 0U; h < heads; ++h) {
        float dot = 0.0F;
        for (unsigned f = 0U; f < dim; f += 16U) {
            double panel = 0.0;
            for (unsigned j = 0U; j < 16U; ++j) panel += static_cast<double>(q[h * dim + f + j]) * k[f + j];
            dot += static_cast<float>(panel);
        }
        total += std::fmax(dot, 0.0F);
    }
    return total / std::sqrt(static_cast<float>(dim));
}
inline std::vector<unsigned> selected_blocks(const float* score, unsigned blocks) {
    std::vector<unsigned> ids(blocks);
    std::iota(ids.begin(), ids.end(), 0U);
    if (blocks > topk) {
        std::nth_element(ids.begin(), ids.begin() + topk, ids.end(), [&](unsigned a, unsigned b) {
            return score[a] != score[b] ? score[a] > score[b] : a < b;
        });
        ids.resize(topk);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}
inline unsigned symmetric_difference(const std::vector<unsigned>& a, const std::vector<unsigned>& b) {
    unsigned n = 0U;
    auto i = a.begin(), j = b.begin();
    while (i != a.end() && j != b.end()) {
        if (*i == *j) { ++i; ++j; }
        else if (*i < *j) { ++n; ++i; }
        else { ++n; ++j; }
    }
    return n + static_cast<unsigned>(a.end() - i) + static_cast<unsigned>(b.end() - j);
}
struct SelectionChanges {
    unsigned rows = 0U, changed = 0U, maximum = 0U;
    std::uint64_t sum = 0U;
    void add(const std::vector<unsigned>& a, const std::vector<unsigned>& b) {
        const auto n = symmetric_difference(a, b);
        ++rows; changed += n != 0U; maximum = std::max(maximum, n); sum += n;
    }
    double mean() const { return rows ? static_cast<double>(sum) / rows : 0.0; }
};
} // namespace index_fixture
