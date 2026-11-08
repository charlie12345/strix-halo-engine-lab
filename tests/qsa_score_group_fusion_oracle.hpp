#pragma once
// Host-only organization model. Scores are final idxscore16 candidate FP32 values;
// this model does not emulate a WMMA instruction or qualify its arithmetic.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace qsa_group_oracle {
inline constexpr unsigned ratio = 4, budget = 1536, topk = budget / ratio, group = 4;
inline std::uint32_t key(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7fffffffU) == 0) bits = 0;
    return bits & 0x80000000U ? ~bits : bits | 0x80000000U;
}
struct Entry { float score; unsigned block; };
inline bool before(const Entry& a, const Entry& b) {
    return key(a.score) != key(b.score) ? key(a.score) > key(b.score) : a.block < b.block;
}
struct Plan {
    std::vector<float> maxima;
    std::vector<unsigned> groups; // ascending group ID, at most topk
    bool dense = false;
    bool fallback = false; // nonfinite/negative input: use unchanged materialized path
};
template<class ScoreAt> Plan plan(unsigned visible, ScoreAt score_at) {
    Plan p;
    const unsigned complete = visible / ratio;
    if (complete <= topk) { p.dense = true; return p; }
    const unsigned groups = (complete + group - 1) / group;
    std::vector<Entry> ranked;
    for (unsigned g = 0; g < groups; ++g) {
        std::array<float, group> s;
        s.fill(-INFINITY);
        for (unsigned j = 0; j < group && g * group + j < complete; ++j) {
            s[j] = score_at(g * group + j);
            p.fallback = p.fallback || !std::isfinite(s[j]) || s[j] < 0;
        }
        // Candidate lane 4g's XOR-1, XOR-2 maxima tree, padding with -inf.
        const float maximum = std::fmax(std::fmax(s[0], s[1]), std::fmax(s[2], s[3]));
        p.maxima.push_back(maximum);
        ranked.push_back({maximum, g});
    }
    if (p.fallback) return p;
    std::sort(ranked.begin(), ranked.end(), before);
    if (ranked.size() > topk) ranked.resize(topk);
    for (const auto& e : ranked) p.groups.push_back(e.block);
    std::sort(p.groups.begin(), p.groups.end());
    return p;
}
template<class ScoreAt> std::vector<Entry> rescore(const Plan& p, unsigned visible, ScoreAt score_at) {
    if (p.fallback) throw std::invalid_argument("materialized fallback required");
    std::vector<Entry> compact;
    for (const auto g : p.groups)
        for (unsigned j = 0; j < group && g * group + j < visible / ratio; ++j)
            compact.push_back({score_at(g * group + j), g * group + j});
    return compact;
}
inline std::vector<unsigned> tokens(unsigned visible, std::vector<Entry> compact) {
    const auto complete = visible / ratio;
    if (complete <= topk) {
        std::vector<unsigned> dense(visible);
        std::iota(dense.begin(), dense.end(), 0U);
        return dense;
    }
    std::sort(compact.begin(), compact.end(), before);
    if (compact.size() < topk) throw std::invalid_argument("insufficient candidates");
    compact.resize(topk);
    std::sort(compact.begin(), compact.end(), [](const Entry& a, const Entry& b) { return a.block < b.block; });
    std::vector<unsigned> out;
    for (const auto& e : compact)
        for (unsigned j = 0; j < ratio; ++j) out.push_back(e.block * ratio + j);
    for (unsigned t = complete * ratio; t < visible; ++t) out.push_back(t);
    return out;
}
} // namespace qsa_group_oracle
