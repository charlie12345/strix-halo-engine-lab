#pragma once
// Host mathematical oracle only; inputs are decoded per-head scores/values.
#include "kvshard_traits.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace strix::xport::kvshard {
inline void validate(OwnerMap map) {
    if (!map.block_tokens || !map.group_tokens || map.block_tokens % map.group_tokens)
        throw std::invalid_argument("ownership must preserve complete groups");
}
inline unsigned owner(std::uint64_t token, OwnerMap map) {
    validate(map);
    return static_cast<unsigned>((token / map.block_tokens) % 2);
}
inline std::uint64_t group_start(std::uint64_t group, OwnerMap map) {
    validate(map);
    if (group > (std::numeric_limits<std::uint64_t>::max)() / map.group_tokens)
        throw std::overflow_error("group position overflow");
    return group * map.group_tokens;
}
inline std::uint64_t block_count(std::uint64_t tokens, OwnerMap map) {
    validate(map);
    return tokens / map.block_tokens + (tokens % map.block_tokens != 0 ? 1U : 0U);
}
inline void finite(float x) {
    if (!std::isfinite(x)) throw std::invalid_argument("nonfinite oracle input");
}
inline float checked(float x) {
    if (!std::isfinite(x)) throw std::overflow_error("FP32 partial overflow");
    return x;
}
struct Candidate {
    std::uint64_t id, visible_at; // complete pool/group visibility supplied by caller
    float selection_score;
};
// Global candidate set must already reflect the engine's sparse/candidate policy.
// Selection scores are distinct from the per-head attention scores below.
inline std::vector<std::uint64_t> select_global(std::vector<Candidate> candidates,
                                              std::uint64_t query, std::size_t k) {
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        finite(candidates[i].selection_score);
        if (i && candidates[i - 1].id == candidates[i].id)
            throw std::invalid_argument("duplicate candidate ID");
    }
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
        [query](const auto& c) { return c.visible_at > query; }), candidates.end());
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.selection_score == b.selection_score ? a.id < b.id : a.selection_score > b.selection_score;
    });
    if (candidates.size() > k) candidates.resize(k);
    std::vector<std::uint64_t> ids;
    for (const auto& c : candidates) ids.push_back(c.id);
    std::sort(ids.begin(), ids.end());
    return ids;
}
struct Row {
    std::uint64_t id, token; // global row ID and original token/group-start position
    std::optional<std::uint64_t> candidate; // null = mandatory tail/window/dense row
    float score;
    std::vector<float> value;
};
struct Partial {
    float m = -std::numeric_limits<float>::infinity(), l = 0;
    std::vector<float> a;
};
inline void validate(const Partial& p) {
    if (p.a.empty()) throw std::invalid_argument("zero partial width");
    if (p.l == 0) {
        if (p.m != -std::numeric_limits<float>::infinity() ||
            std::any_of(p.a.begin(), p.a.end(), [](float x) { return x != 0; }))
            throw std::invalid_argument("noncanonical empty partial");
    } else {
        finite(p.m); finite(p.l);
        if (p.l < 0) throw std::invalid_argument("negative denominator");
        for (float x : p.a) finite(x);
    }
}
// Stable two-pass FP32 reduction; row order is ascending global ID.
inline Partial accumulate(const std::vector<Row>& rows, std::size_t width,
                          std::optional<float> sink = std::nullopt) {
    if (!width) throw std::invalid_argument("zero value width");
    Partial p; p.a.resize(width, 0);
    if (sink) { finite(*sink); p.m = *sink; }
    for (const auto& r : rows) {
        finite(r.score);
        if (r.value.size() != width) throw std::invalid_argument("value width mismatch");
        for (float x : r.value) finite(x);
        p.m = (std::max)(p.m, r.score);
    }
    if (rows.empty() && !sink) return p;
    for (const auto& r : rows) {
        const float w = std::exp(r.score - p.m);
        p.l = checked(p.l + w);
        for (std::size_t j = 0; j < width; ++j)
            p.a[j] = checked(p.a[j] + checked(w * r.value[j]));
    }
    if (sink) p.l = checked(p.l + std::exp(*sink - p.m)); // denominator only
    return p;
}
inline Partial merge_rank0_rank1(const Partial& rank0, const Partial& rank1) {
    validate(rank0); validate(rank1);
    if (rank0.a.size() != rank1.a.size()) throw std::invalid_argument("partial width mismatch");
    if (rank0.l == 0) return rank1;
    if (rank1.l == 0) return rank0;
    Partial p; p.m = (std::max)(rank0.m, rank1.m); p.a.resize(rank0.a.size());
    const float w0 = std::exp(rank0.m - p.m), w1 = std::exp(rank1.m - p.m);
    p.l = checked(rank0.l * w0 + rank1.l * w1);
    for (std::size_t j = 0; j < p.a.size(); ++j)
        p.a[j] = checked(rank0.a[j] * w0 + rank1.a[j] * w1);
    return p;
}
inline std::vector<float> normalize(const Partial& p) {
    validate(p);
    std::vector<float> out(p.a.size(), 0);
    if (p.l != 0)
        for (std::size_t j = 0; j < out.size(); ++j) out[j] = checked(p.a[j] / p.l);
    return out; // both empty => explicit zero output, never NaN
}
inline std::vector<Row> visible_rows(std::vector<Row> rows, std::uint64_t query,
                                    const std::vector<std::uint64_t>& selected) {
    if (!std::is_sorted(selected.begin(), selected.end()) ||
        std::adjacent_find(selected.begin(), selected.end()) != selected.end())
        throw std::invalid_argument("selection must be unique ascending global IDs");
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
    for (std::size_t i = 1; i < rows.size(); ++i)
        if (rows[i - 1].id == rows[i].id) throw std::invalid_argument("duplicate row ID");
    rows.erase(std::remove_if(rows.begin(), rows.end(), [&](const auto& r) {
        return r.token > query || (r.candidate && !std::binary_search(selected.begin(), selected.end(), *r.candidate));
    }), rows.end());
    return rows;
}
struct Evaluation { std::array<Partial, 2> ranks; std::vector<float> output, reference; };
inline Evaluation evaluate(const std::vector<Row>& rows, std::uint64_t query,
                           const std::vector<std::uint64_t>& selected, OwnerMap map,
                           std::size_t width, std::optional<float> sink = std::nullopt) {
    validate(map);
    const auto visible = visible_rows(rows, query, selected);
    std::array<std::vector<Row>, 2> local;
    for (const auto& r : visible) local[owner(r.token, map)].push_back(r);
    Evaluation e{{accumulate(local[0], width), accumulate(local[1], width)}, {}, {}};
    auto merged = merge_rank0_rank1(e.ranks[0], e.ranks[1]);
    // A sink is a single zero-value entry, applied globally after the rank merge.
    if (sink) merged = merge_rank0_rank1(merged, accumulate({}, width, sink));
    e.output = normalize(merged);
    // Independent single-partition reduction, with sink in its original denominator.
    e.reference = normalize(accumulate(visible, width, sink));
    return e;
}
} // namespace strix::xport::kvshard
