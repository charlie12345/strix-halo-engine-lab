#include "xport/kvshard_oracle.hpp"
#include <cstdio>
#include <random>
#include <string>

namespace kv = strix::xport::kvshard;
namespace {
unsigned checks = 0, comparisons = 0;
float max_abs = 0, max_scaled = 0;
constexpr float tolerance = 2e-5F; // |sharded-reference| <= tol * (1+|reference|)
void require(bool ok, const char* why) {
    ++checks;
    if (!ok) throw std::runtime_error(why);
}
template<class Error, class F> void rejects(F f, const char* why) {
    bool caught = false;
    try { f(); } catch (const Error&) { caught = true; }
    require(caught, why);
}
void compare(const kv::Evaluation& e) {
    require(e.output.size() == e.reference.size(), "output dimensions");
    for (std::size_t j = 0; j < e.output.size(); ++j) {
        const float error = std::abs(e.output[j] - e.reference[j]);
        const float scaled = error / (1 + std::abs(e.reference[j]));
        max_abs = (std::max)(max_abs, error);
        max_scaled = (std::max)(max_scaled, scaled);
        ++comparisons;
        require(std::isfinite(e.output[j]) && scaled <= tolerance, "FP32 reference tolerance");
    }
}
void ownership() {
    constexpr auto glm = kv::GlmDsaTraits::owner_map;
    constexpr auto ds = kv::Ds41Traits<2>::owner_map;
    static_assert(kv::PartialDims::bytes == 65792);
    static_assert(kv::GlmDsaTraits::latent.bytes == 1024);
    static_assert(kv::Ds41Traits<1>::compressed_kv.bytes == 288);
    static_assert(kv::Ds41Traits<2>::window.bytes == 528);
    for (std::uint64_t token = 0; token < 35; ++token) {
        require(kv::owner(token, glm) == (token / 4) % 2, "GLM owner");
        require(kv::owner(token, ds) == (token / 2) % 2, "DS41 owner");
    }
    for (std::uint64_t group = 0; group < 12; ++group) {
        const auto start = kv::group_start(group, glm);
        for (std::uint64_t offset = 0; offset < 4; ++offset)
            require(kv::owner(start, glm) == kv::owner(start + offset, glm), "pool indivisible");
    }
    // Two consumers of the same KV/index source share the same global group IDs.
    constexpr kv::SourceMap consumers[]{{3, 1}, {3, 1}};
    require(consumers[0].kv_layer == consumers[1].kv_layer &&
            consumers[0].selection_layer == consumers[1].selection_layer, "source reuse schema");
    require(kv::block_count(0, glm) == 0 && kv::block_count(9, glm) == 3, "empty/tail blocks");
    const auto limit = (std::numeric_limits<std::uint64_t>::max)();
    require(kv::block_count(limit, {1, 1}) == limit, "ceil count avoids overflow");
    require(kv::owner(limit, glm) == (limit / 4) % 2, "maximum position");
    rejects<std::overflow_error>([&] { (void)kv::group_start(limit, glm); }, "group multiplication overflow");
    rejects<std::invalid_argument>([] { (void)kv::owner(0, {0, 4}); }, "zero block");
    rejects<std::invalid_argument>([] { (void)kv::owner(0, {4, 0}); }, "zero group");
    rejects<std::invalid_argument>([] { (void)kv::owner(0, {6, 4}); }, "split pool");
}
void edges() {
    const kv::OwnerMap map{4, 4};
    std::vector<kv::Row> rows;
    compare(kv::evaluate(rows, 0, {}, map, 2));
    const auto empty = kv::evaluate(rows, 0, {}, map, 2);
    require(std::isinf(empty.ranks[0].m) && empty.ranks[0].m < 0 &&
            empty.ranks[0].l == 0 && empty.ranks[0].a == std::vector<float>(2, 0), "canonical empty");
    require(empty.output == std::vector<float>(2, 0), "both empty output");
    compare(kv::evaluate(rows, 0, {}, map, 2, 1000.F)); // sink only
    rows = {{0, 0, {}, 1000.F, {2, -2}}, {4, 4, {}, 1000.F, {4, -4}},
            {8, 8, {}, 1000.F, {6, -6}}, {9, 9, {}, 1e30F, {99, 99}}};
    for (std::uint64_t query = 0; query <= 8; ++query) compare(kv::evaluate(rows, query, {}, map, 2));
    const auto one = kv::evaluate({rows[1]}, 4, {}, map, 2);
    require(one.ranks[0].l == 0 && one.output[0] == 4, "rank0 empty/rank1 nonempty");
    const auto sink = kv::evaluate({rows[0], rows[1]}, 4, {}, map, 2, 1000.F);
    compare(sink);
    require(std::abs(sink.output[0] - 2.F) < 1e-6F, "sink once globally: (2+4)/(1+1+1)");
    const auto limit = (std::numeric_limits<float>::max)();
    compare(kv::evaluate({{0, 0, {}, limit, {1}}, {4, 4, {}, -limit, {2}}}, 4, {}, map, 1));
    rejects<std::overflow_error>([&] {
        (void)kv::evaluate({{0, 0, {}, 0, {limit}}, {1, 1, {}, 0, {limit}}}, 1, {}, map, 1);
    }, "local accumulator overflow");
    rejects<std::overflow_error>([&] {
        (void)kv::evaluate({{0, 0, {}, 0, {limit}}, {4, 4, {}, 0, {limit}}}, 4, {}, map, 1);
    }, "merge accumulator overflow");
    rejects<std::invalid_argument>([&] { (void)kv::evaluate(rows, 8, {}, map, 3); }, "width check");
    rejects<std::invalid_argument>([] { (void)kv::accumulate({}, 0); }, "zero width");
    rejects<std::invalid_argument>([] { (void)kv::accumulate({}, 1, std::numeric_limits<float>::infinity()); }, "infinite sink");
    rejects<std::invalid_argument>([] {
        (void)kv::accumulate({{0, 0, {}, std::numeric_limits<float>::quiet_NaN(), {1}}}, 1);
    }, "NaN score");
    rejects<std::invalid_argument>([] { (void)kv::normalize({0, 0, {0}}); }, "malformed empty");
}
void selection() {
    const std::vector<kv::Candidate> candidates{{2, 11, 99}, {1, 7, 5}, {0, 3, 5}};
    const auto selected = kv::select_global(candidates, 8, 1);
    require(selected == std::vector<std::uint64_t>{0}, "global tie: smallest ID; future candidate masked");
    require(kv::select_global(candidates, 8, 0).empty(), "k zero");
    require(kv::select_global(candidates, 8, 99) == std::vector<std::uint64_t>({0, 1}), "k beyond visible count");
    require(kv::select_global({}, 0, 1).empty(), "empty candidate set");
    std::vector<kv::Row> rows;
    for (std::uint64_t token = 0; token < 10; ++token)
        rows.push_back({token, token, token < 8 ? std::optional<std::uint64_t>(token / 4) : std::nullopt,
                        0, {static_cast<float>(token)}});
    const auto e = kv::evaluate(rows, 8, selected, {4, 4}, 1);
    compare(e);
    require(std::abs(e.output[0] - 2.8F) < 1e-6F, "selected pool expanded, mandatory tail, causal mask");
    const auto wrong = kv::evaluate(rows, 8, {0, 1}, {4, 4}, 1);
    require(std::abs(wrong.output[0] - e.output[0]) > 1, "local top-k union changes global result");
    rejects<std::invalid_argument>([] { (void)kv::select_global({{0, 0, 1}, {0, 1, 2}}, 1, 1); }, "duplicate candidate");
    rejects<std::invalid_argument>([&] { (void)kv::evaluate(rows, 8, {1, 0}, {4, 4}, 1); }, "selection order");
    rejects<std::invalid_argument>([&] { (void)kv::evaluate(rows, 8, {0, 0}, {4, 4}, 1); }, "duplicate selection");
    rejects<std::invalid_argument>([&] { (void)kv::evaluate({rows[0], rows[0]}, 8, {}, {4, 4}, 1); }, "duplicate row");
}
void compressed_source() {
    const auto map = kv::Ds41Traits<2>::owner_map;
    const auto selected = kv::select_global({{0, 1, 4}, {1, 3, 3}, {2, 5, 9}}, 4, 2);
    require(selected == std::vector<std::uint64_t>({0, 1}), "incomplete compressed group masked");
    const std::vector<kv::Row> shared_cache{
        {0, kv::group_start(0, map), 0, 0, {2, 4}},
        {1, kv::group_start(1, map), 1, 0, {4, 2}},
        {2, kv::group_start(2, map), 2, 1000, {99, 99}}};
    const auto first = kv::evaluate(shared_cache, 4, selected, map, 2, 0.F);
    const auto second = kv::evaluate(shared_cache, 4, selected, map, 2, 0.F);
    compare(first); compare(second);
    require(first.output == second.output && first.output[0] == 2 && first.output[1] == 2,
            "source reuse: same owned cache/selection, one sink per consumer head");
    require(first.ranks[0].l == 1 && first.ranks[1].l == 1, "compressed groups alternate owners");
}
void sweep() {
    std::mt19937 rng(711206U);
    auto sample = [&]() { return static_cast<float>(rng() % 20001U) / 10000.F - 1.F; };
    // 32 independent heads, both cache organizations, dense and sparse contexts.
    for (unsigned head = 0; head < 32; ++head) {
        for (unsigned mode = 0; mode < 6; ++mode) {
            const std::uint64_t group = mode % 2 == 0 ? 4 : 2;
            const kv::OwnerMap map{group * (1 + mode / 2), group};
            const std::uint64_t count = 17 + rng() % 241U;
            const std::uint64_t query = rng() % count;
            std::vector<kv::Candidate> candidates;
            for (std::uint64_t g = 0; g < count / group; ++g)
                candidates.push_back({g, (g + 1) * group - 1, sample()});
            const auto selected = kv::select_global(candidates, query, 13);
            std::vector<kv::Row> rows;
            for (std::uint64_t t = 0; t < count; ++t) {
                std::vector<float> value(512);
                for (float& x : value) x = sample() * 2;
                const auto candidate = mode < 2 || t >= count / group * group
                    ? std::nullopt : std::optional<std::uint64_t>(t / group);
                rows.push_back({t, t, candidate, sample() * (mode < 3 ? 4.F : 100.F), std::move(value)});
            }
            const auto sink = head % 2 ? std::optional<float>(sample() * 10) : std::nullopt;
            const auto e = kv::evaluate(rows, query, selected, map, 512, sink);
            compare(e);
            // Input order must not affect the declared ascending-ID reduction.
            std::reverse(rows.begin(), rows.end());
            require(kv::evaluate(rows, query, selected, map, 512, sink).output == e.output, "deterministic row order");
        }
    }
}
} // namespace
int main() {
    try {
        ownership(); edges(); selection(); compressed_source(); sweep();
        std::printf("PASS checks=%u compared_components=%u max_abs=%.9g max_scaled=%.9g tolerance=%.9g*(1+abs(ref))\n",
                    checks, comparisons, static_cast<double>(max_abs), static_cast<double>(max_scaled),
                    static_cast<double>(tolerance));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
