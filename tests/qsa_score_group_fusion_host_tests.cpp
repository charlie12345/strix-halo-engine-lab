#include "qsa_score_group_fusion_oracle.hpp"
#include "qsa_index_score_fixture.hpp"
#include <iostream>
#include <limits>
#include <random>
#include <string>

namespace {
using namespace qsa_group_oracle;
unsigned cases = 0;
void require(bool yes, const char* message) { if (!yes) throw std::runtime_error(message); }
// Independent full-image reference: numeric float order, then ascending block.
// Admitted scores are finite and >= 0, for which numeric order equals select_key.
std::vector<unsigned> reference(const std::vector<float>& scores, unsigned visible) {
    const auto complete = visible / ratio;
    if (complete <= topk) {
        std::vector<unsigned> out(visible);
        std::iota(out.begin(), out.end(), 0U);
        return out;
    }
    std::vector<unsigned> blocks(complete);
    std::iota(blocks.begin(), blocks.end(), 0U);
    std::sort(blocks.begin(), blocks.end(), [&](unsigned a, unsigned b) {
        return scores[a] != scores[b] ? scores[a] > scores[b] : a < b;
    });
    blocks.resize(topk);
    std::sort(blocks.begin(), blocks.end());
    std::vector<unsigned> out;
    for (auto b : blocks) for (unsigned j = 0; j < ratio; ++j) out.push_back(b * ratio + j);
    for (unsigned t = complete * ratio; t < visible; ++t) out.push_back(t);
    return out;
}
void check(const std::vector<float>& scores, unsigned tail) {
    const auto visible = static_cast<unsigned>(scores.size()) * ratio + tail;
    unsigned first_reads = 0, second_reads = 0;
    auto p = plan(visible, [&](unsigned b) { ++first_reads; return scores.at(b); });
    require(!p.fallback, "unexpected fallback");
    require(p.groups.size() <= topk, "group bound");
    auto compact = rescore(p, visible, [&](unsigned b) { ++second_reads; return scores.at(b); });
    require(compact.size() <= group * topk, "compact score bound");
    if (scores.size() <= topk) {
        require(p.dense && first_reads == 0 && second_reads == 0, "dense row consumed skipped query");
    } else {
        require(first_reads == scores.size() && second_reads == compact.size(), "score visits");
        require(p.maxima.size() == (scores.size() + group - 1) / group, "maxima extent");
        for (unsigned g = 0; g < p.maxima.size(); ++g) {
            float maximum = -INFINITY;
            for (unsigned j = 0; j < group && g * group + j < scores.size(); ++j)
                maximum = std::max(maximum, scores[g * group + j]);
            require(key(maximum) == key(p.maxima[g]), "group maximum key changed");
        }
        for (auto e : compact)
            require(std::bit_cast<std::uint32_t>(e.score) == std::bit_cast<std::uint32_t>(scores[e.block]), "score bits changed");
    }
    const auto got = tokens(visible, compact);
    require(got == reference(scores, visible), "complete token/tie/tail mismatch");
    require(got.size() == std::min<unsigned>(static_cast<unsigned>(scores.size()), topk) * ratio + tail, "1536 budget changed");
    if (!p.dense) {
        auto post_a = scores, post_b = scores;
        for (unsigned j = 0; j < topk * ratio; j += ratio) post_a[got[j] / ratio] = -INFINITY;
        const auto ref = reference(scores, visible);
        for (unsigned j = 0; j < topk * ratio; j += ratio) post_b[ref[j] / ratio] = -INFINITY;
        for (unsigned b = 0; b < scores.size(); ++b)
            require(std::bit_cast<std::uint32_t>(post_a[b]) == std::bit_cast<std::uint32_t>(post_b[b]), "post-selection image changed");
    }
    ++cases;
}
void query_contract() {
    // The exact range prologue reads every row of any active 64-row group.
    // Candidate dense rows can independently retain nullptr without affecting OFF.
    unsigned mixed = 0;
    std::array<float, 64U * index_fixture::heads * index_fixture::dim> queries{};
    for (unsigned position : {0U, 16384U}) for (unsigned first = 0; first < 16384; first += 64) {
        std::array<unsigned, 64> blocks{};
        bool active = false;
        for (unsigned r = 0; r < 64; ++r) {
            const auto complete = (position + first + r + 1) / ratio;
            blocks[r] = complete > topk ? complete : 0;
            active = active || blocks[r] != 0;
        }
        if (!active) continue;
        for (unsigned arm = 0; arm < 2; ++arm) for (unsigned r = 0; r < 64; ++r) {
            const auto complete = (position + first + r + 1U) / ratio;
            const float* row = queries.data() + r * index_fixture::heads * index_fixture::dim;
            const float* query = index_fixture::query_for_arm(arm, complete, row);
            require(arm != 0 || query != nullptr, "exact prologue null query");
            if (!arm) require(query[index_fixture::heads * index_fixture::dim - 1U] == 0.0F,
                "exact range prologue cannot read complete query");
            require(arm != 1 || (query != nullptr) == (blocks[r] != 0), "candidate dense sentinel lost");
            if (query) require(query == row, "query row address changed");
            if (!arm && !blocks[r]) ++mixed;
        }
    }
    require(mixed == 3, "cutoff seam coverage lost");
}
} // namespace
int main() try {
    require(budget == 1536 && topk == 384 && group == 4, "fixed geometry");
    require(key(0.0F) == key(-0.0F), "zero key normalization");
    query_contract();
    std::mt19937 rng(0x515341U);
    for (unsigned n : {0U, 1U, 63U, 64U, 65U, 383U, 384U, 385U, 511U, 512U, 513U,
                       1535U, 1536U, 1537U, 4095U, 4096U, 4097U, 8191U, 8192U, 8193U, 65536U}) {
        std::vector<float> scores(n);
        for (unsigned mode = 0; mode < 6; ++mode) {
            for (unsigned b = 0; b < n; ++b) {
                if (mode == 0) scores[b] = 0.0F; // all groups tied; must remain bounded
                if (mode == 1) scores[b] = static_cast<float>(rng() % 17); // cutoff ties
                if (mode == 2) scores[b] = static_cast<float>(b); // later groups dominate
                if (mode == 3) scores[b] = static_cast<float>(n - b);
                if (mode == 4) scores[b] = std::bit_cast<float>(static_cast<std::uint32_t>(0x3f800000U + (rng() % 0x10000U))); // ULPs
                if (mode == 5) scores[b] = b % 2 ? -0.0F : 0.0F;
            }
            for (unsigned tail = 0; tail < ratio; ++tail) check(scores, tail);
        }
    }
    // Concentrate many winners in one group; never retain just one block/group.
    std::vector<float> clustered(8193, 0.0F);
    for (unsigned b = 2000; b < 2384; ++b) clustered[b] = 4.0F;
    for (unsigned tail = 0; tail < ratio; ++tail) check(clustered, tail);
    // Exact group cutoff ordering includes an earlier group's late-column winner.
    std::vector<float> adversary(8193, 0.0F);
    for (unsigned g = 0; g < 2048; ++g) adversary[g * group + 3] = 1.0F;
    check(adversary, 3);
    for (float bad : {INFINITY, -INFINITY, std::numeric_limits<float>::quiet_NaN(), -1.0F}) {
        std::vector<float> scores(8192, 0.0F); scores[4097] = bad;
        const auto p = plan(8192 * ratio, [&](unsigned b) { return scores[b]; });
        require(p.fallback && p.groups.empty(), "invalid score must fall back before rescore");
        bool declined = false;
        try { (void)rescore(p, 8192 * ratio, [&](unsigned b) { return scores[b]; }); }
        catch (const std::invalid_argument&) { declined = true; }
        require(declined, "fallback must not publish compact scores");
    }
    std::cout << "PASS " << cases << " full-image/group-plan cases; 4 fallback cases; exact-query cutoff seam; budget 1536\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
