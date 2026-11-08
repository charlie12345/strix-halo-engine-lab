// Coupled acceptance (STRIX_MTP_SAMPLED_ACCEPT=coupled): a drafted run must emit exactly the tokens of the plain
// sampled run with the same seed, for any draft distribution, draft length, window size and draft schedule. The
// generation loop has the engine's shape (pending token, draft chain proposing positions base + k + 1 from the
// pending row's position base, ordered row decisions until the first row without a draft, a rejection, EOS or the
// budget). Also checks that the default options keep the previous identity and uniforms, forced rejection at every
// depth, an adaptive schedule, cohort permutations against C1 references keyed by seed and request identity, budget
// tails, dense EOS, the argmax-draft gate and more filter combinations.
#include "strix/speculation_rng.hpp"
#include "strix/token_sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

constexpr std::uint32_t vocabulary = 64;

std::uint64_t mix(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}
double hashed_unit(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
    return static_cast<double>(mix(a * 0x100000001b3ULL ^ mix(b * 31U + c)) >> 11U) / 9007199254740992.0;
}

// Synthetic target: the row deciding position `position` after input token `input`.
std::vector<float> target_logits(std::uint32_t input, std::uint64_t position) {
    std::vector<float> out(vocabulary);
    for (std::uint32_t j = 0; j < vocabulary; ++j)
        out[j] = static_cast<float>(2.5 * std::sin(1.3 * input + 0.7 * j + 0.013 * static_cast<double>(position)) +
                                    3.0 * hashed_unit(input, j, position % 7U));
    out[(input * 7U + 3U) % vocabulary] += 2.0F; // a likely continuation, like a language model row
    return out;
}

enum class DraftKind { identical, close, close_subset, unrelated };
const char* name(DraftKind k) {
    switch (k) {
    case DraftKind::identical: return "identical";
    case DraftKind::close: return "close";
    case DraftKind::close_subset: return "close-subset";
    default: return "unrelated";
    }
}

// Draft entry ids: the full vocabulary in id order, or a subset (no id % 5 == 3) in a shuffled order like a
// reduced draft head.
std::vector<std::uint32_t> draft_ids(DraftKind kind) {
    std::vector<std::uint32_t> ids;
    if (kind != DraftKind::close_subset) return ids;
    for (std::uint32_t j = 0; j < vocabulary; ++j) {
        const auto id = (j * 37U) % vocabulary; // a permutation (37 is odd)
        if (id % 5U != 3U) ids.push_back(id);
    }
    return ids;
}

std::vector<float> draft_logits(DraftKind kind, std::span<const std::uint32_t> ids, std::uint32_t input, std::uint64_t position) {
    const auto target = target_logits(input, position);
    std::vector<float> out;
    const auto value = [&](std::uint32_t id) -> float {
        switch (kind) {
        case DraftKind::identical: return target[id];
        case DraftKind::unrelated: return static_cast<float>(4.0 * hashed_unit(input + 1000U, id, position));
        default: return static_cast<float>(target[id] + 0.9 * (hashed_unit(input, id + 77U, position) - 0.5));
        }
    };
    if (ids.empty()) for (std::uint32_t j = 0; j < vocabulary; ++j) out.push_back(value(j));
    else for (const auto id : ids) out.push_back(value(id));
    return out;
}

struct Run {
    std::vector<std::uint32_t> tokens;
    std::uint64_t accepted = 0, rejected = 0, drafts = 0, forced = 0, windows = 0;
    bool eos_stop = false;          // the run ended on an EOS token
    bool eos_mid_window = false;    // an EOS was decided by a row after row 0 of a window
};

// Coverage of the conditions the equality runs are meant to reach (checked once at the end).
struct Coverage {
    std::uint64_t forced = 0, forced_rows[8] = {}, eos_stops = 0, eos_mid_window = 0, tails = 0, adaptive_cycles = 0;
} coverage;

// Plain sampling: one decision per position, no drafts.
Run plain(const strix::TokenSamplingPolicy& policy, strix::TokenSamplerOptions options, std::uint32_t prompt,
    std::uint32_t length, std::span<const std::uint32_t> eos) {
    strix::TokenSampler sampler(policy, options);
    Run run;
    std::uint32_t input = prompt;
    for (std::uint64_t position = 1; run.tokens.size() < length; ++position) {
        const auto d = sampler.decide(target_logits(input, position), position, nullptr);
        run.tokens.push_back(d.token);
        input = d.token;
        if (std::find(eos.begin(), eos.end(), d.token) != eos.end()) break;
    }
    return run;
}

// Speculative run. depth_max: the draft chain length R. schedule 0: R drafts and R + 1 rows every cycle;
// 1: a hashed depth in [0, R] per cycle; 2: R drafts but a window of a hashed size in [1, R + 1] rows (unverified
// drafts are dropped, as when an owner smaller than the chain is chosen); 3: adaptive, the depth (one more than
// the previous cycle's accepted drafts, at most R) and the window (the smallest owner of 2, 4 or 8 rows covering
// two more rows than the cycle before last accepted, at most depth + 1; drafts beyond it are dropped) follow
// earlier decisions, as the timed controller's choices do.
Run speculative(const strix::TokenSamplingPolicy& policy, strix::TokenSamplerOptions options, DraftKind kind,
    std::uint32_t depth_max, unsigned schedule, std::uint32_t prompt, std::uint32_t length, std::span<const std::uint32_t> eos) {
    strix::TokenSampler sampler(policy, options);
    const auto ids = draft_ids(kind);
    Run run;
    // The prompt row: a plain decision at position 1 (the prompt token sits at position 0).
    run.tokens.push_back(sampler.decide(target_logits(prompt, 1), 1, nullptr).token);
    std::uint64_t base = 1; // position of the pending token
    std::uint32_t adaptive_depth = depth_max, previous_accepted = depth_max, last_accepted = depth_max;
    if (std::find(eos.begin(), eos.end(), run.tokens.back()) != eos.end()) run.eos_stop = true;
    for (std::uint64_t cycle = 0; run.tokens.size() < length; ++cycle) {
        if (std::find(eos.begin(), eos.end(), run.tokens.back()) != eos.end()) break;
        const auto pending = run.tokens.back();
        std::uint32_t depth = depth_max;
        if (schedule == 1U) depth = static_cast<std::uint32_t>(mix(policy.seed * 1315423911ULL + cycle) % (depth_max + 1U));
        if (schedule == 3U) { depth = adaptive_depth; ++coverage.adaptive_cycles; }
        std::vector<strix::TokenDraft> drafts;
        std::uint32_t input = pending;
        for (std::uint32_t k = 0; k < depth; ++k) {
            // Draft k proposes the token at base + k + 1: the position target row k decides.
            drafts.push_back(sampler.draft(draft_logits(kind, ids, input, base + k + 1U), ids, base + k + 1U));
            input = drafts.back().token;
        }
        run.drafts += drafts.size();
        std::uint32_t rows = depth + 1U;
        if (schedule == 2U) rows = 1U + static_cast<std::uint32_t>(mix(policy.seed ^ (cycle * 2654435761ULL)) % (depth + 1U));
        if (schedule == 3U) {
            const auto want = previous_accepted + 2U;
            rows = std::min(want <= 2U ? 2U : want <= 4U ? 4U : 8U, depth + 1U);
        }
        rows = std::min<std::uint32_t>(rows, static_cast<std::uint32_t>(length - run.tokens.size()));
        if (rows < depth + 1U && rows == length - run.tokens.size()) ++coverage.tails;
        std::vector<std::uint32_t> inputs{pending};
        for (std::uint32_t r = 1; r < rows; ++r) inputs.push_back(drafts[r - 1U].token);
        std::vector<float> row;
        const auto decisions = sampler.decide_rows(rows, base + 1U, drafts, eos, [&](std::uint32_t r) {
            row = target_logits(inputs[r], base + 1U + r);
            return std::span<const float>(row);
        });
        ++run.windows;
        std::uint32_t accepted_now = 0;
        for (std::size_t r = 0; r < decisions.size(); ++r) {
            const auto& d = decisions[r];
            run.tokens.push_back(d.token);
            if (d.drafted) (d.accepted ? run.accepted : run.rejected) += 1U;
            accepted_now += d.accepted ? 1U : 0U;
            if (d.forced) {
                // A forced rejection keeps the plain token (equal to the draft) and ends the window's decisions.
                expect(r + 1U == decisions.size() && d.token == drafts[r].token && !d.accepted,
                    "a forced rejection is the window's last decision and keeps the draft's token");
                ++run.forced;
                ++coverage.forced;
                ++coverage.forced_rows[r % 8U];
            }
            if (std::find(eos.begin(), eos.end(), d.token) != eos.end()) {
                expect(r + 1U == decisions.size(), "an EOS decision ends the window");
                if (r > 0U) run.eos_mid_window = true;
            }
        }
        if (std::find(eos.begin(), eos.end(), run.tokens.back()) != eos.end()) run.eos_stop = true;
        adaptive_depth = std::min(depth_max, accepted_now + 1U);
        previous_accepted = last_accepted;
        last_accepted = accepted_now;
        // Rollback by committed rows only: the next cycle's pending row is the last decided one.
        base += decisions.size();
    }
    if (run.eos_stop) ++coverage.eos_stops;
    if (run.eos_mid_window) ++coverage.eos_mid_window;
    return run;
}

std::size_t first_difference(const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b) {
    std::size_t at = 0;
    while (at < std::min(a.size(), b.size()) && a[at] == b[at]) ++at;
    return at;
}

strix::TokenSamplerOptions coupled_options(std::uint32_t forced = strix::TokenSamplerOptions::forced_rejection_off) {
    strix::TokenSamplerOptions out;
    out.acceptance = strix::SampledAcceptance::coupled;
    out.force_reject_row = forced;
    return out;
}

std::string forced_name(std::uint32_t forced) {
    if (forced == strix::TokenSamplerOptions::forced_rejection_off) return "off";
    if (forced == strix::TokenSamplerOptions::forced_rejection_rotate) return "rotate";
    return std::to_string(forced);
}

void test_defaults() {
    const strix::TokenSamplingPolicy policy{0.8F, 6, 0.95F, 0.02F, 42};
    const strix::TokenSamplerOptions defaults;
    expect(defaults.acceptance == strix::SampledAcceptance::rejection && defaults.request_identity == 1U,
        "default options: rejection acceptance, request identity 1");
    const auto env = strix::token_sampler_options_from_environment();
    expect(env.acceptance == strix::SampledAcceptance::rejection && env.request_identity == 1U,
        "environment without STRIX_MTP_SAMPLED_ACCEPT: rejection");
    expect(strix::next_token_sampling_request_identity() == 1U && strix::next_token_sampling_request_identity() == 1U,
        "without STRIX_SAMPLING_REQUEST_IDENTITY every request has identity 1");
    // The uniforms are the ones of the previous fixed single-request identity.
    const strix::SpeculationRngIdentity legacy{policy.seed, 1U, strix::token_sampling_policy_identity(policy)};
    strix::TokenSampler sampler(policy);
    strix::TokenSamplerOptions coupled_options;
    coupled_options.acceptance = strix::SampledAcceptance::coupled;
    strix::TokenSampler coupled(policy, coupled_options);
    bool same = true;
    for (std::uint64_t n = 0; n < 64; ++n) {
        const auto u = sampler.uniforms(n, true);
        same = same && u.plain == strix::speculation_uniform01(legacy, n, strix::SpeculationRandomDrawClass::serial_target) &&
               u.accept == strix::speculation_uniform01(legacy, n, strix::SpeculationRandomDrawClass::draft_acceptance) &&
               u.residual == strix::speculation_uniform01(legacy, n, strix::SpeculationRandomDrawClass::residual_fallback) &&
               sampler.draft_uniform(n) == strix::speculation_uniform01(legacy, n, strix::SpeculationRandomDrawClass::draft_proposal);
        const auto c = coupled.uniforms(n, true);
        same = same && c.plain == u.plain && c.accept == 0.0 && c.residual == 0.0 && coupled.draft_uniform(n) == u.plain;
    }
    expect(same, "default uniforms equal the identity-1 draws; coupled draws U(n, plain) for drafts and decisions");
    auto other = strix::TokenSamplerOptions{};
    other.request_identity = 2U;
    expect(strix::TokenSampler(policy, other).uniforms(5, false).plain != sampler.uniforms(5, false).plain,
        "a different request identity draws different uniforms");
    // The argmax-draft gate changes only the draft transform.
    auto low = policy;
    low.temperature = 0.5F;
    strix::TokenSampler low_coupled(low, coupled_options), low_rejection(low, defaults);
    expect(low_coupled.draft_policy().top_k == 1U && low_coupled.policy().top_k == low.top_k &&
               low_rejection.draft_policy().top_k == low.top_k && coupled.draft_policy().top_k == policy.top_k,
        "coupled drafts are first-max below the gate only; targets keep the policy");
    // decide_token_coupled: the plain sample; acceptance is equality.
    const auto p = strix::make_token_distribution(target_logits(3, 9), {}, policy);
    strix::TokenDraft d;
    d.position = 9;
    for (const double u : {0.01, 0.3, 0.77, 0.999}) {
        d.token = p.sample(u);
        const auto hit = strix::decide_token_coupled(p, &d, u);
        d.token = (p.sample(u) + 1U) % vocabulary;
        const auto miss = strix::decide_token_coupled(p, &d, u);
        expect(hit.token == p.sample(u) && hit.drafted && hit.accepted && !hit.residual && miss.token == p.sample(u) &&
                   miss.drafted && !miss.accepted && !miss.residual && strix::decide_token_coupled(p, nullptr, u).token == p.sample(u),
            "decide_token_coupled decides the plain sample and accepts by equality");
    }
}

void test_equality() {
    constexpr std::uint32_t length = 512, seeds = 16;
    const strix::TokenSamplingPolicy policies[] = {
        {1.0F, 0, 1.0F, 0.0F, 0}, {0.7F, 0, 1.0F, 0.0F, 0}, {0.7F, 40, 0.95F, 0.0F, 0}, {1.0F, 0, 0.9F, 0.0F, 0},
        {0.5F, 0, 1.0F, 0.05F, 0}, {1.3F, 20, 0.95F, 0.02F, 0},
    };
    const std::vector<std::uint32_t> no_eos, eos{17};
    strix::TokenSamplerOptions coupled;
    coupled.acceptance = strix::SampledAcceptance::coupled;
    std::uint64_t runs = 0, tokens = 0, rejection_differs = 0;
    for (const auto& base : policies) {
        for (const auto kind : {DraftKind::identical, DraftKind::close, DraftKind::close_subset, DraftKind::unrelated}) {
            std::uint64_t accepted = 0, rejected = 0, drafts = 0, r_accepted = 0, r_rejected = 0;
            for (const std::uint32_t depth : {2U, 4U, 8U}) {
                for (std::uint32_t s = 0; s < seeds; ++s) {
                    auto policy = base;
                    policy.seed = 1000U + s;
                    const auto& stop = s % 6U == 5U ? eos : no_eos;
                    const auto reference = plain(policy, coupled, 3U + s % 5U, length, stop);
                    // The plain run is the same with either option set (the no-draft decision does not change).
                    if (s == 0U) expect(plain(policy, {}, 3U + s % 5U, length, stop).tokens == reference.tokens,
                        "plain sampling is independent of the acceptance option");
                    for (unsigned schedule = 0; schedule < 4U; ++schedule) {
                        const auto run = speculative(policy, coupled, kind, depth, schedule, 3U + s % 5U, length, stop);
                        ++runs;
                        tokens += run.tokens.size();
                        accepted += run.accepted;
                        rejected += run.rejected;
                        drafts += run.drafts;
                        if (run.tokens != reference.tokens) {
                            std::size_t at = 0;
                            while (at < std::min(run.tokens.size(), reference.tokens.size()) && run.tokens[at] == reference.tokens[at]) ++at;
                            expect(false, std::string("coupled tokens differ from plain sampling: T ") + std::to_string(policy.temperature) +
                                " top-k " + std::to_string(policy.top_k) + " top-p " + std::to_string(policy.top_p) + " draft " +
                                name(kind) + " R" + std::to_string(depth) + " schedule " + std::to_string(schedule) + " seed " +
                                std::to_string(policy.seed) + " first difference at " + std::to_string(at));
                        }
                    }
                    // Rejection sampling (the default) is exact only in distribution: its tokens may differ.
                    if (depth == 4U) {
                        const auto r = speculative(policy, {}, kind, depth, 0U, 3U + s % 5U, length, stop);
                        rejection_differs += r.tokens != reference.tokens ? 1U : 0U;
                        r_accepted += r.accepted;
                        r_rejected += r.rejected;
                    }
                }
            }
            const double rate = drafts ? static_cast<double>(accepted) / static_cast<double>(accepted + rejected) : 0.0;
            std::cout << "T " << base.temperature << " top-k " << base.top_k << " top-p " << base.top_p << " min-p " << base.min_p
                      << " draft " << name(kind) << ": coupled acceptance " << rate << " (" << accepted << " / " << accepted + rejected
                      << " decided drafts); rejection R4 acceptance "
                      << static_cast<double>(r_accepted) / static_cast<double>(std::max<std::uint64_t>(1U, r_accepted + r_rejected)) << '\n';
            if (kind == DraftKind::identical && base.temperature >= 0.7F)
                expect(rejected == 0U && accepted > 0U, "a draft identical to the target is always accepted (same uniform, same order)");
            if (kind == DraftKind::close || kind == DraftKind::close_subset) expect(accepted > 0U && rejected > 0U,
                "close drafts are accepted and rejected");
            if (kind == DraftKind::unrelated) expect(rejected > 0U, "unrelated drafts are rejected");
        }
    }
    std::cout << runs << " coupled runs (" << tokens << " tokens) equal plain sampling; rejection sampling differed in "
              << rejection_differs << " sampled runs\n";
    expect(rejection_differs > 0U, "the equality check is discriminative (rejection sampling differs per seed)");
}

// Forced rejection (the test-only STRIX_MTP_SAMPLED_FORCE_REJECT switch): a drafted row of a chosen index whose
// draft was accepted is decided as rejected, so the window commits only the rows up to it and the next cycle
// continues from there. The tokens must stay those of plain sampling at every forced depth, and the switch must
// be refused without coupled acceptance.
void test_forced_rejection() {
    using Options = strix::TokenSamplerOptions;
    const strix::TokenSamplingPolicy base{1.0F, 0, 1.0F, 0.0F, 0};
    bool refused = false;
    try {
        Options o;
        o.force_reject_row = 1U;
        const strix::TokenSampler bad(base, o);
    } catch (const std::invalid_argument&) { refused = true; }
    expect(refused, "forced rejection is refused with rejection sampling");
    refused = false;
    try {
        const strix::TokenSampler bad(base, coupled_options(8U));
    } catch (const std::invalid_argument&) { refused = true; }
    expect(refused, "forced rejection rows are 0..7 or rotate");
    const strix::TokenSamplingPolicy policies[] = {{1.0F, 0, 1.0F, 0.0F, 0}, {0.7F, 20, 0.8F, 0.0F, 0}, {0.5F, 0, 1.0F, 0.05F, 0}};
    const std::vector<std::uint32_t> no_eos, eos{17};
    std::uint64_t runs = 0, forced = 0;
    for (const auto& p : policies) {
        for (const auto kind : {DraftKind::identical, DraftKind::close}) {
            for (const std::uint32_t depth : {1U, 3U, 7U}) {
                for (const std::uint32_t row : {0U, 1U, 3U, 7U, Options::forced_rejection_rotate}) {
                    for (std::uint32_t s = 0; s < 4U; ++s) {
                        auto policy = p;
                        policy.seed = 3000U + s;
                        const auto& stop = s == 3U ? eos : no_eos;
                        const auto reference = plain(policy, coupled_options(), 5U + s, 256U, stop);
                        for (const unsigned schedule : {0U, 3U}) {
                            const auto run = speculative(policy, coupled_options(row), kind, depth, schedule, 5U + s, 256U, stop);
                            ++runs;
                            forced += run.forced;
                            if (run.tokens != reference.tokens)
                                expect(false, "forced rejection at row " + forced_name(row) + " R" + std::to_string(depth + 1U) +
                                    " schedule " + std::to_string(schedule) + " draft " + name(kind) + " seed " +
                                    std::to_string(policy.seed) + ": first difference at " +
                                    std::to_string(first_difference(run.tokens, reference.tokens)));
                            // Identical drafts are always accepted at T >= 0.7, so a reachable forced row fires.
                            if (kind == DraftKind::identical && p.temperature >= 0.7F && row < depth && schedule == 0U)
                                expect(run.forced > 0U && run.forced <= run.windows, "a reachable forced row fires");
                            if (row != Options::forced_rejection_rotate && row >= depth)
                                expect(run.forced == 0U, "a forced row beyond the drafted rows forces nothing");
                        }
                    }
                }
            }
        }
    }
    std::cout << runs << " forced-rejection runs (rows 0, 1, 3, 7 and rotate; R2, R4, R8; fixed and adaptive) equal plain "
                 "sampling; " << forced << " rejections forced\n";
    for (const std::uint32_t row : {0U, 1U, 3U, 6U})
        expect(coverage.forced_rows[row] > 0U, "forced rejections reached row " + std::to_string(row));
}

// Cohort permutations: four members (seed + member index, as the tiled cohort route assigns them, and a request
// identity), each with its own sampler, schedule and forced row. Every member must equal its C1 reference: plain
// sampling of the same prompt with the same seed and request identity, whatever the member order. A different
// request identity draws a different stream, so a reference must match both keys.
void test_cohort_permutations() {
    const strix::TokenSamplingPolicy base{0.9F, 40, 0.95F, 0.0F, 0};
    const std::uint32_t prompts[] = {3U, 11U, 29U, 3U}; // the same prompt twice, with different seeds
    const std::uint32_t orders[][4] = {{0, 1, 2, 3}, {3, 2, 1, 0}, {2, 0, 3, 1}, {1, 3, 0, 2}};
    constexpr std::uint32_t length = 192, seed = 5000, depth = 7;
    std::uint64_t members = 0;
    struct Member {
        std::uint32_t prompt = 0;
        strix::TokenSamplingPolicy policy;
        strix::TokenSamplerOptions options;
        Run run;
    };
    for (const bool counter : {false, true}) {
        for (const auto& order : orders) {
            std::vector<Member> cohort;
            for (std::uint32_t i = 0; i < 4U; ++i) {
                Member m;
                m.prompt = prompts[order[i]];
                m.policy = base;
                m.policy.seed = seed + i;
                m.options = coupled_options();
                m.options.request_identity = counter ? 1U + i : 1U; // admission counter, or the default identity
                cohort.push_back(m);
            }
            // A keyed stream does not depend on interleaving; each member gets its own schedule and forced row,
            // so its windows differ from the other members'.
            for (std::uint32_t i = 0; i < 4U; ++i) {
                auto& m = cohort[i];
                auto o = m.options;
                o.force_reject_row = i == 3U ? strix::TokenSamplerOptions::forced_rejection_off : i * 2U;
                m.run = speculative(m.policy, o, DraftKind::close, depth, 1U + i % 3U, m.prompt, length, {});
            }
            for (const auto& m : cohort) {
                const auto reference = plain(m.policy, m.options, m.prompt, length, {});
                ++members;
                expect(m.run.tokens == reference.tokens, "cohort member (prompt " + std::to_string(m.prompt) + ", seed " +
                    std::to_string(m.policy.seed) + ", request identity " + std::to_string(m.options.request_identity) +
                    ") differs from its C1 reference at " + std::to_string(first_difference(m.run.tokens, reference.tokens)));
                if (m.options.request_identity != 1U) {
                    auto legacy = m.options;
                    legacy.request_identity = 1U;
                    expect(plain(m.policy, legacy, m.prompt, length, {}).tokens != reference.tokens,
                        "a reference with another request identity is a different stream");
                }
            }
        }
    }
    std::cout << members << " cohort members (4 permutations, default and counter request identities) equal their C1 "
                 "references\n";
}

// Budget tails, dense EOS, the argmax-draft gate and further filters: equality with plain sampling.
void test_tails_eos_filters() {
    std::uint64_t runs = 0;
    const strix::TokenSamplingPolicy policies[] = {
        {1.0F, 1, 1.0F, 0.0F, 0}, {1.0F, 0, 0.5F, 0.0F, 0}, {0.8F, 0, 1.0F, 0.2F, 0}, {0.6F, 8, 0.9F, 0.05F, 0},
    };
    // Tails: budgets that end inside a window, from one token up.
    for (const auto& p : policies) {
        for (const std::uint32_t length : {1U, 2U, 3U, 5U, 9U, 61U}) {
            for (const std::uint32_t depth : {1U, 3U, 7U}) {
                for (std::uint32_t s = 0; s < 3U; ++s) {
                    auto policy = p;
                    policy.seed = 7000U + s;
                    const auto reference = plain(policy, coupled_options(), 7U + s, length, {});
                    for (unsigned schedule = 0; schedule < 4U; ++schedule) {
                        const auto run = speculative(policy, coupled_options(), DraftKind::close, depth, schedule, 7U + s, length, {});
                        ++runs;
                        expect(run.tokens == reference.tokens && run.tokens.size() == length,
                            "budget tail " + std::to_string(length) + " R" + std::to_string(depth + 1U) + " schedule " +
                            std::to_string(schedule) + " differs from plain sampling");
                    }
                }
            }
        }
    }
    // Dense EOS: several stop tokens, so runs stop at row 0 and inside windows.
    const std::vector<std::uint32_t> eos{17U, 5U, 40U, 24U};
    for (const auto& p : policies) {
        for (const std::uint32_t depth : {1U, 3U, 7U}) {
            for (std::uint32_t s = 0; s < 12U; ++s) {
                auto policy = p;
                policy.seed = 8000U + s;
                const auto reference = plain(policy, coupled_options(), 9U + s, 128U, eos);
                for (const auto kind : {DraftKind::identical, DraftKind::close_subset}) {
                    for (unsigned schedule = 0; schedule < 4U; ++schedule) {
                        const auto run = speculative(policy, coupled_options(), kind, depth, schedule, 9U + s, 128U, eos);
                        ++runs;
                        expect(run.tokens == reference.tokens, "EOS run R" + std::to_string(depth + 1U) + " schedule " +
                            std::to_string(schedule) + " differs from plain sampling at " +
                            std::to_string(first_difference(run.tokens, reference.tokens)));
                    }
                }
            }
        }
    }
    // The argmax-draft gate: never (0), the default (0.7) and always below 2 (first-max drafts at T = 1).
    for (const float gate : {0.0F, 0.7F, 2.0F}) {
        for (const float t : {0.5F, 1.0F}) {
            for (std::uint32_t s = 0; s < 4U; ++s) {
                const strix::TokenSamplingPolicy policy{t, 0, 0.95F, 0.0F, 9000U + s};
                auto o = coupled_options();
                o.argmax_draft_below = gate;
                const auto reference = plain(policy, o, 13U + s, 256U, {});
                for (const std::uint32_t depth : {3U, 7U}) {
                    const auto run = speculative(policy, o, DraftKind::close, depth, 3U, 13U + s, 256U, {});
                    ++runs;
                    expect(run.tokens == reference.tokens, "argmax gate " + std::to_string(gate) + " T " + std::to_string(t) +
                        " differs from plain sampling");
                }
            }
        }
    }
    std::cout << runs << " budget-tail, dense-EOS and argmax-gate runs equal plain sampling (" << coverage.eos_stops
              << " runs stopped on EOS, " << coverage.eos_mid_window << " with EOS inside a window; " << coverage.tails
              << " windows cut by the budget)\n";
    expect(coverage.eos_stops > 0U && coverage.eos_mid_window > 0U && coverage.tails > 0U,
        "EOS at window rows after row 0 and budget-cut windows were reached");
}

}  // namespace

int main() {
    test_defaults();
    test_equality();
    test_forced_rejection();
    test_cohort_permutations();
    test_tails_eos_filters();
    expect(coverage.adaptive_cycles > 0U, "the adaptive schedule ran");
    if (failures) {
        std::cerr << failures << " coupled sampling check(s) failed\n";
        return 1;
    }
    std::cout << "coupled sampling tests passed\n";
    return 0;
}
