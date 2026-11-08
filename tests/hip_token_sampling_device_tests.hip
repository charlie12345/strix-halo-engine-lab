// Device sampling-v1 against the host transform: for random logit rows, policies, drafts and uniforms the
// device decision must equal decide_token(make_token_distribution(...)) (the device exp and, for large kept sets,
// the parallel summation order could only change a uniform within a few double ulps of a cumulative boundary;
// none is expected over these draws). Drafts are read from pinned host memory or from device memory. Device
// draft steps over draft-head-like entry lists must give the host's token and distribution.
#include "strix/hip_token_sampling_device.hpp"
#include "strix/token_sampling.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strix;

namespace {
void require(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }

// Logits shaped like a language model row: a few strong tokens, a long tail, some exact ties.
std::vector<float> make_logits(std::mt19937_64& rng, std::uint32_t vocab, float spread) {
    std::normal_distribution<float> tail(-4.0F, spread);
    std::vector<float> out(vocab);
    for (auto& v : out) v = tail(rng);
    std::uniform_int_distribution<std::uint32_t> pick(0, vocab - 1U);
    for (unsigned k = 0; k < 24U; ++k) out[pick(rng)] = 12.0F + static_cast<float>(k % 6U) * 0.75F;
    for (unsigned k = 0; k < 64U; ++k) out[pick(rng)] = out[pick(rng)]; // ties
    return out;
}
} // namespace

int main(int argc, char** argv) try {
    const unsigned cases = argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 400U;
    std::mt19937_64 rng(0x5eedULL);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const TokenSamplingPolicy policies[] = {
        {1.0F, 0U, 1.0F, 0.0F, 0U}, {0.7F, 0U, 1.0F, 0.0F, 0U}, {0.6F, 0U, 1.0F, 0.1F, 0U}, {0.7F, 40U, 1.0F, 0.0F, 0U},
        {1.0F, 5U, 1.0F, 0.0F, 0U}, {1.0F, 0U, 0.8F, 0.0F, 0U}, {0.7F, 40U, 0.9F, 0.05F, 0U}, {2.0F, 0U, 0.95F, 0.0F, 0U},
        {0.3F, 0U, 1.0F, 0.0F, 0U}, {1.0F, 3000U, 1.0F, 0.0F, 0U}, {1.5F, 0U, 0.5F, 0.0F, 0U}, {0.7F, 0U, 0.9F, 0.0F, 0U},
        {1.0F, 0U, 0.95F, 0.0F, 0U}, {0.8F, 64U, 0.95F, 0.0F, 0U}, {0.9F, 65U, 1.0F, 0.0F, 0U}, {1.0F, 0U, 0.9F, 0.02F, 0U},
        {1.2F, 20U, 0.9F, 0.0F, 0U}, {0.5F, 0U, 0.99F, 0.0F, 0U}, {1.0F, 50U, 1.0F, 0.0F, 0U},
    };
    unsigned decided = 0, drafted = 0, accepted = 0, residual = 0, fallbacks = 0, device_q = 0;
    for (unsigned c = 0; c < cases; ++c) {
        const auto& policy = policies[c % std::size(policies)];
        const std::uint32_t vocab = c % 3U == 0U ? 248320U : c % 3U == 1U ? 151936U : 4099U;
        const auto logits = make_logits(rng, vocab, c % 2U ? 2.0F : 3.5F);
        // Draft: the same transform over a perturbed row restricted to a token subset (like the draft vocabulary).
        TokenDraft draft;
        const bool with_draft = c % 4U != 0U;
        if (with_draft) {
            std::vector<std::uint32_t> ids;
            std::vector<float> values;
            std::normal_distribution<float> noise(0.0F, 0.8F);
            for (std::uint32_t t = 0; t < vocab; ++t)
                if (t % 3U != 1U) { ids.push_back(t); values.push_back(logits[t] + noise(rng)); }
            draft.position = 7U;
            draft.distribution = make_token_distribution(values, ids, policy);
            draft.token = draft.distribution.sample(uniform(rng));
        }
        const double u_plain = uniform(rng), u_accept = uniform(rng), u_residual = uniform(rng);
        const auto host = decide_token(make_token_distribution(logits, {}, policy), with_draft ? &draft : nullptr,
            u_plain, with_draft ? u_accept : 0.0, with_draft ? u_residual : 0.0);
        bool fallback = false;
        const bool q_on_device = with_draft && c % 2U == 1U;
        device_q += q_on_device ? 1U : 0U;
        const auto device = hip_decide_token_for_testing(logits, policy, with_draft ? &draft : nullptr, u_plain,
            with_draft ? u_accept : 0.0, with_draft ? u_residual : 0.0, &fallback, q_on_device);
        fallbacks += fallback ? 1U : 0U;
        require(device.token == host.token && device.drafted == host.drafted && device.accepted == host.accepted &&
                device.residual == host.residual,
            "device decision differs from the host transform in case " + std::to_string(c) + ": token " +
                std::to_string(device.token) + " vs " + std::to_string(host.token));
        ++decided; drafted += host.drafted; accepted += host.accepted; residual += host.residual;
    }
    require(fallbacks == 0U, std::to_string(fallbacks) + " rows had no device decision");
    // Device draft steps: entry lists like the draft head's (an ascending base list, then context ids appended
    // in arrival order) or the identity, for every policy.
    unsigned draft_steps = 0;
    double max_relative = 0.0;
    for (unsigned c = 0; c < cases / 2U; ++c) {
        const auto& policy = policies[c % std::size(policies)];
        std::vector<std::uint32_t> ids;
        const std::uint32_t vocab = 248320U;
        if (c % 5U != 0U) {
            std::uniform_int_distribution<std::uint32_t> pick(0, vocab - 1U);
            std::vector<std::uint8_t> seen(vocab, 0U);
            for (std::uint32_t t = 0; t < vocab && ids.size() < 32768U; t += 1U + pick(rng) % 13U) { ids.push_back(t); seen[t] = 1U; }
            for (unsigned k = 0; k < 4096U; ++k) { const auto t = pick(rng); if (!seen[t]) { seen[t] = 1U; ids.push_back(t); } }
        }
        const auto entries = ids.empty() ? vocab : static_cast<std::uint32_t>(ids.size());
        const auto logits = make_logits(rng, entries, c % 2U ? 2.0F : 3.5F);
        const double u = uniform(rng);
        const auto host = make_token_distribution(logits, ids, policy);
        const auto host_token = host.sample(u);
        const auto device = hip_draft_for_testing(logits, ids, policy, u);
        require(device.token == host_token, "device draft token differs from the host in case " + std::to_string(c) + ": " +
            std::to_string(device.token) + " vs " + std::to_string(host_token));
        const auto& d = device.distribution;
        require(d.tokens().size() == host.tokens().size(), "device draft support size differs in case " + std::to_string(c));
        for (const auto t : host.tokens()) {
            const double h = host.probability(t), g = d.probability(t);
            require(g > 0.0, "device draft support differs in case " + std::to_string(c));
            max_relative = std::max(max_relative, std::abs(g - h) / h);
        }
        ++draft_steps;
    }
    require(max_relative < 1e-12, "device draft probabilities differ from the host by " + std::to_string(max_relative));
    // A non-finite logit is rejected as on the host.
    {
        std::vector<float> bad(4099U, 0.0F);
        bad[17] = NAN;
        bool rejected = false;
        try { (void)hip_decide_token_for_testing(bad, policies[0], nullptr, 0.5, 0.0, 0.0); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "device sampling accepted a non-finite logit");
    }
    // Coupled acceptance through HipTokenSamplerDevice: windows of drafted rows (drafts equal to the target row or
    // drawn from a perturbed row) decided on the device equal the host coupled sampler row by row. Draft steps with
    // the first-max draft transform (top-k 1, coupled below the argmax gate) give the first-max entry.
    unsigned coupled_rows = 0, coupled_accepted = 0;
    std::uint64_t coupled_forced = 0;
    {
        constexpr std::uint32_t vocab = 151936U, window = 4U;
        HipTokenSamplerDevice device(vocab, 8U);
        float* rows_device = nullptr;
        require(hipMalloc(reinterpret_cast<void**>(&rows_device), 4ULL * vocab * window) == hipSuccess, "coupled logits allocation");
        hipStream_t stream = nullptr;
        require(hipStreamCreate(&stream) == hipSuccess, "coupled stream");
        TokenSamplerOptions coupled;
        coupled.acceptance = SampledAcceptance::coupled;
        for (unsigned c = 0; c < std::max(8U, cases / 8U); ++c) {
            auto policy = policies[c % std::size(policies)];
            policy.seed = 77U + c;
            TokenSampler host(policy, coupled), on_device(policy, coupled);
            std::vector<std::vector<float>> logits(window);
            std::vector<TokenDraft> drafts(window);
            std::vector<DeviceSampleRow> rows(window);
            for (std::uint32_t r = 0; r < window; ++r) {
                logits[r] = make_logits(rng, vocab, c % 2U ? 2.0F : 3.5F);
                require(hipMemcpy(rows_device + static_cast<std::size_t>(r) * vocab, logits[r].data(), 4ULL * vocab,
                    hipMemcpyHostToDevice) == hipSuccess, "coupled logits upload");
                const std::uint64_t position = 1000U + 17U * c + r;
                auto draft_row = logits[r];
                if ((r + c) % 2U) { std::normal_distribution<float> noise(0.0F, 0.8F); for (auto& v : draft_row) v += noise(rng); }
                drafts[r] = host.draft(draft_row, {}, position);
                ExecutionBufferAddress address{};
                address.address = reinterpret_cast<std::uintptr_t>(rows_device + static_cast<std::size_t>(r) * vocab);
                address.bytes = 4ULL * vocab;
                rows[r] = {address, position, r + 1U < window ? &drafts[r] : nullptr};
            }
            const auto decisions = device.decide(on_device, rows, stream);
            for (std::uint32_t r = 0; r < window; ++r) {
                const auto expected = host.decide(logits[r], rows[r].position, rows[r].draft);
                require(decisions[r].token == expected.token && decisions[r].drafted == expected.drafted &&
                        decisions[r].accepted == expected.accepted && !decisions[r].residual,
                    "coupled device decision differs from the host in case " + std::to_string(c) + " row " + std::to_string(r));
                ++coupled_rows; coupled_accepted += expected.accepted;
            }
            // Forced rejection (test-only switch) on device decisions, applied as the window verifier does
            // (begin_window, then force_rejection per row before the stopping rule), equals host decide_rows.
            auto forced = coupled;
            forced.force_reject_row = c % 3U == 2U ? TokenSamplerOptions::forced_rejection_rotate : c % 3U;
            TokenSampler forced_host(policy, forced), forced_device(policy, forced);
            const std::span<const TokenDraft> window_drafts(drafts.data(), window - 1U);
            const auto host_rows = forced_host.decide_rows(window, rows[0].position, window_drafts, {},
                [&](std::uint32_t r) { return std::span<const float>(logits[r]); });
            auto device_rows = device.decide(forced_device, rows, stream);
            forced_device.begin_window();
            std::vector<TokenDecision> device_kept;
            for (std::uint32_t r = 0; r < window; ++r) {
                forced_device.force_rejection(device_rows[r], r);
                device_kept.push_back(device_rows[r]);
                forced_device.count(device_rows[r]);
                if (!rows[r].draft || !device_rows[r].accepted) break;
            }
            require(device_kept.size() == host_rows.size(), "forced device window length differs from the host in case " + std::to_string(c));
            for (std::size_t r = 0; r < host_rows.size(); ++r)
                require(device_kept[r].token == host_rows[r].token && device_kept[r].accepted == host_rows[r].accepted &&
                        device_kept[r].forced == host_rows[r].forced,
                    "forced device decision differs from the host in case " + std::to_string(c) + " row " + std::to_string(r));
            require(forced_device.stats().forced == forced_host.stats().forced, "forced rejection counts differ");
            coupled_forced += forced_host.stats().forced;
        }
        require(device.host_fallbacks() == 0U, "coupled rows had no device decision");
        (void)hipStreamDestroy(stream);
        (void)hipFree(rows_device);
        for (unsigned c = 0; c < 16U; ++c) {
            TokenSamplingPolicy policy = policies[c % std::size(policies)];
            policy.top_k = 1U;
            const auto logits = make_logits(rng, 4099U, 2.0F);
            const auto first = static_cast<std::uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            require(hip_draft_for_testing(logits, {}, policy, uniform(rng)).token == first, "top-k 1 device draft is not the first max");
        }
    }
    std::cout << "device sampling-v1: " << decided << " decisions equal the host (" << drafted << " drafted, " << device_q
              << " with the draft distribution in device memory, " << accepted << " accepted, " << residual << " residual, "
              << fallbacks << " without a device decision); " << draft_steps
              << " device draft steps equal the host (token and support; max relative probability difference "
              << max_relative << "); " << coupled_rows << " coupled device rows equal the host (" << coupled_accepted
              << " drafts accepted; " << coupled_forced << " forced rejections equal the host)\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "device sampling test failed: " << error.what() << '\n';
    return 1;
}
