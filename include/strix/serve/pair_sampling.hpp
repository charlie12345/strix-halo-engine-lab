#pragma once

#include "strix/serve/engine.hpp"
#include "strix/serve/job_sampling.hpp"
#include "strix/token_sampling.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix::serve {

// The follower replays the lead's seeds only while every request shares identity 1 (the shared mode switch).
inline bool pair_sampling_replays_seeds() { return !token_sampling_request_identity_counts(); }

// A per-request sampler held by value in the pair schedulers (the shared translation, strix/serve/job_sampling.hpp).
// `acceptance` decides drafted rows only (a route without drafts ignores it).
inline std::optional<TokenSampler> make_pair_sampler(const GenerationJob& job,
                                                     SampledAcceptance acceptance = SampledAcceptance::rejection) {
    const auto shared = make_job_sampler(job.sampling, acceptance);
    if (!shared) return std::nullopt;
    return std::optional<TokenSampler>(std::in_place, shared->policy(), shared->options());
}

struct PairSamplingRow {
    TokenSampler* sampler = nullptr;
    std::uint64_t position = 0;
};

namespace detail {
inline constexpr std::size_t kPairSliceFrameFloats = 16384;  // 64 KiB control-frame limit

// One row's follower slice moves to the lead in frames of at most kPairSliceFrameFloats floats (the full-duplex control
// transports need equal message lengths, so the lead sends padding). Lead: `local` is its slice and `full` receives
// local + follower slice; follower: `local` is its slice (`full` unused).
template<class Exchange>
void move_pair_slice(bool lead, std::span<const float> local, std::size_t follower_width, std::vector<float>& full,
                     std::vector<float>& scratch, Exchange&& exchange) {
    if (lead) {
        full.resize(local.size() + follower_width);
        std::copy(local.begin(), local.end(), full.begin());
    } else if (local.size() != follower_width) {
        throw std::logic_error("pair sampling: wrong follower vocabulary slice");
    }
    scratch.resize(std::min(kPairSliceFrameFloats, follower_width));
    for (std::size_t begin = 0; begin < follower_width; begin += kPairSliceFrameFloats) {
        const std::size_t count = std::min(kPairSliceFrameFloats, follower_width - begin);
        auto buffer = std::span(scratch).first(count);
        if (lead) exchange(std::span<const float>(buffer), std::span(full).subspan(local.size() + begin, count));
        else exchange(local.subspan(begin, count), buffer);
    }
}
}  // namespace detail

// The follower sends its vocabulary slice; only the lead assembles the row and samples it.
// A zero follower width is the one-node route. A zero mask leaves the greedy picks in place.
template<class Read, class Exchange>
void sample_pair_rows(std::uint32_t mask, bool lead, std::size_t follower_width, std::size_t vocabulary,
                      std::span<const PairSamplingRow> choices, std::span<std::uint32_t> tokens,
                      Read&& read, Exchange&& exchange) {
    if (mask == 0) return;
    std::vector<float> scratch, full;
    for (std::size_t row = 0; mask != 0; ++row, mask >>= 1) {
        if ((mask & 1U) == 0) continue;
        const std::span<const float> local = read(row);
        if (lead && (row >= choices.size() || row >= tokens.size() || !choices[row].sampler))
            throw std::logic_error("pair sampling: missing lead sampler");
        detail::move_pair_slice(lead, local, follower_width, full, scratch, exchange);
        if (lead) {
            const auto logits = std::span<const float>(full).first(std::min(vocabulary, full.size()));
            tokens[row] = choices[row].sampler->decide(logits, choices[row].position, nullptr).token;
        }
    }
}

// The lead's message after each row of a window: [kPairWindowMore] while another row follows, then
// [kPairWindowDone, n, token 0 .. token n-1, 0 ...].
inline constexpr std::size_t kPairWindowWords = 10;
inline constexpr std::uint32_t kPairWindowDone = 0, kPairWindowMore = 1;

// One sampled verify window across the pair. Row r of the window decides the token at first_position + r with
// TokenSampler::decide_rows (drafts[r] is the draft row r checks; no EOS stop): it stops after the first row without
// a draft, the first rejected draft or max_rows rows. Under coupled acceptance a draft only needs its token; under
// rejection it needs the distribution it was drawn from. Every row the lead reaches is assembled from both ranks'
// vocabulary slices (the follower's slice moves as in sample_pair_rows); after each row the lead tells the follower
// whether another follows, then sends the decided tokens. Both ranks return the same tokens (the lead's sampler
// counts the decisions). The follower ignores sampler, first_position, drafts and max_rows. follower_width 0: one
// node, no exchange. `words` exchanges kPairWindowWords words (the follower sends zeros).
template<class Read, class ExchangeFloats, class ExchangeWords>
std::vector<std::uint32_t> decide_pair_window(bool lead, std::size_t follower_width, std::size_t vocabulary,
                                              TokenSampler* sampler, std::uint64_t first_position,
                                              std::span<const TokenDraft> drafts, std::uint32_t max_rows,
                                              Read&& read, ExchangeFloats&& exchange, ExchangeWords&& words) {
    const bool paired = follower_width != 0;
    std::vector<float> scratch, full;
    std::array<std::uint32_t, kPairWindowWords> send{}, receive{};
    if (!lead) {
        if (!paired) throw std::logic_error("pair window: a follower needs a vocabulary slice");
        for (std::size_t row = 0;; ++row) {
            detail::move_pair_slice(false, read(row), follower_width, full, scratch, exchange);
            words(std::span<const std::uint32_t>(send), std::span<std::uint32_t>(receive));
            if (receive[0] == kPairWindowMore) continue;
            if (receive[0] != kPairWindowDone || receive[1] != row + 1U || receive[1] > kPairWindowWords - 2U)
                throw std::runtime_error("pair window: the lead's decision message is out of protocol");
            return {receive.begin() + 2, receive.begin() + 2 + receive[1]};
        }
    }
    if (sampler == nullptr) throw std::logic_error("pair window: missing lead sampler");
    if (max_rows == 0 || max_rows > kPairWindowWords - 2U || drafts.size() + 1U < max_rows)
        throw std::invalid_argument("pair window: max_rows must be 1 .. drafts + 1 (at most 8)");
    std::uint32_t reached = 0;
    const auto logits = [&](std::uint32_t row) -> std::span<const float> {
        if (row != reached) throw std::logic_error("pair window: rows out of order");
        if (paired && row != 0) {  // the previous row continues the window: the follower sends this one
            send.fill(0);
            send[0] = kPairWindowMore;
            words(std::span<const std::uint32_t>(send), std::span<std::uint32_t>(receive));
        }
        ++reached;
        const std::span<const float> local = read(row);
        if (!paired) return local.first(std::min(vocabulary, local.size()));
        detail::move_pair_slice(true, local, follower_width, full, scratch, exchange);
        return std::span<const float>(full).first(std::min(vocabulary, full.size()));
    };
    const auto decisions = sampler->decide_rows(max_rows, first_position, drafts, {}, logits);
    std::vector<std::uint32_t> tokens;
    tokens.reserve(decisions.size());
    for (const auto& d : decisions) tokens.push_back(d.token);
    if (paired) {
        send.fill(0);
        send[0] = kPairWindowDone;
        send[1] = static_cast<std::uint32_t>(tokens.size());
        std::copy(tokens.begin(), tokens.end(), send.begin() + 2);
        words(std::span<const std::uint32_t>(send), std::span<std::uint32_t>(receive));
    }
    return tokens;
}

}  // namespace strix::serve
