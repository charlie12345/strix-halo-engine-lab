#pragma once
// MoE router policy: the vendor-free contract every family router binds (host oracle, HIP and CUDA kernels).
//
// A router row is: logits z[e] (the family's product order) -> scores s[e] by the score transform (the family's Math
// trait supplies exp / log1p / sqrt, so the bits stay the family's) -> choice c[e] = s[e] (+ bias[e] when the policy
// has a selection bias) -> top_k experts by the selection key -> output order -> weights.
//
// Declared order (the bits of every binding):
//   selection: k passes; pass k scans the experts in ascending id, skips the chosen ones, and takes e when there is no
//     pick yet or route_better(key, c[e], e, c[pick], pick). float_serial: c[e] > c[pick] (lower id on ties; a NaN
//     taken as the first candidate stays). ordered_bits: route_ordered_key(c) (-0 == +0, a total order on the bits).
//     For finite choices both keys give the same experts: (choice desc, id asc).
//   order: selection order, or ids sorted ascending.
//   weights: sum = ((0 + s[id_0]) + s[id_1]) + ... in OUTPUT order; w_k = (s[id_k] / (sum + renorm_eps)) * scale.
//     renormalize == false: w_k = s[id_k] * scale. top_k == 1 && !renorm_single: w_0 = s[id_0] * scale.
//   groups: n_group = topk_group = 1 only (no bound family uses group-limited routing; anything else is rejected).
//   gfx1151_v1 softmax: 256 lanes, each summing +0, p[e], p[e+256]; halve lower-first through 128..1.
//     Mask NaN probabilities after full division. Empty slots keep the expert-count ID and raw -inf score.
//
// Bound families (presets below, each the bits of the source named): DS41, GLM-5, MiniMax-M3. A family whose binding
// lives in its model package keeps its preset there as data (a RoutePolicy value checked by validate).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <math.h>
#include <span>
#include <stdexcept>
#include <vector>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_MOE_ROUTE_HD __host__ __device__
#else
#define STRIX_MOE_ROUTE_HD
#endif

namespace strix::moe {

enum class RouteScore : std::uint8_t { softmax, sigmoid, sqrt_softplus };
enum class RouteKey : std::uint8_t { float_serial, ordered_bits };
enum class RouteOrder : std::uint8_t { selection, ascending_id };
enum class RouteMath : std::uint8_t { native, gfx1151_v1 };
enum class RouteStatus : std::uint32_t { ok, invalid_route };

inline constexpr unsigned kGfx1151RouteLanes = 256U;
inline constexpr unsigned kGfx1151RouteExperts = 2U * kGfx1151RouteLanes;
inline constexpr unsigned kGfx1151RouteMaxTopK = 16U;

// In selection order, any empty slots are at the end. Publish the status in
// addition to preserving the expert-count sentinel for downstream readers.
STRIX_MOE_ROUTE_HD constexpr RouteStatus route_status_from_last_id(std::uint32_t id, std::uint32_t experts) {
    return id < experts ? RouteStatus::ok : RouteStatus::invalid_route;
}

// Device execution traits. Checked rows write status and use sentinel IDs on failure.
// Explicit round-to-nearest keeps the arithmetic of bindings that require it.
template <bool CheckedStatus = false, bool RoundNearest = false>
struct RouteWaveTraits {
    static constexpr bool checked_status = CheckedStatus;
    static constexpr bool round_nearest = RoundNearest;
};

struct RoutePolicy {
    RouteScore score = RouteScore::sigmoid;
    bool selection_bias = false;
    RouteKey key = RouteKey::float_serial;
    RouteOrder order = RouteOrder::selection;
    std::uint32_t experts = 0;
    std::uint32_t top_k = 0;
    std::uint32_t n_group = 1;
    std::uint32_t topk_group = 1;
    bool renormalize = true;
    bool renorm_single = true;  // false: a single pick keeps its raw score (DS41)
    float renorm_eps = 1e-20F;
    float scale = 1.0F;
    RouteMath math = RouteMath::native;
};

// nullptr when the policy is bindable, else the reason.
constexpr const char* validate(const RoutePolicy& p) {
    if (p.experts == 0U || p.top_k == 0U || p.top_k > p.experts) return "route policy: need 0 < top_k <= experts";
    if (p.n_group != 1U || p.topk_group != 1U) return "route policy: group-limited routing is not bound (n_group = topk_group = 1)";
    if (p.math == RouteMath::gfx1151_v1 &&
        (p.score != RouteScore::softmax || p.selection_bias || p.key != RouteKey::float_serial ||
         p.order != RouteOrder::selection || p.experts != kGfx1151RouteExperts || p.top_k > kGfx1151RouteMaxTopK ||
         (p.renormalize && !p.renorm_single) || p.renorm_eps != 0.0F || p.scale != 1.0F)) {
        return "route policy: gfx1151_v1 needs 512-expert unbiased softmax, top_k <= 16, selection order, and unit scale without epsilon";
    }
    return nullptr;
}

namespace preset {
// src/ds41_moe.cpp reference_router; HIP route_block_topk (hip_ds41_formats.hip); CUDA ds41_core_moe.cu.
constexpr RoutePolicy ds41(std::uint32_t experts, std::uint32_t top_k, float scale) {
    return {RouteScore::sqrt_softplus, true, RouteKey::float_serial, RouteOrder::selection, experts, top_k, 1U, 1U, true, false, 1e-20F, scale};
}
// src/glm/host/glm_host.cpp route; HIP glm_tier_a.hip route_kernel (route_wave_topk). 288 experts, top-8, x 2.5.
inline constexpr RoutePolicy glm5{RouteScore::sigmoid, true, RouteKey::ordered_bits, RouteOrder::ascending_id, 288U, 8U, 1U, 1U, true, true, 1e-20F, 2.5F};
// src/minimax_moe.cpp route_row; HIP hip_minimax_moe.hip route_topk. 128 experts, top-4, x 2.0.
inline constexpr RoutePolicy minimax_m3{RouteScore::sigmoid, true, RouteKey::float_serial, RouteOrder::selection, 128U, 4U, 1U, 1U, true, true, 1e-20F, 2.0F};
}  // namespace preset

// Score transforms of one logit. Math supplies exp / log1p / sqrt with the family's bits.
template <class Math>
STRIX_MOE_ROUTE_HD inline float sigmoid_score(float z) {
    return 1.0F / (1.0F + Math::exp(-z));
}
template <class Math>
STRIX_MOE_ROUTE_HD inline float sqrt_softplus_score(float z) {
    return Math::sqrt(z > 20.0F ? z : Math::log1p(Math::exp(z)));
}

// Declared prefill exp from minimax_msa.hpp. Nearest-even Cody-Waite range
// reduction and degree-eight Horner polynomial, with the donor's cutoffs.
struct RouteCodyWaiteExpV1 {
    STRIX_MOE_ROUTE_HD static float exp(float x) {
        if (!(x >= -87.0F)) return x < -87.0F ? 0.0F : x;
        if (x > 88.0F) return __builtin_bit_cast(float, std::uint32_t{0x7f800000U});
        const float n = ::rintf(x * 1.44269504088896341F);
        float r = ::fmaf(n, -0.693145751953125F, x);
        r = ::fmaf(n, -1.42860676533018e-06F, r);
        float p = 2.48015873e-05F;
        p = ::fmaf(p, r, 1.98412698e-04F);
        p = ::fmaf(p, r, 1.38888889e-03F);
        p = ::fmaf(p, r, 8.33333333e-03F);
        p = ::fmaf(p, r, 4.16666667e-02F);
        p = ::fmaf(p, r, 1.66666667e-01F);
        p = ::fmaf(p, r, 0.5F);
        p = ::fmaf(p, r, 1.0F);
        p = ::fmaf(p, r, 1.0F);
        const auto bits = static_cast<std::uint32_t>(static_cast<int>(n) + 127) << 23U;
        return p * __builtin_bit_cast(float, bits);
    }
};

struct RouteScorePair {
    float unbiased;
    float biased;
};

// Token-major row/expert views. Separate arrays use stride 1. An interleaved
// buffer uses stride 2 with bases at element 0 and element 1.
template <unsigned Stride>
struct RouteScorePlanes {
    static_assert(Stride > 0U);
    float* unbiased;
    float* biased;
};

template <class Math>
STRIX_MOE_ROUTE_HD inline RouteScorePair route_sigmoid_score_pair(float logit, float bias) {
    const float score = sigmoid_score<Math>(logit);
    return {score, score + bias};
}

// Lane l folds dimensions l, l+32, ... by FP32 FMA. The reduction uses XOR
// masks 16, 8, 4, 2, 1; the host computes the operands that reach lane zero.
struct RouteStridedFmaXor32 {
    static constexpr unsigned lanes = 32U;

    static float dot_ref(std::span<const float> hidden, std::span<const float> weight) {
        if (hidden.empty() || hidden.size() != weight.size())
            throw std::invalid_argument("route projection: expected equal nonempty input and weight spans");
        std::array<float, lanes> partial{};
        for (unsigned lane = 0; lane < lanes; ++lane)
            for (std::size_t j = lane; j < hidden.size(); j += lanes)
                partial[lane] = std::fma(hidden[j], weight[j], partial[lane]);
        for (unsigned mask = lanes / 2U; mask > 0U; mask >>= 1U)
            for (unsigned lane = 0; lane < mask; ++lane)
                partial[lane] = partial[lane] + partial[lane ^ mask];
        return partial[0];
    }
};

template <class Dot, class Math>
RouteScorePair route_project_sigmoid_ref(std::span<const float> hidden, std::span<const float> weight, float bias) {
    return route_sigmoid_score_pair<Math>(Dot::dot_ref(hidden, weight), bias);
}

// Selection key: -0 is +0; negatives below positives; a total order on the remaining bits.
STRIX_MOE_ROUTE_HD inline std::uint32_t route_ordered_key(float v) {
    std::uint32_t u = __builtin_bit_cast(std::uint32_t, v);
    if (u == 0x80000000U) u = 0;
    return (u & 0x80000000U) ? ~u : (u | 0x80000000U);
}
// True when candidate a (id ai) replaces the current pick b (id bi).
STRIX_MOE_ROUTE_HD inline bool route_better(RouteKey key, float a, std::uint32_t ai, float b, std::uint32_t bi) {
    if (key == RouteKey::float_serial) return a > b;
    const std::uint32_t ka = route_ordered_key(a), kb = route_ordered_key(b);
    return ka > kb || (ka == kb && ai < bi);
}

inline void validate_route_row(const RoutePolicy& p, std::size_t scores, std::size_t bias,
                               std::size_t ids, std::size_t weights) {
    if (const char* why = validate(p)) throw std::invalid_argument(why);
    if (scores < p.experts || (p.selection_bias && bias < p.experts) || ids < p.top_k || weights < p.top_k)
        throw std::invalid_argument("route policy: row spans are smaller than the policy");
}

// Shared single-row selection. Callers validate spans and supply experts bytes of scratch.
// Inputs are already transformed scores. Bias changes selection only.
inline void route_select_row(const RoutePolicy& p, const float* scores, const float* bias,
                             std::uint32_t* ids, float* weights, std::uint8_t* taken) {
    const bool gfx1151 = p.math == RouteMath::gfx1151_v1;
    constexpr float empty = -std::numeric_limits<float>::infinity();
    for (std::uint32_t e = 0; e < p.experts; ++e) taken[e] = 0U;
    for (std::uint32_t k = 0; k < p.top_k; ++k) {
        std::uint32_t pick = p.experts;
        float best = 0.0F;
        for (std::uint32_t e = 0; e < p.experts; ++e) {
            if (taken[e]) continue;
            const float c = p.selection_bias ? scores[e] + bias[e] : scores[e];
            if (gfx1151 && std::isnan(c)) continue;
            if (pick == p.experts || route_better(p.key, c, e, best, pick)) {
                pick = e;
                best = c;
            }
        }
        if (pick != p.experts) taken[pick] = 1U;
        ids[k] = pick;
    }
    if (p.order == RouteOrder::ascending_id) {
        for (std::uint32_t k = 1; k < p.top_k; ++k) {
            const auto expert = ids[k];
            auto slot = k;
            while (slot > 0U && ids[slot - 1U] > expert) {
                ids[slot] = ids[slot - 1U];
                --slot;
            }
            ids[slot] = expert;
        }
    }
    const bool renorm = p.renormalize && (p.top_k > 1U || p.renorm_single);
    float sum = 0.0F;
    for (std::uint32_t k = 0; k < p.top_k; ++k) sum = sum + (ids[k] == p.experts ? empty : scores[ids[k]]);
    const float denominator = sum + p.renorm_eps;
    for (std::uint32_t k = 0; k < p.top_k; ++k) {
        const float s = ids[k] == p.experts ? empty : scores[ids[k]];
        weights[k] = (renorm ? s / denominator : s) * p.scale;
    }
}

inline void route_select_ref(const RoutePolicy& p, std::span<const float> scores, std::span<const float> bias,
                             std::span<std::uint32_t> ids, std::span<float> weights) {
    validate_route_row(p, scores.size(), bias.size(), ids.size(), weights.size());
    std::vector<std::uint8_t> taken(p.experts);
    route_select_row(p, scores.data(), bias.data(), ids.data(), weights.data(), taken.data());
}

// Host transform of one row: softmax (m = max over e ascending, p_e = exp(z_e - m), sum ascending from 0, s_e = p_e / sum),
// sigmoid or sqrt_softplus per element.
template <class Math>
void route_scores_ref(RouteScore score, std::span<const float> logits, std::span<float> scores) {
    if (score == RouteScore::softmax) {
        float m = -std::numeric_limits<float>::infinity();
        for (const float z : logits) m = z > m ? z : m;
        float sum = 0.0F;
        for (std::size_t e = 0; e < logits.size(); ++e) {
            scores[e] = Math::exp(logits[e] - m);
            sum = sum + scores[e];
        }
        for (std::size_t e = 0; e < logits.size(); ++e) scores[e] = scores[e] / sum;
        return;
    }
    if (score == RouteScore::sigmoid) {
        for (std::size_t e = 0; e < logits.size(); ++e) scores[e] = sigmoid_score<Math>(logits[e]);
        return;
    }
    // sqrt_softplus needs Math::log1p and Math::sqrt; a sigmoid-only Math (GLM, MiniMax) need not supply them.
    if constexpr (requires { Math::log1p(0.0F); Math::sqrt(0.0F); }) {
        for (std::size_t e = 0; e < logits.size(); ++e) scores[e] = sqrt_softplus_score<Math>(logits[e]);
    } else {
        throw std::invalid_argument("route policy: Math has no log1p / sqrt for sqrt_softplus");
    }
}

// CPU oracle for the gfx1151 softmax tree. The callable binds the engine's exp
// implementation and tables.
template <class Exp>
void route_gfx1151_scores_ref(std::span<const float, kGfx1151RouteExperts> logits,
                             std::span<float, kGfx1151RouteExperts> scores, const Exp& exp) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (const float z : logits) maximum = std::fmax(maximum, z);
    std::array<float, kGfx1151RouteLanes> partial{};
    for (unsigned lane = 0; lane < kGfx1151RouteLanes; ++lane) {
        const auto upper = lane + kGfx1151RouteLanes;
        scores[lane] = exp(logits[lane] - maximum);
        scores[upper] = exp(logits[upper] - maximum);
        partial[lane] = (0.0F + scores[lane]) + scores[upper];
    }
    for (unsigned stride = kGfx1151RouteLanes / 2U; stride != 0U; stride >>= 1U)
        for (unsigned lane = 0; lane < stride; ++lane) partial[lane] = partial[lane] + partial[lane + stride];
    for (float& score : scores) score = score / partial[0];
}

// Reuse one score buffer over token-major rows. Weights retain the order and
// normalization of the route policy, independent of any later grouping.
template <class Math>
void route_rows_ref(const RoutePolicy& policy, std::span<const float> logits, std::span<const float> bias,
                    std::span<std::uint32_t> ids, std::span<float> weights) {
    if (const char* why = validate(policy)) throw std::invalid_argument(why);
    const auto rows = logits.size() / policy.experts;
    if (logits.size() % policy.experts != 0 || (policy.selection_bias && bias.size() != policy.experts) ||
        ids.size() % policy.top_k != 0 || ids.size() / policy.top_k != rows || weights.size() != ids.size()) {
        throw std::invalid_argument("route policy: invalid token-major geometry");
    }
    if (rows == 0) return;
    std::vector<float> scores(policy.experts);
    for (std::size_t row = 0; row < rows; ++row) {
        const auto first = row * policy.top_k;
        const auto input = logits.subspan(row * policy.experts, policy.experts);
        if (policy.math == RouteMath::gfx1151_v1)
            route_gfx1151_scores_ref(input.first<kGfx1151RouteExperts>(),
                                     std::span{scores}.first<kGfx1151RouteExperts>(),
                                     [](float z) { return Math::exp(z); });
        else route_scores_ref<Math>(policy.score, input, scores);
        route_select_ref(policy, scores, bias, ids.subspan(first, policy.top_k), weights.subspan(first, policy.top_k));
    }
}

}  // namespace strix::moe
