#pragma once

#include "strix/engine_profile.hpp"
#include "strix/weight/levers.hpp"
#include "strix/weight/registry.hpp"

#include <array>
#include <string_view>

namespace strix::inkling2 {

// The resident binding supplies its consumer formats and complete-model receipts. The lever is the shared
// generic policy weights.attention_head.q8_0@1; records under its former model-named id still resolve to it.
inline constexpr weight::WeightLoadPolicy kMiniWeightLoadPolicy = weight::with_legacy(
    weight::kAttentionHeadQ8Policy, {"inkling2-mini-attention-head-bf16-q8_0", 1});

// One resident image serves both regimes (prompt chunks and one-row steps share it, in the CLI and in a served
// cohort step), so its storage changes only when every regime has a gate v2 PASS row and one of them a measured gain.
inline constexpr std::array<std::string_view, 2> kMiniWeightLoadRegimes{weight::kPrefillRegime, weight::kDecodeRegime};

// Gate v2 rows of the shared policy (receipt ids name the evidence register entry; detailed receipts stay outside
// the repository). PREFILL, gfx1151, HIP 7.16: PASS, 39 windows (prose / code / structured at 8K, 32K and 64K),
// 2,457 pooled rows; no measured gain in this regime (prompt projections are WMMA-bound: 8K 4,684 -> 4,673 ms,
// 32K 20,650 -> 20,732 ms), so the row is gate-only and admits prefill onto a Q8_0 image a decode row selects.
// DECODE: no row (gate v2 HOLD on one reference window by rule R4).
inline constexpr weight::WeightLoadDecision kMiniWeightLoadRows[] = {
    {kMiniWeightLoadPolicy.id,
     {weight::Backend::hip, "gfx1151", "hip-7.16", weight::kPrefillRegime,
      weight::digest_from_hex("06b924b4f4116787df43a3668795417070ca22669e6314fd3bb8250eb29bd9c6"),
      weight::digest_from_hex("65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1")},
     weight::digest_from_hex("d64b562553020001310a9ad9831d4197fb17ae47ae8cc49fa3fec80fdc51518c"), {},
     {"docs/decisions/evidence-register.md#inkling-2-mini-q8_0-attention-and-head-on-gfx1151", "2026-11-23", 2457,
      65536}},
};
inline constexpr std::span<const weight::WeightLoadDecision> kMiniWeightLoadDecisions{kMiniWeightLoadRows};

// `context`: the largest context the instance admits (served: its configured context; serial CLI: the model limit,
// fixed before the prompt is read). A banded row applies only inside its band.
inline weight::WeightLoadPlan resolve_mini_weight_load(
    profile::Name tier, const weight::WeightLoadScope& scope,
    std::span<const weight::FormatId> consumer_formats, bool enabled, bool qualify,
    std::span<const weight::WeightLoadDecision> decisions = kMiniWeightLoadDecisions, std::uint32_t context = 0) {
    return weight::resolve_weight_load(weight::builtin_registry(), kMiniWeightLoadPolicy,
        {tier, scope, consumer_formats, enabled, qualify, kMiniWeightLoadRegimes, context}, decisions);
}

// FAST lever: W4A4 experts. Multi-row prompt-prefill chunks run the NVFP4 expert layers through the shared grouped W4A4
// operand class (moe_core/grouped_w4a4.hpp, identity grouped-nvfp4-w4a4-f32-v1: activations quantised once per projection
// input by the family NVFP4 rule, sm_120 block-scaled FP4 MMA, weights used as stored). MXFP8 layers, decode, verify and
// one-row calls keep the BF16 tile. CUDA sm_120 only (HIP ignores it). Runnable, not ready: opt in with
// STRIX_INK_MOE_W4A4=1 until the FAST gate (prefill regime) and a prefill receipt exist. Off switch: STRIX_INK_MOE_W4A4=0.
inline constexpr std::string_view kMiniMoeW4a4 = "inkling2-mini-moe-w4a4@1";
inline constexpr std::array<profile::Lever, 1> kMiniLevers{{
    {kMiniMoeW4a4, "STRIX_INK_MOE_W4A4", "", {"", "1", "", ""}, "0", profile::Kind::bit_changing,
     {}, {false, true, false, true}, "1", "0",
     "waits for the FAST gate (prefill regime, routes replayed) and an RTX prefill receipt",
     false, {}, true},
}};

// The lever table must stay bound here: an unbound row is never resolved, so its switch is silently ignored.
inline constexpr profile::Engine kMiniProfile{
    "inkling2-mini", "", kMiniLevers, false, "inkling2-mini-turbo-v1",
};

}  // namespace strix::inkling2
