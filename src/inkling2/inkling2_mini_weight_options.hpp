#pragma once

#include "moe_core/grouped_w4a4.hpp"
#include "strix/profiles/inkling2_mini.hpp"

#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::inkling2 {

struct MiniWeightOptions {
    profile::Inputs tier;
    bool qualify_q8{};
    bool disable_q8{};
};

template <class Value>
bool parse_mini_weight_option(MiniWeightOptions& options, std::string_view argument, const Value& value,
                              bool native) {
    if (argument == "--profile") {
        const std::string_view tier(value());
        if (tier != "exact" && tier != "fast" && tier != "turbo")
            throw std::invalid_argument("--profile must be exact, fast or turbo");
        options.tier.profile = profile::parse(tier);
    } else if (native && argument == "--qualify-q8") {
        options.qualify_q8 = true;
    } else if (native && argument == "--no-weight-q8") {
        options.disable_q8 = true;
    } else {
        return false;
    }
    return true;
}

// Resolved operand class of the W4A4 expert lever (multi-row prefill chunks run NVFP4 expert layers through the shared
// grouped W4A4 class on CUDA): 0 = bf16 (off), 1 = nvfp4 (nearest@1), mse = nvfp4_mse, mse-gate-up = nvfp4_mse_gate_up,
// mse-r2-gate-up = nvfp4_mse_r2_gate_up, mse-r2 = nvfp4_mse_r2 (two-term activation rule r2@1 of moe_core/nvfp4_act_rule.hpp),
// r2 = nvfp4_r2 (two nearest@1 terms on both projections).
// Every value but 0 / 1 is a candidate of the route-flip re-qualification, not yet gated.
inline moe_core::grouped_w4a4::Activations mini_moe_w4a4_class(const profile::Resolution& tier) {
    using A = moe_core::grouped_w4a4::Activations;
    const auto* entry = tier.find(kMiniMoeW4a4);
    if (entry == nullptr || entry->value == "0") return A::bf16;
    if (entry->value == "1") return A::nvfp4;
    if (entry->value == "mse") return A::nvfp4_mse;
    if (entry->value == "mse-gate-up") return A::nvfp4_mse_gate_up;
    if (entry->value == "mse-r2-gate-up") return A::nvfp4_mse_r2_gate_up;
    if (entry->value == "mse-r2") return A::nvfp4_mse_r2;
    if (entry->value == "r2") return A::nvfp4_r2;
    throw std::invalid_argument("STRIX_INK_MOE_W4A4 must be 0, 1, mse, mse-gate-up, mse-r2-gate-up, mse-r2 or r2");
}

// Resolved state of the W4A4 expert lever (true: multi-row prefill chunks run NVFP4 expert layers as W4A4 on CUDA).
inline bool mini_moe_w4a4(const profile::Resolution& tier) {
    return mini_moe_w4a4_class(tier) != moe_core::grouped_w4a4::Activations::bf16;
}

// The W4A4 class a run on `backend` ("hip", "cuda") executes. Only the CUDA backend holds the grouped W4A4 tile: a
// selected class that the backend cannot run is refused, never replaced by the BF16 tile under the class's name. Every
// entry point (generation CLI, gate, step program, server) admits through here before it reports or binds the class.
inline moe_core::grouped_w4a4::Activations admit_mini_moe_w4a4(moe_core::grouped_w4a4::Activations activations,
                                                                std::string_view backend) {
    if (activations != moe_core::grouped_w4a4::Activations::bf16 && backend != "cuda")
        throw std::invalid_argument(std::string("STRIX_INK_MOE_W4A4 selects ") +
                                    moe_core::grouped_w4a4::identity(activations) + ": the " + std::string(backend) +
                                    " backend has no grouped W4A4 tile (CUDA only); select 0 to run the BF16 tile");
    return activations;
}

inline moe_core::grouped_w4a4::Activations admit_mini_moe_w4a4(const profile::Resolution& tier, std::string_view backend) {
    return admit_mini_moe_w4a4(mini_moe_w4a4_class(tier), backend);
}

// Backend of this translation unit's device build, for the entry points compiled once per backend.
#if defined(__HIPCC__)
inline constexpr std::string_view kMiniBuildBackend = "hip";
#else
inline constexpr std::string_view kMiniBuildBackend = "cuda";
#endif

inline profile::Resolution resolve_mini_profile(const MiniWeightOptions& options, bool native_execution) {
    auto tier = profile::resolve(kMiniProfile, options.tier);
    tier.weights.require_honoured();
    if (options.qualify_q8 && (tier.profile != profile::Name::fast || options.disable_q8 || !native_execution))
        throw std::invalid_argument("--qualify-q8 requires --profile fast and native model execution");
    static_cast<void>(mini_moe_w4a4(tier));  // refuse a malformed or unrunnable explicit switch early
    return tier;
}

inline const char* mini_profile_name(profile::Name tier) {
    return tier == profile::Name::standard ? "exact" : profile::name(tier);
}

inline weight::WeightLoadPlan resolve_mini_resident_weight_load(profile::Name tier, const MiniWeightOptions& options,
                                                              const weight::WeightLoadScope& scope) {
    constexpr std::array consumers{weight::kBf16WeightStorage, weight::kQ8WeightStorage};
    const auto plan = resolve_mini_weight_load(tier, scope, consumers, !options.disable_q8, options.qualify_q8);
    if (options.qualify_q8 && plan.storage != weight::kQ8WeightStorage)
        throw std::invalid_argument("native Q8_0 weights were not admitted");
    return plan;
}

} // namespace strix::inkling2
