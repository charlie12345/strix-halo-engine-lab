#pragma once
#include "glm/common/glm_math.hpp"
#include "glm/engine/glm_kvshard_layout.hpp"

namespace strix::glm::fast {
// Only partial instantiations carry this context; replicated kernel arguments
// and workspace layouts are unchanged. Query pointers are independent of policy.
struct DsaPartialContext {
    unsigned rank;
    float* output;  // [rows][heads][m, l, a[512]]
    unsigned* flag;
};

template <bool Partial> struct DsaAttentionPolicy;

template <> struct DsaAttentionPolicy<false> {
    static constexpr bool partial = false;
    STRIX_GLM_HD static bool owns(std::uint32_t) { return true; }
    STRIX_GLM_HD static std::uint32_t address(std::uint32_t token) { return token; }
    STRIX_GLM_HD static float weight(float score, float maximum) { return declared_exp(score - maximum); }
    STRIX_GLM_HD static float probability(float weight, float sum) { return weight / sum; }
    STRIX_GLM_HD static std::size_t output_index(std::size_t head, unsigned dim) {
        return head * kKvLora + dim;
    }
};

template <> struct DsaAttentionPolicy<true> {
    static constexpr bool partial = true;
    STRIX_GLM_HD static bool owns(std::uint32_t token, const DsaPartialContext& context) {
        return kvshard::owner(token) == context.rank;
    }
    // Unowned lanes may load a safe row but never contribute to scores or PV.
    STRIX_GLM_HD static std::uint32_t address(std::uint32_t token, const DsaPartialContext& context) {
        return owns(token, context) ? static_cast<std::uint32_t>(kvshard::local_token(token)) : 0U;
    }
    STRIX_GLM_HD static float weight(float score, float maximum) {
        // Empty ranks have m=-inf: do not evaluate -inf - -inf.
        return score == -__builtin_huge_valf() ? 0.0F : declared_exp(score - maximum);
    }
    STRIX_GLM_HD static float probability(float weight, float) { return weight; }
    STRIX_GLM_HD static std::size_t output_index(std::size_t head, unsigned dim) {
        return head * (kKvLora + 2) + 2 + dim;
    }
};
} // namespace strix::glm::fast
