#pragma once
// FAST identity glm-pf-wmma-x, host side (no HIP): calls that actually ran on the WMMA linears, so a run's identity
// line reports executed arithmetic, not only the requested switch (STRIX_GLM_PF_WMMA=1 still leaves the dense calls
// below the X3 threshold and the MoE calls below V11 on the exact cores).
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strix::glm::pf::wmma_x {
// STRIX_GLM_PF_WMMA=1 selects this identity for the exact prefill X3 calls (dense and MoE); unset or 0 = exact.
// Part of the TP2 startup agreement (rank_defaults::kPfWmmaConfig): both ranks must resolve the same value.
inline bool enabled() {
    const char* p = std::getenv("STRIX_GLM_PF_WMMA");
    return p && std::strcmp(p, "1") == 0;
}
inline std::atomic<std::uint64_t> dense_calls{0}, moe_calls{0};
}  // namespace strix::glm::pf::wmma_x
