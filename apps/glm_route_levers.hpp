#pragma once
// GLM levers of the shared route receipt (strix/route_receipt.hpp). The rank CLI and the server record the same keys
// through this one function, so their requested and effective values can be compared entry point against entry point.

#include <cstdint>
#include <string>
#include <string_view>

#include "strix/env_flags.hpp"
#include "strix/route_receipt.hpp"

namespace strix::glm {

// Records the levers that decide the prefill arithmetic and returns the numerical-route fingerprint.
// entry: "cli" or "http" (informative, outside the fingerprint, like the chunk rows).
inline std::uint64_t record_route_levers(std::string_view entry, unsigned rank, std::string_view prefill_requested,
                                         std::string_view prefill_effective, std::string_view identity, unsigned max_rows) {
    namespace rr = strix::route;
    rr::set_engine("glm");
    rr::set_rank(static_cast<int>(rank));
    const auto asked = [](const char* name) {
        const char* v = strix::envcfg::text(name);
        return std::string(v != nullptr && *v != 0 ? v : "default");
    };
    const std::string wmma = asked("STRIX_GLM_PF_WMMA"), x = asked("STRIX_GLM_PF_EXACT_X"), kda = asked("STRIX_GLM_PF_KDA_WMMA");
    rr::lever("glm.entry", entry, entry, rr::Reason::selected, false);
    rr::lever("glm.prefill", prefill_requested, prefill_effective);
    rr::lever("glm.identity", identity, identity);
    rr::lever("glm.pf_wmma", wmma, wmma == "1" ? "f16_wmma" : "exact", wmma == "1" ? rr::Reason::selected : rr::Reason::disabled);
    rr::lever("glm.pf_kda_wmma", kda, kda);
    rr::lever("glm.pf_exact_x", x, x + ":min_rows=" + asked("STRIX_GLM_PF_EXACT_X_MIN_ROWS"), rr::Reason::selected, true,
              "auto takes the X3 route (and the WMMA identity when on) only for calls of at least min_rows rows");
    rr::lever("glm.max_rows", std::to_string(max_rows), std::to_string(max_rows), rr::Reason::selected, false);
    return rr::fingerprint();
}

}  // namespace strix::glm
