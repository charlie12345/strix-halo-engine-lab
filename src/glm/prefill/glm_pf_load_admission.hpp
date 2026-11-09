#pragma once

#include "glm/engine/glm_mem.hpp"
#include "glm/prefill/glm_pf_workspace.hpp"

namespace strix::glm::pf {

// Project the same two quantities make_prefill_path reads after loading:
// hipMemGetInfo free and MemAvailable, after retained weights/state/pools.
// Load-only staging and GTT limits belong to the load guard;
// Include the shared install/staging bound before admitting any layout; the
// later install guard still rechecks actual availability and prepared handles.
template<class Admission>
inline void admit_prefill_load(const Admission& a, std::uint64_t path,
                              std::uint64_t workspace, std::uint64_t margin,
                              std::uint64_t communication_growth) {
    auto device = a.device_free, host = a.snapshot.meminfo_ok ? a.snapshot.mem_available : 0;
    const auto take = [](std::uint64_t& free, std::uint64_t bytes) { free = free > bytes ? free - bytes : 0; };
    take(device, a.post_load_device_bytes);
    take(device, communication_growth);
    take(host, a.post_load_host_bytes);
    take(host, a.model.physical(communication_growth));
    // Preserve the same first-use headroom when projecting the post-load check.
    // This is a policy margin, not an allocation: make_prefill_path charges the
    // identical byte count against both availability readers.
    take(device, a.prefill_install);
    take(host, a.prefill_install);
    take(device, a.first_forward_reserve);
    take(host, a.first_forward_reserve);
    admit_prefill_bytes(path, workspace, margin, device, host);
}

}  // namespace strix::glm::pf
