#pragma once

#include "glm/engine/glm_mem.hpp"
#include "glm/prefill/glm_pf_workspace.hpp"

namespace strix::glm::pf {

// Project the same two quantities make_prefill_path reads after loading:
// hipMemGetInfo free and MemAvailable, after retained weights/state/pools.
// Load-only staging, forward reserves and GTT limits belong to the load guard;
// install_prefill separately admits its prepared experts and replacement scratch.
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
    admit_prefill_bytes(path, workspace, margin, device, host);
}

}  // namespace strix::glm::pf
