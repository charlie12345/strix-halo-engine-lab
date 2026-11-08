#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace strix {

// ISA support inferred from an exact architecture family, not a claim that a
// compiler, binary variant or fast path has been qualified on this device.
struct DeviceIsaCapabilities {
    bool rdna35_wmma = false;
    bool wmma_bf16_w32 = false;
    bool dpp_row = false;
    bool sudot4 = false;
};

DeviceIsaCapabilities device_isa_capabilities(std::string_view arch);

struct DeviceProfile {
    std::string arch_name;
    int device_index = 0;
    int cu_count = 0;
    int wave_size = 0;
    std::uint64_t lds_bytes = 0;
    std::uint64_t total_memory_bytes = 0;
    std::uint64_t free_memory_bytes = 0;
    DeviceIsaCapabilities isa;
};

DeviceProfile probe_device_profile(); // current HIP device; no allocations/benchmark
std::string format_device_profile(const DeviceProfile& profile); // one JSON line
// Profile-only inspection exits before model/peer setup. Effective-config keeps
// its host-only contract: explicit combination is refused, env-only is ignored.
bool print_device_profile_if_requested(int argc, const char* const* argv);

} // namespace strix
