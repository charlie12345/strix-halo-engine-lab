#pragma once

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace strix::glm::pf {

// One score window plus the retained top-512 pools; independent of slot capacity.
inline constexpr unsigned kIndexerPoolWindow = 4096;
inline constexpr unsigned kIndexerRetainedPools = 512;
inline constexpr unsigned kIndexerKeyStride = kIndexerPoolWindow + kIndexerRetainedPools;
inline constexpr unsigned kPrefillAllocationBound = 64;
// Target, not a ceiling: the byte-admitted planner can pack more rows.
inline constexpr unsigned kKvShardPrefillRows = 128;
// Startup reserves the minimum only. Extra temporary bytes are opportunistic,
// rank-min admitted after pool scratch is retired; never charge them to capacity.
inline constexpr std::uint64_t kKvShardPrefillTemporaryBytes = std::uint64_t{64} << 20;
inline constexpr std::uint64_t kKvShardPrefillMaximumBytes = std::uint64_t{2} << 30;
constexpr std::uint64_t indexer_workspace_bytes(unsigned rows) {
    return std::uint64_t{rows} * kIndexerKeyStride * sizeof(std::uint32_t);
}

// Unmodelled runtime/host growth only. Device owners, allocation rounding, shared
// expert preparation, staging and replacement workspace are charged separately.
// 256 MiB HIP/runtime + 256 MiB host metadata + 512 MiB availability drift.
inline constexpr std::uint64_t kPrefillSafetyBytes = std::uint64_t{1} << 30;

inline void admit_prefill_bytes(std::uint64_t path, std::uint64_t workspace, std::uint64_t margin,
                               std::uint64_t device_free, std::uint64_t host_available) {
    const auto fits = [&](std::uint64_t free) {
        return path <= free && workspace <= free - path && margin <= free - path - workspace;
    };
    if (fits(device_free) && fits(host_available)) return;
    const auto gib = [](std::uint64_t v) { return static_cast<double>(v) / static_cast<double>(std::uint64_t{1} << 30); };
    std::ostringstream os;
    os << "glm pf path: refusing to allocate " << (gib(path) + gib(workspace)) << " GiB (path " << gib(path)
       << " + workspace " << gib(workspace) << ") + margin " << gib(margin) << " GiB: device free "
       << gib(device_free) << " GiB, host MemAvailable " << gib(host_available) << " GiB (lower max_rows / capacity)";
    throw std::runtime_error(os.str());
}

}  // namespace strix::glm::pf
