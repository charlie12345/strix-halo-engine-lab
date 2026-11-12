// Prefill group policy (shared): how many forward chunks of one prompt a pair-prefill interleaves as one
// layer-major group. A group runs all layers once over its chunks, so per-layer work that is independent of the
// rows (JIT weight conversion of the FAST_I4 slots, expert-table uploads, weight sweeps) is paid once per group
// instead of once per chunk window. Measured (MiniMax M3.1 FAST_I4, 32K prompt, TP2 USB4, ABAB both orders): group 4
// vs 2 = TTFT 55.0 / 54.5 -> 52.2 / 51.7 s (-5.1 %); group 8 vs 2 = 55.5 / 55.2 -> 52.5 / 51.4 s (-6.1 %, i.e. the
// same as 4 within run noise); identical state / rows / result hashes, decode unchanged. Default 4: the gain of 8 is
// not distinguishable from 4 and 8 costs 1.6 GiB more device memory per rank plus a deeper exchange (opt-in
// STRIX_MM_PF_GROUP=8 with STRIX_MM_PFX_DEPTH=8).
//
// The engine clamps the width to the prompt's chunk count at run time (an 8K prompt of 4096-row chunks is a group
// of 2 whatever the default), so a wider default only costs memory: (width - 1) x 4 row buffers of max_rows x
// hidden floats per rank (MiniMax: 0.8 GiB at width 4, 2.4 GiB at width 8), plus one exchange slot per chunk.
#pragma once

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::pair::prefill_group {

inline constexpr unsigned kMin = 2;
inline constexpr unsigned kMax = 8;      // one 32K prompt of 4096-row chunks as a single group
inline constexpr unsigned kDefault = 4;
// Prefix-cache reuse alignment, in chunks. Chunk bits do not depend on the group a chunk is scheduled in (group 2
// and group 4 produce identical state hashes, and a resumed prefill's tail group is already narrower), but reuse
// keeps the alignment it had when the default width was 2 until a chunk-granular reuse is qualified on its own.
inline constexpr unsigned kReuseChunks = 2;

// Width from an environment value (nullptr / empty = default). Throws on an out-of-range explicit value.
inline unsigned width_from_env(const char* value, const char* name) {
    if (value == nullptr || *value == '\0') return kDefault;
    const unsigned long w = std::strtoul(value, nullptr, 10);
    if (w < kMin || w > kMax)
        throw std::invalid_argument(std::string(name) + " must be " + std::to_string(kMin) + ".." + std::to_string(kMax));
    return static_cast<unsigned>(w);
}

// The exchange module needs one TX / RX slot per interleaved chunk (slot reuse of the schedule): the width
// a given exchange depth admits.
inline unsigned width_for_depth(unsigned width, unsigned depth) { return std::max(kMin, std::min(width, depth)); }

}  // namespace strix::pair::prefill_group
