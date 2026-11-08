#pragma once
#include "glm_kvshard_layout.hpp"
#include "xport/kvshard_traits.hpp"
#include <algorithm>
#include <limits>

namespace strix::glm::kvshard {
// Decode frames use the shared TP Channel; prefill keeps its bulk protocol.
struct DecodeHeader {
    std::uint64_t layout, epoch, layer, position, phase, pools, words;
};
static_assert(sizeof(DecodeHeader) == 56);
inline constexpr std::size_t query_words = xport::kvshard::PartialDims::local_heads * xport::kvshard::PartialDims::accumulator;
inline constexpr std::size_t header_words = sizeof(DecodeHeader) / sizeof(std::uint32_t);
inline constexpr std::size_t query_u32 = query_words * sizeof(float) / sizeof(std::uint32_t);
inline constexpr std::size_t partial_u32 = xport::kvshard::PartialDims::bytes / sizeof(std::uint32_t);
inline constexpr std::size_t prepare_bytes(std::size_t pools) {
    return sizeof(DecodeHeader) + query_words * sizeof(float) + pools * sizeof(std::uint32_t);
}
inline constexpr std::size_t partial_bytes = sizeof(DecodeHeader) + xport::kvshard::PartialDims::bytes;
inline std::size_t decode_capacity(std::uint64_t tokens) {
    if (tokens / block_tokens > (std::numeric_limits<std::size_t>::max() - prepare_bytes(0)) / sizeof(std::uint32_t))
        throw std::overflow_error("kvshard: frame capacity overflow");
    return std::max(prepare_bytes(static_cast<std::size_t>(tokens / block_tokens)), partial_bytes);
}
inline DecodeHeader decode_header(std::uint64_t epoch, unsigned layer, std::uint64_t position,
                                 bool partial, std::size_t pools) {
    return {layout_id, epoch, layer, position, partial ? 3U : 2U, pools, query_words};
}
// Either rank's replicated-plan refusal requires the same owned-pool plan on both.
inline constexpr bool auto_shard(bool local_replicated_fits, bool peer_replicated_fits) {
    return !local_replicated_fits || !peer_replicated_fits;
}
enum class PrefillLayout { replicated, sharded, refuse };
inline constexpr PrefillLayout prefill_layout(bool local_replicated_fits, bool peer_replicated_fits,
                                              bool local_sharded_fits, bool peer_sharded_fits) {
    if (!auto_shard(local_replicated_fits, peer_replicated_fits)) return PrefillLayout::replicated;
    return local_sharded_fits && peer_sharded_fits ? PrefillLayout::sharded : PrefillLayout::refuse;
}
} // namespace strix::glm::kvshard
