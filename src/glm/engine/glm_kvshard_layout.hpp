#pragma once
#include "xport/kvshard_traits.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strix::glm::kvshard {
// Cache pools are indivisible ownership blocks. Global positions never change.
inline constexpr std::uint64_t block_tokens = strix::xport::kvshard::GlmDsaTraits::owner_map.block_tokens;
inline constexpr std::uint64_t pair_tokens = 2 * block_tokens;
inline constexpr std::size_t bytes_per_token = strix::xport::kvshard::GlmDsaTraits::latent.bytes +
    strix::xport::kvshard::GlmDsaTraits::index_key.bytes + strix::xport::kvshard::GlmDsaTraits::index_gate.bytes +
    strix::xport::kvshard::GlmDsaTraits::pooled_key.bytes / block_tokens;
// Reject peers/snapshots using the earlier serial partial accumulation identity.
inline constexpr std::uint64_t layout_id = 0x474c4d4b56533033ULL;
inline constexpr std::uint64_t snapshot_magic = 0x474c4d5348524431ULL;
inline constexpr const char* numerical_identity = "glm-dsa-kvshard-rank01-fp32-v3-declaredexp-owned-tree256-fma-rankmerge-FAST";
#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_GLM_KV_HD __host__ __device__
#else
#define STRIX_GLM_KV_HD
#endif
STRIX_GLM_KV_HD constexpr unsigned owner(std::uint64_t token) {
    return static_cast<unsigned>((token / block_tokens) % 2);
}
STRIX_GLM_KV_HD constexpr std::uint64_t local_token(std::uint64_t token) {
    return (token / (2 * block_tokens)) * block_tokens + token % block_tokens;
}
STRIX_GLM_KV_HD constexpr std::uint64_t global_token(std::uint64_t local, unsigned rank) {
    return ((local / block_tokens) * 2 + rank) * block_tokens + local % block_tokens;
}
#undef STRIX_GLM_KV_HD
inline std::uint64_t storage_tokens(std::uint64_t capacity) {
    if (!capacity || capacity % block_tokens != 0 || capacity > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("kvshard: capacity must be a positive u32 multiple of the pool size");
    // Equal allocations on both ranks, including a rounded final block pair.
    return (capacity / pair_tokens + (capacity % pair_tokens != 0 ? 1 : 0)) * block_tokens;
}
constexpr std::uint64_t owned_prefix(std::uint64_t frontier, unsigned rank) {
    const std::uint64_t full = (frontier / pair_tokens) * block_tokens;
    const std::uint64_t tail = frontier % pair_tokens, start = rank * block_tokens;
    return full + (tail <= start ? 0 : (tail - start < block_tokens ? tail - start : block_tokens));
}
inline std::vector<std::uint32_t> merge_score_keys(const std::vector<std::uint32_t>& local,
                                                  const std::vector<std::uint32_t>& peer, unsigned rank) {
    if (rank > 1 || local.size() != peer.size()) throw std::invalid_argument("kvshard: score rank / shape");
    std::vector<std::uint32_t> result(local.size());
    for (std::size_t pool = 0; pool < result.size(); ++pool)
        result[pool] = pool % 2 == rank ? local[pool] : peer[pool];
    return result;
}
// These words are the snapshot compatibility key, not an unsharded numerical identity.
using SnapshotKey = std::array<std::uint64_t, 7>;
inline SnapshotKey snapshot_key(std::uint64_t capacity, unsigned rank, std::uint64_t frontier,
                                std::uint64_t payload_bytes) {
    (void)storage_tokens(capacity);
    if (rank > 1 || frontier > capacity) throw std::invalid_argument("kvshard: snapshot rank / frontier");
    return {snapshot_magic, layout_id, block_tokens, capacity, rank, frontier, payload_bytes};
}
inline void validate_snapshot(const void* bytes, std::size_t size, bool sharded,
                              const SnapshotKey& expected = {}) {
    std::uint64_t magic = 0;
    if (size >= sizeof(magic)) std::memcpy(&magic, bytes, sizeof(magic));
    if (!sharded) {
        if (magic == snapshot_magic) throw std::invalid_argument("restore_state: sharded layout on unsharded engine");
        return;
    }
    SnapshotKey received{};
    if (size < sizeof(received)) throw std::invalid_argument("restore_state: missing sharded compatibility key");
    std::memcpy(received.data(), bytes, sizeof(received));
    if (received != expected || received.back() != size - sizeof(received))
        throw std::invalid_argument("restore_state: sharded layout / rank / capacity / frontier mismatch");
}
} // namespace strix::glm::kvshard
