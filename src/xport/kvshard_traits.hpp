#pragma once
// Schema only. No engine/device headers, allocation, or production routing.
#include <cstddef>
#include <cstdint>

namespace strix::xport::kvshard {
enum class CacheFormat { bf16, fp32, e2m1_e4m3_16, e2m1_e8m0_32, e4m3_e8m0_32 };
enum class OwnerRule { floor_token_over_block_mod_two };
struct OwnerMap {
    std::uint64_t block_tokens;
    std::uint64_t group_tokens;
    static constexpr unsigned world = 2;
    static constexpr OwnerRule rule = OwnerRule::floor_token_over_block_mod_two;
};
struct CacheEntry { CacheFormat format; std::size_t elements, bytes; };
// Consumers reuse these source-layer IDs, not independent copies of owned history.
struct SourceMap { std::uint32_t kv_layer, selection_layer; };
struct PartialDims {
    static constexpr unsigned local_heads = 32, accumulator = 512;
    static constexpr unsigned fp32_per_head = accumulator + 2; // m,l,a[512]
    static constexpr std::size_t bytes = local_heads * fp32_per_head * sizeof(float);
};
struct GlmDsaTraits {
    static constexpr OwnerMap owner_map{4, 4};
    static constexpr CacheEntry latent{CacheFormat::bf16, 512, 1024};
    static constexpr CacheEntry index_key{CacheFormat::bf16, 128, 256};
    static constexpr CacheEntry index_gate{CacheFormat::bf16, 128, 256};
    static constexpr CacheEntry pooled_key{CacheFormat::fp32, 128, 512}; // per pool
    using partial = PartialDims; // latent accumulator, before value projection
    static constexpr bool includes_kda = false, has_sink = false;
};
// Released DS41 compressed groups: ratio 1 or 2. Window/carries stay replicated.
template<unsigned Ratio> struct Ds41Traits {
    static_assert(Ratio == 1 || Ratio == 2, "unsupported compressed group");
    static constexpr OwnerMap owner_map{Ratio, Ratio};
    static constexpr CacheEntry compressed_kv{CacheFormat::e2m1_e4m3_16, 512, 288};
    static constexpr CacheEntry index_key{CacheFormat::e2m1_e8m0_32, 128, 68};
    static constexpr CacheEntry window{CacheFormat::e4m3_e8m0_32, 512, 528};
    using partial = PartialDims;
    static constexpr bool has_sink = true, replicate_window_and_carry = true;
};
inline constexpr char identity[] = "kvshard-fast-fp32-r0-r1-v1";
inline constexpr bool global_selection_before_partition = true;
inline constexpr bool selected_ids_ascending = true;
inline constexpr unsigned sink_applications = 1;
} // namespace strix::xport::kvshard
