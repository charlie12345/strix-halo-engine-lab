#pragma once

#include "strix/model_manifest.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>

namespace strix {
// Bit tests, never floating conversion: signed zero, infinities and NaN payloads
// are preserved only when widening the upper half reproduces all 32 bits.
inline bool router_bf16_exact(std::span<const std::byte> f32) noexcept {
    if (f32.size() % 4U) return false;
    for (std::size_t i = 0; i < f32.size(); i += 4U) {
        std::uint32_t bits;
        std::memcpy(&bits, f32.data() + i, 4U);
        if (bits & 0xffffU) return false;
    }
    return true;
}
inline bool qwen_router_bf16_candidate(const ManifestTensor& t) noexcept {
    if (t.role != TensorRole::moe_router || !t.name.ends_with("ffn_gate_inp.weight") ||
        t.ggml_type != 0U || t.dimensions.size() != 2U) return false;
    const auto cols = t.dimensions[0], rows = t.dimensions[1];
    return cols && rows && cols <= UINT32_MAX && rows <= UINT32_MAX &&
        cols <= UINT64_MAX / rows / 4U && t.byte_size == rows * cols * 4U;
}
inline constexpr std::uint64_t qwen_router_bf16_staging_bytes = 65536U;
struct QwenRouterBf16Reservation {
    std::uint64_t device_bytes = 0, setup_host_bytes = 0;
};
// Conservative pre-device census: price every candidate, including tensors
// later declined by the exactness scan or supplied by an existing dense pack.
// The F32 backing remains charged independently; no reclaim credit for copies.
inline QwenRouterBf16Reservation qwen_router_bf16_reservation(const ModelManifest& manifest, bool enabled) {
    QwenRouterBf16Reservation result;
    if (!enabled) return result;
    for (const auto& t : manifest.tensors) if (qwen_router_bf16_candidate(t)) {
        const auto bytes = t.byte_size / 2U;
        if (bytes > UINT64_MAX - result.device_bytes) throw std::overflow_error("router BF16 reservation overflow");
        result.device_bytes += bytes;
    }
    if (result.device_bytes) result.setup_host_bytes = qwen_router_bf16_staging_bytes;
    return result;
}
} // namespace strix
