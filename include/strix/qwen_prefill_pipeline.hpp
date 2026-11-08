#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix::qwen_pp {
// Wire v2 rows: BF16 HC streams, BF16 attention input (HC output), four FP32 coefficients.
// A folded next-layer HC is already executed: its output must travel with its skip marker.
struct Seam {
    static constexpr std::uint64_t magic = 0x5050325345414d02ULL;
    static constexpr std::size_t hc_bytes = 4U * 2560U * sizeof(std::uint16_t);
    static constexpr std::size_t attention_bytes = 2560U * sizeof(std::uint16_t);
    static constexpr std::size_t residual_bytes = attention_bytes; // legacy size alias, not hidden(1)
    static constexpr std::size_t coefficient_bytes = 4U * sizeof(float);
    static constexpr std::size_t row_bytes = hc_bytes + residual_bytes + coefficient_bytes;
    std::vector<std::byte> bytes;
    explicit Seam(std::uint32_t rows) : bytes(static_cast<std::size_t>(rows) * row_bytes) {}
    std::byte* row(std::uint32_t index) { return bytes.data() + static_cast<std::size_t>(index) * row_bytes; }
};
static_assert(Seam::row_bytes == 25616U);
static_assert(sizeof(float) == 4U);

struct LayerRange {
    std::uint32_t begin = 0;
    std::uint32_t end = 48;
    bool enabled = false;
};
inline LayerRange parse_range(const char* value, std::uint32_t layers) {
    if (!value) return {0, layers, false};
    const std::string_view text(value);
    const auto colon = text.find(':');
    LayerRange range{0, 0, true};
    if (colon == std::string_view::npos || colon == 0 || colon + 1U == text.size())
        throw std::invalid_argument("STRIX_QWEN_PF_LAYER_RANGE requires begin:end");
    const auto first = std::from_chars(text.data(), text.data() + colon, range.begin);
    const auto last = std::from_chars(text.data() + colon + 1U, text.data() + text.size(), range.end);
    if (first.ec != std::errc{} || first.ptr != text.data() + colon || last.ec != std::errc{} ||
        last.ptr != text.data() + text.size() || range.begin >= range.end || range.end > layers)
        throw std::invalid_argument("STRIX_QWEN_PF_LAYER_RANGE is outside the model");
    return range;
}
} // namespace strix::qwen_pp
