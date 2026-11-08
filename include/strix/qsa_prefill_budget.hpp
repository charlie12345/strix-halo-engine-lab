#pragma once
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace strix {
// Budget covers ranked complete blocks; the incumbent partial-block local tail is additional.
// For fixed q/k/v in real arithmetic, removing softmax mass p changes one head by at most
// 2*p*max(norm(v)). This is not a rounding bound or a model-fidelity qualification.
inline std::uint32_t parse_qsa_prefill_budget(std::string_view text) {
    if (text.empty()) return 0U;
    std::uint32_t value = 0U;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
        throw std::invalid_argument("STRIX_QPF2K_QSA_BUDGET requires a nonnegative decimal uint32");
    return value;
}
inline std::uint32_t qsa_prefill_block_topk(std::uint32_t unrestricted, std::uint32_t ratio, std::uint32_t budget) {
    if (!ratio || (budget && budget % ratio))
        throw std::invalid_argument("STRIX_QPF2K_QSA_BUDGET must be a multiple of the indexer compression ratio");
    return budget ? std::min(unrestricted, budget / ratio) : unrestricted;
}
} // namespace strix
