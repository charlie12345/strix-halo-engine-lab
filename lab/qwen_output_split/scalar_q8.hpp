#pragma once
#include "output_split.hpp"
#include <bit>
namespace strix::qwen_split {
inline float bf16(std::uint16_t bits) { return std::bit_cast<float>(std::uint32_t{bits} << 16U); }
inline float fp16(std::uint16_t h) {
    const auto sign = std::uint32_t{h & 0x8000U} << 16U;
    auto exponent = std::uint32_t{(h >> 10U) & 31U}, fraction = std::uint32_t{h & 1023U};
    if (exponent == 31U) {
        return std::bit_cast<float>(sign | 0x7f800000U | (fraction << 13U));
    }
    if (!exponent) {
        if (!fraction) {
            return std::bit_cast<float>(sign);
        }
        exponent = 113U;
        while (!(fraction & 1024U)) {
            fraction <<= 1U;
            --exponent;
        }
        fraction &= 1023U;
    } else {
        exponent += 112U;
    }
    return std::bit_cast<float>(sign | (exponent << 23U) | (fraction << 13U));
}
// Scalar translation of src/hip_lm_head_launcher.hip: independent host
// execution of the project-owned recipe's leaf and wave32/local
// tree. Mode is declared before execution; never try both to select a passing
// result. The native target compiles this host code with contraction OFF.
inline float scalar_q8(std::span<const std::uint16_t> input, std::span<const std::byte> weights,
                       std::uint64_t leaf) {
    require(!input.empty() && input.size() <= 10240 && input.size() % 32 == 0 &&
                weights.size() == input.size() / 32 * 34 && (leaf == 1 || leaf == 2),
            "scalar packed head shape/leaf invalid");
    std::array<std::array<float, 8>, 32> partial{};
    for (std::size_t chunk = 0; chunk < input.size(); chunk += 256) {
        for (std::size_t lane = 0; lane < 32; ++lane) {
            const auto base = chunk + 8 * lane;
            if (base >= input.size()) {
                continue;
            }
            const auto block = base / 32 * 34;
            const auto scale_bits =
                static_cast<std::uint16_t>(std::to_integer<unsigned>(weights[block]) |
                                           (std::to_integer<unsigned>(weights[block + 1]) << 8U));
            const float scale = fp16(scale_bits);
            require(std::isfinite(scale), "scalar nonfinite packed scale");
            for (std::size_t k = 0; k < 8; ++k) {
                const float a = bf16(input[base + k]);
                require(std::isfinite(a), "scalar nonfinite hidden input");
                const auto raw = static_cast<std::uint8_t>(
                    std::to_integer<unsigned>(weights[block + 2 + base % 32 + k]));
                const float b = scale * static_cast<float>(std::bit_cast<std::int8_t>(raw));
                if (leaf == 1) {
                    partial[lane][k] = std::fma(a, b, partial[lane][k]);
                } else {
                    const float product = a * b;
                    partial[lane][k] = partial[lane][k] + product;
                }
            }
        }
    }
    for (std::size_t distance = 16; distance; distance /= 2) {
        for (std::size_t lane = 0; lane < distance; ++lane) {
            for (std::size_t k = 0; k < 8; ++k) {
                partial[lane][k] += partial[lane + distance][k];
            }
        }
    }
    auto &v = partial[0];
    v[0] += v[4];
    v[1] += v[5];
    v[2] += v[6];
    v[3] += v[7];
    v[0] += v[2];
    v[1] += v[3];
    v[0] += v[1];
    return v[0];
}
} // namespace strix::qwen_split
