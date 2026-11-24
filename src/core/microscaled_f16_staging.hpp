#pragma once
// Microscaled weight rows (strix/weight/formats/microscaled.hpp: NVFP4 = E2M1 codes with one E4M3 scale per 16 columns
// and an FP32 outer scale; MXFP8 = E4M3 codes with one E8M0 scale per 32 columns) as operands of the small-M schedule of
// the f16 tile family (core/f16_tile_gemm_small.hpp, Operand trait). Bit-changing (FAST): the codes are exact FP16 WMMA
// operands, the group multiplier is applied once per group to the FP32 accumulator, so only the summation order differs
// from a decoded-weight dot product.
//   NVFP4: operand = E2M1 value (exact in FP16), multiplier = E4M3 scale (exact), the outer scale is the binding's (store).
//   MXFP8: operand = E4M3 value x 2^-8 (exact in FP16, small values as FP16 subnormals). Compensate in the scale
//   when it fits in FP32, otherwise compensate in the group partial and use the unshifted E8M0 scale.
// Malformed scales follow PackedOperand::decode_scale: an NVFP4 scale byte above 0x7e and the E8M0 byte 0xff give NaN.
// MXFP8 NaN codes (0x7f / 0xff) are not reproduced (finite operand); the load-time row check rejects them.
#include "strix/weight/formats/microscaled.hpp"

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#define STRIX_SMALL_SCALE_HD __host__ __device__
#else
#define STRIX_SMALL_SCALE_HD
#endif

namespace strix::core::microscaled_f16 {

struct Mxfp8GroupScale {
    float multiplier;
    float partial_multiplier;
};

STRIX_SMALL_SCALE_HD inline Mxfp8GroupScale mxfp8_group_scale(std::uint8_t byte) {
    namespace sf = strix::weight::scalar_float;
    if (byte <= 246U) return {sf::u2f((std::uint32_t{byte} + 8U) << 23U), 1.0F};
    return {sf::e8m0(byte), 256.0F};
}

}  // namespace strix::core::microscaled_f16
#undef STRIX_SMALL_SCALE_HD

#if defined(__HIPCC__)
#include "core/lowbit_f16_staging.hpp"

namespace strix::core::microscaled_f16 {

namespace packed = strix::weight::microscaled;
using lowbit_f16::Words4;
using lowbit_f16::Words4G;
using lowbit_f16::WordsG;
using Words2G = const std::uint16_t __attribute__((address_space(1), aligned(2)));

// One weight row of a microscaled matrix: its code bytes and its scale bytes.
struct Row {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
};

// FP16 pairs of the 8 E2M1 codes of one word (element 2 j in the low half of word j, 2 j + 1 in the high half).
// v_perm_b32 reads the FP16 high byte of the magnitude from an 8-entry table (the low bytes are zero); the sign bit of
// the nibble moves to bit 15 / 31.
__device__ __forceinline__ void e2m1_pairs(std::uint32_t w, std::uint32_t* out) {
    constexpr std::uint32_t kTableLo = 0x3E3C3800U, kTableHi = 0x46444240U;  // 0, 0.5, 1, 1.5 | 2, 3, 4, 6
    const std::uint32_t even = w & 0x0F0F0F0FU, odd = (w >> 4U) & 0x0F0F0F0FU;
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) {
        // [0, even nibble j, 0, odd nibble j]
        const std::uint32_t x = __builtin_amdgcn_perm(odd, even, ((4U + j) << 24U) | 0x000C0000U | (j << 8U) | 0x0CU);
        const std::uint32_t magnitude = __builtin_amdgcn_perm(kTableHi, kTableLo, (x & 0x07000700U) | 0x000C000CU);
        out[j] = magnitude | ((x & 0x08000800U) << 4U);
    }
}

// FP16 pairs of the 4 E4M3 codes of one word, each scaled by 2^-8: the 7 magnitude bits shift into bits 7..13 (the E4M3
// exponent becomes the FP16 exponent field, so subnormals stay exact), the sign into bit 15.
__device__ __forceinline__ void e4m3_pairs(std::uint32_t w, std::uint32_t* out) {
#pragma unroll
    for (unsigned j = 0; j < 2U; ++j) {
        // [0, byte 2 j, 0, byte 2 j + 1]
        const std::uint32_t x = __builtin_amdgcn_perm(0U, w, ((2U * j + 1U) << 24U) | 0x000C0000U | ((2U * j) << 8U) | 0x0CU);
        out[j] = (x & 0x80008000U) | ((x >> 1U) & 0x3F803F80U);
    }
}

template <packed::MatrixFormat Format>
struct Operand {
    static constexpr bool kNvfp4 = Format == packed::MatrixFormat::nvfp4;
    static constexpr unsigned kGroup = kNvfp4 ? 16U : 32U, kUnits = kGroup / 16U, kSuperK = 128U;
    static_assert(kGroup == packed::PackedOperand<Format>::kBlockColumns);
    using Row = microscaled_f16::Row;
    // Code and scale bytes of 64 Groups elements: NVFP4 32 Groups code bytes + 4 Groups scale bytes, MXFP8 64 Groups code
    // bytes + 2 Groups scale bytes.
    static constexpr unsigned code_bytes(unsigned groups) { return (kNvfp4 ? 32U : 64U) * groups; }
    template <unsigned Groups>
    struct Words {
        Words4 c[code_bytes(Groups) / 16U];
        std::uint32_t scale[kNvfp4 ? Groups : 1U];
    };

    template <unsigned Groups>
    __device__ __forceinline__ static Words<Groups> load(Row row, unsigned index) {
        Words<Groups> w{};
        const auto* codes = row.codes + std::size_t{index} * code_bytes(Groups);
#pragma unroll
        for (unsigned i = 0; i < code_bytes(Groups) / 16U; ++i) w.c[i] = *(Words4G*)(codes + 16U * i);
        if constexpr (kNvfp4) {
#pragma unroll
            for (unsigned i = 0; i < Groups; ++i) w.scale[i] = *(WordsG*)(row.scales + 4U * (Groups * index + i));
        } else if constexpr (Groups == 2U) {
            w.scale[0] = *(WordsG*)(row.scales + 4U * index);
        } else {
            w.scale[0] = *(Words2G*)(row.scales + 2U * index);
        }
        return w;
    }

    // Unit u of group gi = elements kGroup gi + 16 u .. + 16 of the block.
    template <unsigned Groups>
    __device__ __forceinline__ static void unit(const Words<Groups>& w, unsigned gi, unsigned u, std::uint32_t (&words)[8]) {
        if constexpr (kNvfp4) {
            const unsigned word = 2U * gi;  // 8 code bytes per unit
            e2m1_pairs(w.c[word / 4U][word % 4U], words);
            e2m1_pairs(w.c[word / 4U][word % 4U + 1U], words + 4);
        } else {
            const auto c = w.c[2U * gi + u];  // 16 code bytes per unit
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) e4m3_pairs(c[i], words + 2U * i);
        }
    }

    template <unsigned Groups>
    __device__ __forceinline__ static auto scale(const Words<Groups>& w, unsigned gi) {
        const auto byte = static_cast<std::uint8_t>(w.scale[gi / 4U] >> (8U * (gi % 4U)));
        if constexpr (kNvfp4) {
            // E4M3 scale x 2^-8 as FP16 bits (the shift of e4m3_pairs), widened exactly and scaled back by 2^8.
            const float value = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(byte << 7U)));
            return byte <= 0x7EU ? value * 256.0F : __builtin_bit_cast(float, 0x7FC00000U);
        } else {
            return mxfp8_group_scale(byte);
        }
    }

    // Row `row` (logical) of a matrix with `columns` K elements.
    __device__ __forceinline__ static Row row(const packed::PackedMatrix& m, unsigned logical_row) {
        const std::size_t r = packed::physical_row(m.row_order, m.rows, logical_row);
        return {m.codes + r * (kNvfp4 ? m.columns / 2U : m.columns), m.block_scales + r * (m.columns / kGroup)};
    }
};

}  // namespace strix::core::microscaled_f16
#endif
