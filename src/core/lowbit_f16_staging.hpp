#pragma once
// Low-bit resident rows (strix/weight/host_lowbit.hpp q4_g64 / q6_g64) as FP16 matrix operands of the f16 tile GEMM
// family. Two consumers share one decoder:
//   - LowBitF16Staging: the staging trait of the 128-row projection tile (f16_tile_gemm.hpp, prefill). Same contract as
//     the Q8_0 staging: w16 = fp16_rne(float(code) * float(d)) (one packed FP16 multiply of the exact integer code).
//   - the small-M schedule (f16_tile_gemm_small.hpp): integer codes are the operand as is (exact FP16 integers) and the
//     FP16 group scale d is applied once per K64 group to the FP32 accumulator.
// Decode of 16 consecutive elements (two 4-bit plane words, one 2-bit plane word for q6): bytes of biased codes
// (code + 8 / code + 32) are built with v_perm_b32, then v_perm_b32 puts 0x64 above each byte, i.e. the FP16 integer
// 1024 + biased code; the packed subtraction of 1032 / 1056 is exact.

#include "strix/weight/host_lowbit.hpp"

#include <cstdint>

#if defined(__HIPCC__)
#include "hip/weight/arch/gfx1151_sources.hpp"
#endif

namespace strix::core::lowbit_f16 {

// FP16 integer bias of a biased code: 1024 + 2^(bits - 1).
template <class Storage> inline constexpr float kBias = Storage::high_bytes == 0U ? 1032.0F : 1056.0F;

#if defined(__HIPCC__)
using Half2 = weight::hip::gfx1151::Half2;
using WordsG = const std::uint32_t __attribute__((address_space(1), aligned(4)));
typedef std::uint32_t Words4 __attribute__((ext_vector_type(4), aligned(4)));
using Words4G = const Words4 __attribute__((address_space(1)));

// Bytes of four biased codes, element e of the result in byte e, for elements 4 q .. 4 q + 3 (q = 0..3) of the 16-element
// run whose 4-bit plane words are lo0 (elements 0..7) and lo1 (8..15) and whose 2-bit plane word is hi (q6 only).
template <class Storage>
__device__ __forceinline__ void biased_bytes(std::uint32_t lo0, std::uint32_t lo1, std::uint32_t hi, std::uint32_t (&z)[4]) {
    constexpr bool kQ6 = Storage::high_bytes != 0U;
#pragma unroll
    for (unsigned p = 0; p < 2U; ++p) {
        // q4: XOR 8 per nibble turns the signed code into code + 8.
        const std::uint32_t word = (p == 0U ? lo0 : lo1) ^ (kQ6 ? 0U : 0x88888888U);
        const std::uint32_t even = word & 0x0F0F0F0FU, odd = (word >> 4U) & 0x0F0F0F0FU;
        z[2U * p] = __builtin_amdgcn_perm(odd, even, 0x05010400U);       // elements 8p + 0..3
        z[2U * p + 1U] = __builtin_amdgcn_perm(odd, even, 0x07030602U);  // elements 8p + 4..7
    }
    if constexpr (kQ6) {
#pragma unroll
        for (unsigned q = 0; q < 4U; ++q) {
            // 2-bit fields of elements 4q..4q+3 (hi bits 8q..8q+7) to bits 0-1 of bytes 0..3, then above the nibble;
            // XOR 32 turns the signed 6-bit code into code + 32.
            const std::uint32_t t = (hi >> (8U * q)) & 0xFFU;
            const std::uint32_t a = (t | (t << 12U)) & 0x000F000FU;
            const std::uint32_t b = (a | (a << 6U)) & 0x03030303U;
            z[q] = (z[q] | (b << 4U)) ^ 0x20202020U;
        }
    }
}

// Eight FP16 pairs (elements 2 i, 2 i + 1 in word i) of the exact integer codes of a 16-element run.
template <class Storage>
__device__ __forceinline__ void integer_pairs(std::uint32_t lo0, std::uint32_t lo1, std::uint32_t hi, std::uint32_t (&words)[8]) {
    std::uint32_t z[4];
    biased_bytes<Storage>(lo0, lo1, hi, z);
    const auto bias = static_cast<_Float16>(kBias<Storage>);
#pragma unroll
    for (unsigned q = 0; q < 4U; ++q) {
        const auto l = __builtin_bit_cast(Half2, __builtin_amdgcn_perm(0x64646464U, z[q], 0x04010400U)) - Half2{bias, bias};
        const auto h = __builtin_bit_cast(Half2, __builtin_amdgcn_perm(0x64646464U, z[q], 0x04030402U)) - Half2{bias, bias};
        words[2U * q] = __builtin_bit_cast(std::uint32_t, l);
        words[2U * q + 1U] = __builtin_bit_cast(std::uint32_t, h);
    }
}

// Staging trait of the 128-row projection tile (interface of q8_f16_staging::Q8F16Staging): a thread stages the 16
// elements [k0 + 16 half, + 16) of one weight row (a K32 stage is half of one 64-group, so one scale per thread).
template <class Storage>
struct LowBitF16Staging {
    struct Raw {
        std::uint32_t lo0, lo1, hi, scale;
    };

    __device__ __forceinline__ static Raw load(const std::uint8_t* row, unsigned k0, unsigned half) {
        if (!row) return {};
        const unsigned k = k0 + 16U * half, j = k % Storage::elements;
        const auto* block = row + std::size_t{k / Storage::elements} * Storage::bytes;
        Raw raw{};
        raw.lo0 = *(WordsG*)(block + j / 2U);
        raw.lo1 = *(WordsG*)(block + j / 2U + 4U);
        if constexpr (Storage::high_bytes != 0U) raw.hi = *(WordsG*)(block + Storage::low_bytes + j / 4U);
        // The two scales of a superblock are one aligned word: pick the group's half.
        raw.scale = *(WordsG*)(block + Storage::scale_offset) >> (16U * (j / Storage::group));
        return raw;
    }

    __device__ __forceinline__ static void publish(const Raw& raw, unsigned, std::uint32_t* dst) {
        using U4 = weight::hip::gfx1151::U4;
        const std::uint32_t bits = raw.scale & 0xFFFFU;
        const auto scale = __builtin_bit_cast(Half2, bits | (bits << 16U));
        std::uint32_t words[8];
        integer_pairs<Storage>(raw.lo0, raw.lo1, raw.hi, words);
#pragma unroll
        for (unsigned i = 0; i < 8U; ++i) words[i] = __builtin_bit_cast(std::uint32_t, __builtin_bit_cast(Half2, words[i]) * scale);
        *reinterpret_cast<U4*>(dst) = U4{words[0], words[1], words[2], words[3]};
        *reinterpret_cast<U4*>(dst + 4U) = U4{words[4], words[5], words[6], words[7]};
    }
};
#endif

}  // namespace strix::core::lowbit_f16
