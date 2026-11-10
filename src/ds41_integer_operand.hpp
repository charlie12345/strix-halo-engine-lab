#pragma once
#include "strix/ds41_formats.hpp"
#include "moe_core/grouped_integer_decode.hpp"
#include "core/tiled_core.hpp"
namespace strix::ds41::integer_core {
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
constexpr int dot_magic = 0x4b400000; // 1.5 * 2^23: bits + r are the float 1.5 * 2^23 + r for |r| < 2^22

template <WeightFormat Format> struct Raw;
template <> struct Raw<WeightFormat::iq2_xxs> { unsigned short d; unsigned long long aux; };
template <> struct Raw<WeightFormat::q2_k> { unsigned q[8]; unsigned short scales; unsigned dm; unsigned shift; };
template <> struct Raw<WeightFormat::mxfp4> { unsigned e; unsigned q[4]; };
template <> struct Raw<WeightFormat::q8_0> { unsigned short d; unsigned q[8]; };

struct Operands { v4i b0, b1; float scale, d, dmin; int m0, m1; unsigned dm_bits, sc_bits; };

// DS41 keeps its integer operand representation and launch policy at the adapter.
struct Ds41PrefillTraits {
    using WeightFormat = ::strix::ds41::WeightFormat;
    using Int4 = v4i;
    using Int8 = v8i;
    using OperandGroup = Operands;
    template <WeightFormat Format> using RawGroup = Raw<Format>;
    static constexpr int magic_bits = dot_magic;
    static constexpr auto iq2_xxs_block_bytes = ::strix::ds41::iq2_xxs_block_bytes;
    static constexpr auto q2_k_block_bytes = ::strix::ds41::q2_k_block_bytes;
    static constexpr auto mxfp4_block_bytes = ::strix::ds41::mxfp4_block_bytes;
    static constexpr auto q8_0_block_bytes = ::strix::ds41::q8_0_block_bytes;
    using Operations = strix::moe_core::GroupedIntegerDecodeOps<Ds41PrefillTraits>;
    template <WeightFormat Format> using WeightDecode = std::conditional_t<
        Format == WeightFormat::q2_k, strix::moe_core::GroupedQ2KWeightDecode<Ds41PrefillTraits>,
        std::conditional_t<Format == WeightFormat::iq2_xxs,
            strix::moe_core::GroupedIq2XxsWeightDecode<Ds41PrefillTraits>,
            std::conditional_t<Format == WeightFormat::q8_0,
                strix::moe_core::GroupedWeightDecode<strix::moe_core::GroupedWeightFormat::q8_0, Operations>,
                Operations>>>;
};

template<WeightFormat F, bool Fast = false, bool Relaxed = false>
using Contract = strix_trellis::IntegerContract<strix_trellis::IntegerOperand<Ds41PrefillTraits, F, Fast>, Relaxed>;
// Header instantiation only: no separate archive or library device link is needed.
template<class C, int BN, class Adapter>
inline void launch(Adapter ad, unsigned blocks, unsigned columns, unsigned width, unsigned outputs, hipStream_t stream) {
    using namespace strix_trellis::wmma_tiled;
    using G = Geometry<16, BN, 16, 32, typename C::Operand>;
    hipLaunchKernelGGL((tiled_core<C, 16, BN, 16, 32, false, 0, false, false, 64, false, Adapter>),
        dim3(blocks, columns), dim3(G::threads), 0, stream,
        nullptr, nullptr, nullptr, int(width), int(outputs), 16, 1,
        nullptr, nullptr, nullptr, 1.0f, 0.0f, 1,
        nullptr, nullptr, nullptr, nullptr, GroupedArgs{}, ad);
}
}
