#pragma once
#include "integer_math.hpp"
#include <type_traits>
namespace strix_trellis {
// Decode original GGUF blocks; no resident weight pack or converter.
template<class Traits, typename Traits::WeightFormat Format, bool FastI4 = false>
struct IntegerOperand {
    using Ops = typename Traits::Operations;
    using Int4 = typename Traits::Int4;
    using Int8 = typename Traits::Int8;
    static constexpr auto format = Format;
    static constexpr bool fast_i4 = FastI4;
    static constexpr bool affine = Format == Traits::WeightFormat::q2_k;
    static constexpr int element_bits = FastI4 ? 4 : 8, stage_tiles = 8;
    static_assert(!FastI4 || affine || Format == Traits::WeightFormat::iq2_xxs);
    static __device__ __forceinline__ auto load_raw(const unsigned char* row, unsigned group) {
        return Ops::template load_raw<Format>(row, group);
    }
    static __device__ __forceinline__ auto decode(const unsigned char* row, unsigned group,
                                                 const unsigned long long* grid) {
        return Ops::decode(load_raw(row, group), grid);
    }
};
template<class T> struct is_integer_operand : std::false_type {};
template<class T, typename T::WeightFormat F, bool Fast>
struct is_integer_operand<IntegerOperand<T, F, Fast>> : std::true_type {};
// The integer fragment operation is shared with the packed IU4 core branch.
// A/B carry signed bytes or signed nibbles; integer addition is exact.
template<int Bits, class Fragment, class Accumulator>
__device__ __forceinline__ Accumulator integer_wmma(Fragment a, Fragment b, Accumulator c) {
    static_assert(Bits == 4 || Bits == 8);
    if constexpr (Bits == 4)
        return __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, false);
    else return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
}
enum class OutputTransform { hadamard128, none };
// int_flush=32: groups c,c+32,... from +0, then class bits 4..0.
template<class OperandPolicy, bool Relaxed = false> struct IntegerContract {
    using Operand = OperandPolicy;
    static constexpr unsigned version = 1, int_flush = Relaxed ? 1 : 32;
    static constexpr bool relaxed = Relaxed;
    static constexpr OutputTransform out_transform = OutputTransform::none;
};
struct A8Result { signed char code; float delta; int sum; };
__device__ __forceinline__ A8Result quantize_a8_row32(float x) {
    float amax = fabsf(x);
    for (int stride = 16; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 32));
    const bool tiny = amax > 0.0f && amax < 0x1p-119f;
    const float d = amax == 0.0f || tiny ? 1.0f : amax / 127.0f;
    const float rounded = tiny ? 0.0f : rintf(x / d);
    const int code = int(fminf(127.0f, fmaxf(-127.0f, rounded)));
    int sum = code;
    for (int stride = 16; stride; stride /= 2) sum += __shfl_xor(sum, stride, 32);
    return {static_cast<signed char>(code), d, sum};
}
}
