#pragma once

#include "../../moe_core/exl3_trellis.hpp"

namespace strix::minimax::moe::opt {

template <int Bits> using WeightDecode = strix::moe_core::Exl3Trellis<Bits>;

struct ExpertLayout {
    // Expert-major packed streams; gate/up work items are consecutive, separate matrices.
    static constexpr unsigned kGateUpInterleave = 2;
    static constexpr unsigned kRowBlock = 16;
    template <class Geometry, class Columns>
    static __host__ __device__ constexpr std::size_t packed_stride(Columns columns) {
        return static_cast<std::size_t>(columns) * Geometry::kWords;
    }
};

struct InputTransform : WeightDecode<3>::InputTransform {
    // RN16(H128(RN16(RN32(x*suh)))); the bit width does not affect this transform.
    using WeightDecode<3>::InputTransform::input_block;
};

struct Epilogue {
    static constexpr float kAlpha = 1.702F, kLimit = 7.0F;
static __device__ __forceinline__ float swiglu(float g, float u) {  // tests/minimax_gate_common.hpp swiglu_kernel
    const float gg = g < kLimit ? g : kLimit;
    const float uu = u < -kLimit ? -kLimit : (u > kLimit ? kLimit : u);
    return gg * (1.0F / (1.0F + expf(-kAlpha * gg))) * (uu + 1.0F);
}

static __device__ __forceinline__ void finish_block(const float* part, unsigned splits, unsigned rows, unsigned row, unsigned output,
                                             unsigned b, const std::uint16_t* svh, unsigned lane, float (&v)[4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = 0.0F;
    for (unsigned s = 0; s < splits; ++s)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            v[j] += part[(static_cast<std::size_t>(s) * rows + row) * output + b * 128 + lane + 32U * static_cast<unsigned>(j)];
    strix::moe_core::hadamard128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = v[j] * strix::moe_core::half_bits(svh[b * 128 + lane + 32U * static_cast<unsigned>(j)]);
}

    static __device__ __forceinline__ float router_accumulate(float w, float v, float acc) {
        return __builtin_fmaf(w, v, acc);
    }
    static __device__ __forceinline__ float shared_add(float acc, float v) { return acc + v; }
};

}  // namespace strix::minimax::moe::opt
