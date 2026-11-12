#pragma once
// Shared dense FP16 GEMM for prefill row batches on v_wmma_f32_16x16x16_f16: y[r][o] = sum_k fp16(x[r][k]) * w[o][k].
//
// Class: FAST (not the exact FP32 chains). Arithmetic contract: x is rounded to FP16 by the _Float16 cast (RNE, finite
// values beyond the FP16 range become inf, NaN stays NaN); the weights are FP16 as stored; per (row tile, column tile)
// one WMMA chain from +0 over the 16-wide k units in ascending order (FP32 accumulation in hardware order).
//
// Geometry (core<TC>): one CTA = 128 rows x 16 TC columns, 8 waves; wave w owns row tile w and all TC column tiles,
// so a row fragment (the only in-loop VALU work: 16 conversions) feeds TC WMMAs.
//   rows    : A fragment = 16 FP32 values of the lane's row, converted in registers (lane = row); never staged.
//   weights : per stage of four k units, thread i of the CTA copies the 64 FP16 weights of column i into LDS (once per
//             CTA, so a weight byte is read from memory once per 128 rows); every wave reads its B fragments from
//             there (lane = output column). The next stage's weights are requested before the current WMMAs.
// The k loop holds two stages per iteration: the accumulators are loop-carried and a back edge costs register moves.
// Cols policy: tile(i) -> ColumnTile for column tile i of the launch (16 consecutive outputs). Dead tiles (padding of
// the last CTA column) must still name readable weights; they are multiplied and never stored.
// Limits (callers check): in % 128 == 0, rows * in * 4 < 2^32.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::core::f16_dense_tall {
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
using Float4 = float __attribute__((ext_vector_type(4)));
using U4 = std::uint32_t __attribute__((ext_vector_type(4)));

inline constexpr unsigned kRows = 128;  // rows per CTA

struct ColumnTile {
    const _Float16* w;  // weight rows of the tile's 16 outputs, [16][in] row-major
    float* y;           // output of row 0 for the tile's first column
    unsigned stride;    // floats between consecutive rows of y
    bool live;          // false: computed, never stored
};

namespace detail {
using GlobalF4 = const Float4 __attribute__((address_space(1), aligned(4)));
using GlobalU4 = const U4 __attribute__((address_space(1), aligned(2)));

__device__ __forceinline__ unsigned uni(unsigned v) {  // wave-uniform value -> scalar register
    return static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(v)));
}
template <class T> __device__ __forceinline__ T* uniform(T* p) {  // wave-uniform pointer -> scalar registers (saddr loads)
    const auto v = reinterpret_cast<std::uintptr_t>(p);
    const auto lo = uni(static_cast<std::uint32_t>(v)), hi = uni(static_cast<std::uint32_t>(v >> 32U));
    return reinterpret_cast<T*>((static_cast<std::uintptr_t>(hi) << 32U) | lo);
}
__device__ __forceinline__ Half16 row_fragment(const unsigned char* x, std::uint32_t offset) {
    asm("" : "+v"(offset));
    Float4 f[4];
#pragma unroll
    for (unsigned i = 0; i < 4U; ++i) f[i] = *(GlobalF4*)(x + offset + 16U * i);
    Half16 h;
#pragma unroll
    for (unsigned i = 0; i < 16U; ++i) h[i] = static_cast<_Float16>(f[i / 4U][i % 4U]);
    return h;
}
__device__ __forceinline__ Half16 staged_fragment(const std::uint32_t* p) {
    struct { U4 l, h; } v{*reinterpret_cast<const U4*>(p), *reinterpret_cast<const U4*>(p + 4)};
    return __builtin_bit_cast(Half16, v);
}
}  // namespace detail

// Pitch 36 dwords per column: b128 accesses stay aligned and columns rotate banks.
struct alignas(16) Storage {
    std::uint32_t w[256][36];
};
static_assert(sizeof(Storage) == 36864);

template <unsigned TC, class Cols>
__device__ __forceinline__ void core(const float* x, unsigned in, unsigned nr, Cols cols, Storage& lds) {
    static_assert(TC == 4 || TC == 8 || TC == 16);
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U, c = lane & 15U;
    const unsigned row0 = blockIdx.x * kRows + wave * 16U;
    const unsigned units = detail::uni(in) / 16U;
    const auto* xb = reinterpret_cast<const unsigned char*>(detail::uniform(x));
    // Rows past nr read row 0 (their sums are never stored).
    const std::uint32_t xo = (row0 + c < nr ? row0 + c : 0U) * in * 4U;
    // Stager role: thread tid owns column tid of the CTA's column block.
    const bool stager = tid < 16U * TC;
    const auto* wp = reinterpret_cast<const unsigned char*>(cols.tile(blockIdx.y * TC + (stager ? tid / 16U : 0U)).w) +
                     static_cast<std::size_t>(tid & 15U) * in * 2U;
    U4 nb[8];
    const auto fetch = [&](unsigned unit) {
#pragma unroll
        for (unsigned i = 0; i < 8U; ++i) nb[i] = *(detail::GlobalU4*)(wp + std::size_t{unit} * 32U + 16U * i);
    };
    Float8 acc[TC] = {};
    Half16 an = detail::row_fragment(xb, xo);
    if (stager) fetch(0U);
    for (unsigned base = 0; base < units; base += 8U) {
#pragma unroll
        for (unsigned s = 0; s < 2U; ++s) {
            const unsigned u0 = base + 4U * s;
            __syncthreads();  // every wave finished reading the previous stage
            if (stager) {
#pragma unroll
                for (unsigned i = 0; i < 8U; ++i) *reinterpret_cast<U4*>(&lds.w[tid][4U * i]) = nb[i];
            }
            __syncthreads();
            if (stager && (s == 0U || u0 + 4U < units)) fetch(u0 + 4U);
            asm volatile("" ::: "memory");  // the requests stay ahead of this stage's WMMAs
#pragma unroll
            for (unsigned kk = 0; kk < 4U; ++kk) {
                const Half16 a = an;
                if (kk + 1U < 4U || s == 0U || u0 + 4U < units) an = detail::row_fragment(xb, xo + (u0 + kk + 1U) * 64U);
#pragma unroll
                for (unsigned t = 0; t < TC; ++t)
                    acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, detail::staged_fragment(&lds.w[16U * t + c][kk * 8U]), acc[t]);
            }
        }
    }
    // acc[t][j] -> row row0 + 2 j + lane / 16, column 16 t + lane % 16.
#pragma unroll
    for (unsigned t = 0; t < TC; ++t) {
        const ColumnTile ct = cols.tile(blockIdx.y * TC + t);
        if (!detail::uni(ct.live ? 1U : 0U)) continue;
        float* const y = detail::uniform(ct.y);
        const unsigned stride = detail::uni(ct.stride);
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            const unsigned r = row0 + 2U * j + (lane >> 4U);
            if (r < nr) y[static_cast<std::size_t>(r) * stride + c] = acc[t][j];
        }
    }
}

}  // namespace strix::core::f16_dense_tall
