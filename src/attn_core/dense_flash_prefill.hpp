#pragma once
// Shared FAST dense causal GQA flash-attention prefill core (f16 WMMA 16x16x16 -> FP32, FP32 online softmax).
//
// Identity: FAST, opt-in per engine (gemma4: STRIX_G4_PREFILL_ATTN=flash). Nothing here runs unless an engine
// launches it. Lossy against exact attention: Q, K, V and the probabilities are FP16 WMMA operands, the score and
// PV sums run in WMMA order, and the softmax is the online (running max) form with exp2. Judged on end metrics.
//
// Shape: one work-group = one KV head x Group query heads x Positions consecutive query positions (Group *
// Positions query rows, 16-row WMMA tiles). Every K/V tile of 16 keys is staged into LDS ONCE per work-group and
// reused by all its query rows (GQA: the whole KV group), instead of once per 16-row tile. Wave w = (row tile
// w / Split, head-dim slice w % Split): the slice's FP16 queries stay in registers as A fragments for the whole
// row tile, the wave computes the slice's partial scores Q[16][D/Split] . K[16][D/Split]^T and owns the output
// accumulators O[16][D/Split]. Per key tile:
//   1. stage: the prefetched K rows ([key][D], padded pitch) and V^T rows ([d][16 keys]) go to LDS;   barrier
//      the next tile's global loads are issued right away and land during the math;
//   2. QK on WMMA, the Split partials go to LDS as FP32;                                             barrier
//   3. every wave of the row tile sums the partials (fixed order), scales by scale*log2(e), applies the causal and
//      sliding-window mask, updates its running max/sum and builds the FP16 P fragment (the two half-waves split
//      the exponentials), rescales O by alpha and adds P[16][16] . V[16][D/Split] on WMMA;           barrier
// Key tiles outside [lo(first row), last row] are never visited (causal + window block skip); tiles a row tile
// cannot see are skipped per wave. Epilogue: O / l to the policy's FP32 output.
// LDS: 16 x (D + 8) x 2 (K) + D x 32 (V^T) + waves x 16 x 20 x 4 (partials) = 53,504 B at D512 / 16 waves.
//
// Policy P (all __device__ static unless noted):
//   using Args;  using Shape = dense::Shape<D, Group, Split, Positions>;  static constexpr float kScaleLog2;
//   bool valid(const Args&)                         geometry check (false: failure |= 2, nothing written)
//   unsigned rows(const Args&)                      query rows (consecutive positions) - also host
//   unsigned position(const Args&)                  absolute position of row 0
//   unsigned window(const Args&)                    sliding window (keys p-window+1 .. p), 0 = full causal
//   const float* query(const Args&, row, head)      FP32 [D], 16-B aligned
//   float* out(const Args&, row, head)              FP32 [D]
//   const _Float16* key(const Args&, pos, kv_head)  FP16 [D] of key pos, 16-B aligned (finite for every pos of a
//                                                   visited 16-key tile; masked keys are multiplied by zero)
//   const _Float16* value_t(const Args&, kv_head, d) FP16 V^T row: element pos = V[pos][d]; 16-B aligned at pos%16==0
//   unsigned* failure(const Args&)                  bit 1 nonfinite output, bit 4 FP16 query overflow
// rows() is __host__ __device__. Launch: launch<P>(args, heads, stream) - grid (row blocks, heads / Group), longest blocks first.
#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::attn_core::dense {

typedef _Float16 Half16 __attribute__((ext_vector_type(16)));
typedef float Float8 __attribute__((ext_vector_type(8)));
typedef float Float4 __attribute__((ext_vector_type(4)));
typedef unsigned U4 __attribute__((ext_vector_type(4)));

template <unsigned Dim, unsigned Group, unsigned Split, unsigned Positions = 16>
struct Shape {
    static_assert(Dim % (16 * Split) == 0, "head dim must split into 16-wide slices");
    static_assert(Positions % 16 == 0, "16 consecutive positions per row tile");
    static constexpr unsigned dim = Dim, group = Group, split = Split, positions = Positions;
    static constexpr unsigned tiles = Group * Positions / 16, waves = tiles * Split, threads = 32 * waves;
    static constexpr unsigned slice = Dim / Split, frags = slice / 16;
    static constexpr unsigned kpitch = Dim + 8, spitch = 20; // halfs / floats
    static constexpr unsigned pieces = 2 * Dim, per_thread = (pieces + threads - 1) / threads;
    static constexpr unsigned lds_bytes = 16 * kpitch * 2 + Dim * 16 * 2 + waves * 16 * spitch * 4;
    static_assert(threads <= 1024, "work-group too large");
    static_assert(lds_bytes <= 65536, "LDS budget");
};

__device__ inline unsigned window_lo(unsigned p, unsigned window) { return window && p >= window ? p - window + 1U : 0U; }

template <class P>
__global__ __launch_bounds__(P::Shape::threads) void flash_prefill(typename P::Args a) {
    using S = typename P::Shape;
    constexpr unsigned D = S::dim, KP = S::kpitch, SP = S::spitch, F = S::frags, T = S::threads;
    __shared__ __attribute__((aligned(16))) _Float16 ks[16 * KP];
    __shared__ __attribute__((aligned(16))) _Float16 vs[D * 16];
    __shared__ __attribute__((aligned(16))) float part[S::waves * 16 * SP];
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5, rr = lane & 15U, hw = lane >> 4;
    const unsigned tile = wave / S::split, piece = wave % S::split;
    if (!P::valid(a)) {
        if (!tid) atomicOr(P::failure(a), 2U);
        return;
    }
    const unsigned rows = P::rows(a), blocks = (rows + S::positions - 1U) / S::positions;
    const unsigned block = blocks - 1U - blockIdx.x, kvh = blockIdx.y, row_base = block * S::positions;
    const unsigned p0 = P::position(a), window = P::window(a);
    const unsigned block_rows = min(S::positions, rows - row_base);
    const unsigned pos_first = p0 + row_base, pos_last = pos_first + block_rows - 1U;
    const unsigned key_begin = window_lo(pos_first, window) & ~15U;
    // This wave's 16 rows: one query head, 16 consecutive positions.
    const unsigned head = kvh * S::group + tile % S::group, sub = tile / S::group * 16U;
    const bool wave_live = sub < block_rows;
    const unsigned tile_last = pos_first + min(sub + 15U, block_rows - 1U), tile_lo = window_lo(pos_first + sub, window);
    const unsigned my_row = row_base + sub + rr, my_pos = p0 + my_row, my_lo = window_lo(my_pos, window);
    const bool live = wave_live && sub + rr < block_rows;
    const unsigned d0 = piece * S::slice;
    bool bad_half = false;
    Half16 qf[F];
    {
        const float* q = live ? P::query(a, my_row, head) + d0 : nullptr;
#pragma unroll
        for (unsigned f = 0; f < F; ++f) {
#pragma unroll
            for (unsigned c = 0; c < 4; ++c) {
                const Float4 x = live ? *reinterpret_cast<const Float4*>(q + 16U * f + 4U * c) : Float4{};
#pragma unroll
                for (unsigned i = 0; i < 4; ++i) {
                    const _Float16 h = static_cast<_Float16>(x[i]);
                    bad_half |= (__builtin_bit_cast(unsigned short, h) & 0x7c00U) == 0x7c00U;
                    qf[f][4U * c + i] = h;
                }
            }
        }
    }
    U4 kreg[S::per_thread], vreg[S::per_thread];
    const auto fetch = [&](unsigned kt) {
#pragma unroll
        for (unsigned i = 0; i < S::per_thread; ++i) {
            const unsigned idx = tid + i * T;
            if (S::pieces % T == 0 || idx < S::pieces) {
                const unsigned key = idx / (D / 8U), c = idx % (D / 8U);
                kreg[i] = *reinterpret_cast<const U4*>(P::key(a, kt + key, kvh) + 8U * c);
                const unsigned d = idx >> 1, h = idx & 1U;
                vreg[i] = *reinterpret_cast<const U4*>(P::value_t(a, kvh, d) + kt + 8U * h);
            }
        }
    };
    const auto stage = [&] {
#pragma unroll
        for (unsigned i = 0; i < S::per_thread; ++i) {
            const unsigned idx = tid + i * T;
            if (S::pieces % T == 0 || idx < S::pieces) {
                const unsigned key = idx / (D / 8U), c = idx % (D / 8U);
                *reinterpret_cast<U4*>(&ks[key * KP + 8U * c]) = kreg[i];
                const unsigned d = idx >> 1, h = idx & 1U;
                *reinterpret_cast<U4*>(&vs[d * 16U + 8U * h]) = vreg[i];
            }
        }
    };
    constexpr float neg_inf = -__builtin_huge_valf();
    float maximum = neg_inf, denominator = 0.0F;
    Float8 acc[F];
#pragma unroll
    for (unsigned f = 0; f < F; ++f) acc[f] = Float8{};
    fetch(key_begin);
    for (unsigned kt = key_begin; kt <= pos_last; kt += 16U) {
        stage();
        __syncthreads();
        if (kt + 16U <= pos_last) fetch(kt + 16U);
        const bool active = wave_live && kt <= tile_last && kt + 15U >= tile_lo; // wave-uniform
        if (active) {
            Float8 s{};
#pragma unroll
            for (unsigned f = 0; f < F; ++f) {
                const Half16 b = *reinterpret_cast<const Half16*>(&ks[rr * KP + d0 + 16U * f]);
                s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qf[f], b, s);
            }
            float* pw = part + wave * 16U * SP;
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) pw[(2U * e + hw) * SP + rr] = s[e];
        }
        __syncthreads();
        if (active) {
            float sc[16];
#pragma unroll
            for (unsigned j = 0; j < 16; ++j) sc[j] = 0.0F;
#pragma unroll
            for (unsigned p = 0; p < S::split; ++p) {
                const float* pp = part + (tile * S::split + p) * 16U * SP + rr * SP;
#pragma unroll
                for (unsigned c = 0; c < 4; ++c) {
                    const Float4 v = *reinterpret_cast<const Float4*>(pp + 4U * c);
#pragma unroll
                    for (unsigned i = 0; i < 4; ++i) sc[4U * c + i] += v[i];
                }
            }
            float m = maximum;
            bool present = false;
#pragma unroll
            for (unsigned j = 0; j < 16; ++j) {
                const unsigned key = kt + j;
                const bool v = live && key >= my_lo && key <= my_pos;
                sc[j] = v ? sc[j] * P::kScaleLog2 : neg_inf;
                m = sc[j] > m ? sc[j] : m;
                present |= v;
            }
            const float alpha = present ? __builtin_amdgcn_exp2f(maximum - m) : 1.0F;
            float own[8];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) own[i] = present ? __builtin_amdgcn_exp2f(sc[8U * hw + i] - m) : 0.0F;
            Half16 pa;
            float sum = 0.0F;
#pragma unroll
            for (unsigned j = 0; j < 16; ++j) {
                const float other = __shfl_xor(own[j % 8U], 16, 32);
                const float p = j / 8U == hw ? own[j % 8U] : other;
                sum += p;
                pa[j] = static_cast<_Float16>(p);
            }
            denominator = denominator * alpha + sum;
            maximum = m;
            float scale[8];
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) scale[e] = __shfl(alpha, int(2U * e + hw), 32);
#pragma unroll
            for (unsigned f = 0; f < F; ++f) {
                const Half16 b = *reinterpret_cast<const Half16*>(&vs[(d0 + 16U * f + rr) * 16U]);
                Float8 c = acc[f];
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) c[e] *= scale[e];
                acc[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, b, c);
            }
        }
        __syncthreads();
    }
    bool bad_value = false;
    if (wave_live) {
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) {
            const unsigned r = sub + 2U * e + hw;
            const float total = __shfl(denominator, int(2U * e + hw), 32);
            if (r < block_rows) {
                float* o = P::out(a, row_base + r, head) + d0;
#pragma unroll
                for (unsigned f = 0; f < F; ++f) {
                    const float value = acc[f][e] / total;
                    bad_value |= (__builtin_bit_cast(unsigned, value) & 0x7f800000U) == 0x7f800000U;
                    o[16U * f + rr] = value;
                }
            }
        }
    }
    if (bad_value) atomicOr(P::failure(a), 1U);
    if (bad_half) atomicOr(P::failure(a), 4U);
}

template <class P>
inline void launch(const typename P::Args& a, unsigned heads, hipStream_t stream) {
    using S = typename P::Shape;
    const unsigned rows = P::rows(a);
    if (!rows) return;
    hipLaunchKernelGGL(flash_prefill<P>, dim3((rows + S::positions - 1U) / S::positions, heads / S::group),
                       dim3(S::threads), 0, stream, a);
}

} // namespace strix::attn_core::dense
