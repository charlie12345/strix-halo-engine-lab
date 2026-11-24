#pragma once
// Small-M schedule of the f16 tile GEMM family for low-bit resident rows (q4_g64 / q6_g64, strix/weight/host_lowbit.hpp):
//   y[r][n] = sum_g d[n][g] * sum_{k in group g} fp16(a[r][k]) * code[n][k]      (r < rows <= 48 per row block)
// Weight bytes are read once per launch for up to 48 rows (one row block); FP16 WMMA does the row products, so the cost
// per weight does not grow with the row count the way the VALU rows sweep does (core/dense_linear.hpp).
//
// Class: FAST (FP16 activations, as the resident tile route). Arithmetic: codes are exact FP16 integers
// (core/lowbit_f16_staging.hpp); per output, per K64 group one WMMA chain from +0 over the four K16 units in ascending
// order, then acc = fma(group sum, float(d), acc); groups ascending within a split; splits summed in ascending order
// ((s0 + s1) + s2) + s3. The bits depend on the split count only (a function of the shape, never of the row count, see
// core/resident_row_plan.hpp plan_lowbit_rows), not on the grid, the row count or the row tile count.
//
// CTA: one 64-column projection panel (the panel of the projection tile), 4 waves x 16 columns per K split, S = 1/2/4
// splits (128 S threads). A stage covers S blocks of kBlockGroups groups (K = 128 S for <= 2 row tiles, 64 S for 3):
// all threads stage A (FP32 -> FP16) into LDS once per stage; the wave of split s multiplies block stage * S + s. A lane
// holds ONE column (lane % 16; lanes 16..31 replicate it, as the WMMA operand layout requires): it loads that column's
// block (q4 32 B codes per group, q6 + 16 B 2-bit plane, the scale word) kPrefetch stages ahead, so its accumulator
// elements share one scale per group.
// The K loop (run) is the shared code-as-operand trait of the family: an Operand trait supplies the column's codes as
// exact FP16 WMMA operands and one FP32 multiplier per group (LowBitOperand here; NVFP4 / MXFP8 in
// core/microscaled_f16_staging.hpp), a Tile binding supplies rows, weights, A chunks and the store (ProjectionTile here;
// the grouped expert tiles in moe_core/grouped_microscaled.hpp).

#include "core/f16_tile_gemm.hpp"
#include "core/lowbit_f16_staging.hpp"

#if defined(__HIPCC__)
namespace strix::core::f16_tile_gemm::small_m {

inline constexpr unsigned kWaveColumns = 16U, kMaxRowTiles = 3U, kMaxRows = 16U * kMaxRowTiles;

// Per wave and stage: a whole superblock (two groups) for <= 2 row tiles, one group for three (whose A fragments, sums
// and staged A take the registers). Blocks in flight per wave: two, one for three row tiles.
template <unsigned RowTiles> inline constexpr unsigned kBlockGroups = RowTiles <= 2U ? 2U : 1U;
template <unsigned RowTiles> inline constexpr unsigned kPrefetch = RowTiles <= 2U ? 2U : 1U;

template <unsigned RowTiles, unsigned Splits, unsigned SuperK>
struct Schedule {
    static_assert(RowTiles >= 1U && RowTiles <= kMaxRowTiles && (Splits == 1U || Splits == 2U || Splits == 4U));
    static constexpr unsigned kRows = 16U * RowTiles, kThreads = 128U * Splits, kStageK = SuperK * Splits;
    static constexpr unsigned kChunksPerRow = kStageK / 8U, kAChunks = kRows * kChunksPerRow / kThreads;
    static constexpr unsigned kPitch = kStageK / 2U + 4U;  // dwords per LDS row: data + 4 pad (b128 reads conflict free)
    static_assert(kRows * kChunksPerRow % kThreads == 0U);
    struct alignas(16) Storage {
        std::uint32_t a[kRows][kPitch];
    };
    // The split reduction exchanges 4 waves x RowTiles x 8 x 32 FP32 through the stage memory.
    static_assert(Splits == 1U || sizeof(Storage) >= 4U * RowTiles * 8U * 32U * sizeof(float));
    static_assert(sizeof(Storage) <= 65536U);
};

// Groups groups (1 or 2) of one column, group index 2 b + g0 .. of the row: Groups = 2 is superblock b, Groups = 1 is
// group g of superblock g / 2.
template <class Storage, unsigned Groups>
struct BlockWords {
    lowbit_f16::Words4 c[2U * Groups];  // 4-bit plane, 32 bytes per group, element j in nibble j of the byte stream
    lowbit_f16::Words4 h[Storage::high_bytes != 0U ? Groups : 1U];  // q6 2-bit plane, 16 bytes per group
    std::uint32_t scale;  // FP16 scale of the block's group gi in bits 16 gi .. 16 gi + 15
};

template <class Storage, unsigned Groups>
__device__ __forceinline__ BlockWords<Storage, Groups> load_block(const std::uint8_t* row, unsigned index) {
    const unsigned gi = Groups == 2U ? 0U : index % 2U;
    const auto* block = row + std::size_t{Groups == 2U ? index : index / 2U} * Storage::bytes;
    BlockWords<Storage, Groups> w{};
#pragma unroll
    for (unsigned i = 0; i < 2U * Groups; ++i) w.c[i] = *(lowbit_f16::Words4G*)(block + 32U * gi + 16U * i);
    if constexpr (Storage::high_bytes != 0U)
#pragma unroll
        for (unsigned i = 0; i < Groups; ++i)
            w.h[i] = *(lowbit_f16::Words4G*)(block + Storage::low_bytes + 16U * gi + 16U * i);
    w.scale = *(lowbit_f16::WordsG*)(block + Storage::scale_offset) >> (16U * gi);
    return w;
}

// Operand trait of the q4_g64 / q6_g64 resident rows: Words = Groups (1 or 2) groups of one column, unit u of group gi =
// the exact FP16 integer codes of elements 64 gi + 16 u .. + 16 of the block, scale = the group's FP16 scale.
template <class Storage>
struct LowBitOperand {
    static constexpr unsigned kGroup = Storage::group, kUnits = Storage::group / 16U, kSuperK = Storage::elements;
    static_assert(kGroup == 64U && kSuperK == 128U);
    using Row = const std::uint8_t*;
    template <unsigned Groups> using Words = BlockWords<Storage, Groups>;
    template <unsigned Groups>
    __device__ __forceinline__ static Words<Groups> load(Row row, unsigned index) {
        return load_block<Storage, Groups>(row, index);
    }
    template <unsigned Groups>
    __device__ __forceinline__ static void unit(const Words<Groups>& w, unsigned gi, unsigned u, std::uint32_t (&words)[8]) {
        // Unit u of group gi: code words 8 gi + 2 u and + 1 (8 elements each), 2-bit plane word 4 gi + u.
        const unsigned word = 8U * gi + 2U * u;
        std::uint32_t hi = 0U;
        if constexpr (Storage::high_bytes != 0U) hi = w.h[gi][u];
        lowbit_f16::integer_pairs<Storage>(w.c[word / 4U][word % 4U], w.c[word / 4U][word % 4U + 1U], hi, words);
    }
    template <unsigned Groups>
    __device__ __forceinline__ static float scale(const Words<Groups>& w, unsigned gi) {
        const auto bits = static_cast<std::uint16_t>(w.scale >> (16U * gi));
        return static_cast<float>(__builtin_bit_cast(_Float16, bits));
    }
};

// The K loop of the small-M schedule over one bound tile. Operand: the weight trait (LowBitOperand, or
// core/microscaled_f16_staging.hpp): kGroup elements per scale group (a multiple of 16, dividing 64), kUnits = kGroup / 16
// WMMA units per group, kSuperK = 128 elements per superblock, Words<Groups> = the 64 Groups elements of a column a stage
// reads, unit() = FP16 pairs of 16 of them, scale() = the group's FP32 multiplier. Tile (a binding): K (% 128 == 0), rows
// (live rows of the tile), outputs, row(column) (the column's weight row), AChunk / load_a(row, k, in) / pack_a (8 A
// elements of row `row` from k, zero when !in, as 4 FP16 pair words) and store(row, column, sum).
template <class Operand, unsigned RowTiles, unsigned Splits, class Tile>
__device__ __forceinline__ void run(const Tile& t, unsigned r0, unsigned col0) {
    constexpr unsigned kGroups = kBlockGroups<RowTiles>, kDepth = kPrefetch<RowTiles>;
    constexpr unsigned kStageGroups = 64U * kGroups / Operand::kGroup;
    using Sched = Schedule<RowTiles, Splits, 64U * kGroups>;
    using Words = typename Operand::template Words<kGroups>;
    __shared__ typename Sched::Storage lds;
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U, wn = wave & 3U, split = wave >> 2U;
    const unsigned c = lane & 15U;
    const unsigned K = t.K, rows = t.rows;
    const unsigned col = col0 + kWaveColumns * wn + c;
    const bool live = col < t.outputs;
    // Columns past the outputs read the panel's first column (valid bytes) and never store.
    const typename Operand::Row wrow = t.row(live ? col : col0);
    const unsigned live_tiles = rows - r0 >= Sched::kRows ? RowTiles : (rows - r0 + 15U) / 16U;
    // Split s owns superblocks s, s + S, ... (groups in ascending order) whatever the block size, so the partition of
    // the groups over the splits (and the bits) does not depend on the row tile count. Block of split s at stage t:
    // superblock (t / per) S + s, group half t % per (per = 2 / kGroups stages per superblock).
    constexpr unsigned kPer = 2U / kGroups;
    const unsigned supers = K / Operand::kSuperK, stages = kPer * ((supers + Splits - 1U) / Splits);
    const auto super_of = [&](unsigned st, unsigned s) { return (st / kPer) * Splits + s; };
    const auto block_of = [&](unsigned st, unsigned s) {
        return kGroups == 2U ? super_of(st, s) : 2U * super_of(st, s) + st % kPer;
    };

    // A stager: 8-element chunk q = tid + kThreads j of the stage = row q / kChunksPerRow, k 8 (q % kChunksPerRow).
    // Rows past `rows` read row 0 (never stored); k past K (last stage of a split count that does not divide the
    // superblocks) stages zeros.
    typename Tile::AChunk ra[Sched::kAChunks];
    const auto load_a = [&](unsigned stage) __attribute__((always_inline)) {
#pragma unroll
        for (unsigned j = 0; j < Sched::kAChunks; ++j) {
            const unsigned q = tid + Sched::kThreads * j, row = q / Sched::kChunksPerRow;
            const unsigned kl = 8U * (q % Sched::kChunksPerRow), s = kl / (64U * kGroups);
            const unsigned k = super_of(stage, s) < supers ? 64U * kGroups * block_of(stage, s) + kl % (64U * kGroups) : K;
            ra[j] = t.load_a(r0 + row < rows ? r0 + row : 0U, k, k < K);
        }
    };
    const auto publish_a = [&]() __attribute__((always_inline)) {
#pragma unroll
        for (unsigned j = 0; j < Sched::kAChunks; ++j) {
            const unsigned q = tid + Sched::kThreads * j, row = q / Sched::kChunksPerRow;
            const unsigned dword = 4U * (q % Sched::kChunksPerRow);
            if constexpr (requires { t.pack_a(ra[j], row); })
                *reinterpret_cast<U4*>(&lds.a[row][dword]) = t.pack_a(ra[j], row);
            else
                *reinterpret_cast<U4*>(&lds.a[row][dword]) = Tile::pack_a(ra[j]);
        }
    };

    Words ring[kDepth];
#pragma unroll
    for (unsigned p = 0; p < kDepth; ++p) {
        ring[p] = super_of(p, split) < supers ? Operand::template load<kGroups>(wrow, block_of(p, split)) : Words{};
    }
    load_a(0U);
    Float8 acc[RowTiles] = {};
    for (unsigned t0 = 0; t0 < stages; t0 += kDepth) {
#pragma unroll
        for (unsigned p = 0; p < kDepth; ++p) {
            const unsigned st = t0 + p;
            if (st >= stages) break;  // uniform
            __syncthreads();  // every wave finished reading the previous stage
            publish_a();
            __syncthreads();
            if (st + 1U < stages) load_a(st + 1U);  // in flight across this stage's WMMAs
            if (super_of(st, split) >= supers) continue;  // uniform per wave: a split count that does not divide
            const Words w = ring[p];
            if (super_of(st + kDepth, split) < supers)
                ring[p] = Operand::template load<kGroups>(wrow, block_of(st + kDepth, split));
#pragma unroll
            for (unsigned gi = 0; gi < kStageGroups; ++gi) {
                Float8 sum[RowTiles] = {};  // the group chain starts from +0
#pragma unroll
                for (unsigned u = 0; u < Operand::kUnits; ++u) {
                    std::uint32_t words[8];
                    Operand::template unit<kGroups>(w, gi, u, words);
                    const Frag fb{words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]};
#pragma unroll
                    for (unsigned m = 0; m < RowTiles; ++m)
                        if (m < live_tiles) {
                            const Frag fa = detail::fragment(
                                &lds.a[16U * m + c][32U * kGroups * split + 8U * (Operand::kUnits * gi + u)]);
                            sum[m] = STRIX_HIP_WMMA_F16(__builtin_bit_cast(Half16, fa), __builtin_bit_cast(Half16, fb), sum[m]);
                        }
                    if constexpr (RowTiles >= 3U) __builtin_amdgcn_sched_barrier(0);  // bound the live A fragments
                }
                const auto d = Operand::template scale<kGroups>(w, gi);
#pragma unroll
                for (unsigned m = 0; m < RowTiles; ++m)
                    if (m < live_tiles)
#pragma unroll
                        for (unsigned j = 0; j < 8U; ++j) {
                            if constexpr (requires { d.partial_multiplier; })
                                acc[m][j] = __builtin_fmaf(sum[m][j] * d.partial_multiplier, d.multiplier, acc[m][j]);
                            else
                                acc[m][j] = __builtin_fmaf(sum[m][j], d, acc[m][j]);
                        }
            }
        }
    }
    if constexpr (Splits > 1U) {
        float* exchange = reinterpret_cast<float*>(&lds);
#pragma unroll
        for (unsigned s = 1; s < Splits; ++s) {
            __syncthreads();  // the K loop's (or the previous round's) LDS reads are done
            if (split == s)
#pragma unroll
                for (unsigned m = 0; m < RowTiles; ++m)
#pragma unroll
                    for (unsigned j = 0; j < 8U; ++j) exchange[((wn * RowTiles + m) * 8U + j) * 32U + lane] = acc[m][j];
            __syncthreads();
            if (split == 0U)
#pragma unroll
                for (unsigned m = 0; m < RowTiles; ++m)
#pragma unroll
                    for (unsigned j = 0; j < 8U; ++j) acc[m][j] += exchange[((wn * RowTiles + m) * 8U + j) * 32U + lane];
        }
    }
    if constexpr (requires { t.before_store(); }) t.before_store();
    if (split != 0U || !live) return;
    // acc[m][j] -> row r0 + 16 m + 2 j + lane / 16, column col.
#pragma unroll
    for (unsigned m = 0; m < RowTiles; ++m)
        if (m < live_tiles)
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) {
                const unsigned r = r0 + 16U * m + 2U * j + (lane >> 4U);
                if (r < rows) t.store(r, col, acc[m][j]);
            }
}

// Binding of a projection group (dense rows, FP32 activations [rows][K] staged as FP16): panel blockIdx.x of the group,
// row block blockIdx.y.
struct ProjectionTile {
    struct AChunk {
        Float4 v[2];
    };
    const float* a;
    unsigned rows, K;
    std::size_t w_row_bytes;
    const Q8Projection& projection;
    unsigned outputs;

    __device__ __forceinline__ const std::uint8_t* row(unsigned column) const {
        return projection.weights + std::size_t{column} * w_row_bytes;
    }
    __device__ __forceinline__ AChunk load_a(unsigned row, unsigned k, bool in) const {
        const float* src = a + std::size_t{row} * K + k;
        AChunk chunk;
        chunk.v[0] = in ? *reinterpret_cast<const Float4*>(src) : Float4{};
        chunk.v[1] = in ? *reinterpret_cast<const Float4*>(src + 4) : Float4{};
        return chunk;
    }
    __device__ __forceinline__ static U4 pack_a(const AChunk& x) {
        return U4{detail::pack(x.v[0][0], x.v[0][1]), detail::pack(x.v[0][2], x.v[0][3]),
                  detail::pack(x.v[1][0], x.v[1][1]), detail::pack(x.v[1][2], x.v[1][3])};
    }
    __device__ __forceinline__ void store(unsigned r, unsigned column, float sum) const {
        projection.output[std::size_t{r} * projection.outputs + column] = sum;
    }
};

template <class Storage, unsigned RowTiles, unsigned Splits, unsigned Capacity>
__global__ __launch_bounds__(128U * Splits)  // Schedule::kThreads
void lowbit_kernel(const float* a, unsigned rows, unsigned K, std::size_t w_row_bytes, Q8ProjectionGroup<Capacity> group) {
    const unsigned panel = blockIdx.x;
    unsigned index = 0U;
    while (index + 1U < group.count && panel >= group.first_panel[index + 1U]) ++index;
    const auto& projection = group.projections[index];
    const unsigned col0 = (panel - group.first_panel[index]) * kProjectionPanel;
    const ProjectionTile tile{a, rows, K, w_row_bytes, projection, projection.outputs};
    run<LowBitOperand<Storage>, RowTiles, Splits>(tile, blockIdx.y * (16U * RowTiles), col0);
}

template <class Storage, unsigned RowTiles, unsigned Capacity>
inline void launch_splits(const float* a, unsigned rows, unsigned K, const Q8ProjectionGroup<Capacity>& group,
                          unsigned splits, hipStream_t stream) {
    const dim3 grid(group.first_panel[group.count], (rows + 16U * RowTiles - 1U) / (16U * RowTiles));
    const auto w_row_bytes = static_cast<std::size_t>(Storage::row_bytes(K));
    if (splits == 4U)
        STRIX_HIP_LAUNCH(Rdna3Layout, lowbit_kernel<Storage, RowTiles, 4U, Capacity><<<grid, 512, 0, stream>>>(a, rows, K, w_row_bytes, group));
    else if (splits == 2U)
        STRIX_HIP_LAUNCH(Rdna3Layout, lowbit_kernel<Storage, RowTiles, 2U, Capacity><<<grid, 256, 0, stream>>>(a, rows, K, w_row_bytes, group));
    else
        STRIX_HIP_LAUNCH(Rdna3Layout, lowbit_kernel<Storage, RowTiles, 1U, Capacity><<<grid, 128, 0, stream>>>(a, rows, K, w_row_bytes, group));
}

// rows >= 1, K % 128 == 0, splits 1 / 2 / 4 from plan_lowbit_rows; rows above 48 run several 48-row blocks.
template <class Storage, unsigned Capacity>
inline hipError_t gemm_lowbit(const float* a, unsigned rows, unsigned K, const Q8ProjectionGroup<Capacity>& group,
                              unsigned splits, hipStream_t stream) {
    static_assert(Capacity <= 32U);
    const unsigned tiles = (rows + 15U) / 16U;
    if (tiles == 1U) launch_splits<Storage, 1U>(a, rows, K, group, splits, stream);
    else if (tiles == 2U) launch_splits<Storage, 2U>(a, rows, K, group, splits, stream);
    else launch_splits<Storage, 3U>(a, rows, K, group, splits, stream);
    return hipGetLastError();
}

}  // namespace strix::core::f16_tile_gemm::small_m
#endif
