#pragma once
// Sync-free device MoE route planner shared by the prefill engines (GLM exact MoE, MiniMax prefill MoE), plus the
// device item queue that lets grouped launches run from a device-built item table without a host round trip.
//
// Planner: three launches on the caller's stream, no host synchronization, no allocation.
//   rank    one thread per row: canonical ids, rank of the row among the block's earlier rows holding each expert
//   scan    one block: per-expert block prefixes, expert counts, then the policy's group/tile/item tables
//   scatter one thread per row: each (row, slot) pair at off[expert] + block prefix + rank, plus the shared pair
// Expert e's pairs come out in ascending row order, experts in ascending id, the shared expert (id kExperts, every row)
// last: the order of the engines' host plans, so the tables are byte-equal to them.
//
// Policy contract (engines keep only this glue):
//   static constexpr unsigned kExperts, kTopK;          routed experts, slots per row
//   static constexpr unsigned kRows;                    rows per rank/scatter block (multiple of 32)
//   static constexpr bool kDropInvalid;                 invalid row: true = no routed pairs, false = P::fallback ids
//   using Out = ...;                                    engine tables, carried by value in Args
//   __device__ static bool load(const Args<P>&, unsigned row, unsigned (&e)[kTopK], float (&w)[kTopK]);
//                                                       canonical slot order; returns true when the row is invalid
//   __device__ static void fallback(unsigned (&e)[kTopK], float (&w)[kTopK]);  (kDropInvalid == false)
//   __device__ static void finish(const Args<P>&, const unsigned* count, const unsigned* start);
//                                                       scan block, every thread; count/start [kExperts + 1] in LDS
//   __device__ static void routed(const Args<P>&, unsigned row, unsigned slot, unsigned e, float w, unsigned pos);
//   __device__ static void shared(const Args<P>&, unsigned row, unsigned pos);
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::moe_core::device_plan {

template <class P> struct Args {
    unsigned rows;
    const std::uint32_t* ids;  // device [rows][kTopK]
    const float* weights;      // device [rows][kTopK] (optional for policies that ignore weights)
    unsigned* blk;             // [blocks][kExperts] counts, then exclusive per-expert block prefixes
    unsigned* rank;            // [rows][kTopK] rank among the block's earlier rows holding the same expert
    unsigned* off;             // [kExperts + 1] expert offsets ([kExperts] = shared), policy may use further words
    unsigned* flag;            // set (|= 1) when any row is invalid
    typename P::Out out;
};

constexpr std::size_t align256(std::size_t v) { return (v + 255) & ~std::size_t{255}; }
// Scratch of the planner for up to `rows` rows: blk, rank, off (+ 2 policy words), each 256-byte aligned.
template <class P> constexpr std::size_t blocks_for(unsigned rows) { return (std::size_t{rows} + P::kRows - 1U) / P::kRows; }
template <class P> constexpr std::size_t scratch_bytes(unsigned rows) {
    return align256(blocks_for<P>(rows) * P::kExperts * sizeof(unsigned)) +
           align256(std::size_t{rows} * P::kTopK * sizeof(unsigned)) + align256((P::kExperts + 2U) * sizeof(unsigned));
}
// Carve the planner scratch out of `base` (sized by scratch_bytes(capacity)).
template <class P> inline void bind_scratch(Args<P>& a, void* base, unsigned capacity) {
    auto* b = static_cast<unsigned char*>(base);
    a.blk = reinterpret_cast<unsigned*>(b);
    a.rank = reinterpret_cast<unsigned*>(b + align256(blocks_for<P>(capacity) * P::kExperts * sizeof(unsigned)));
    a.off = a.rank + align256(std::size_t{capacity} * P::kTopK * sizeof(unsigned)) / sizeof(unsigned);
}

template <class P>
__device__ __forceinline__ bool load_row(const Args<P>& a, unsigned r, unsigned (&e)[P::kTopK], float (&w)[P::kTopK]) {
    bool bad = false;
    if (r < a.rows) {
        bad = P::load(a, r, e, w);
    } else {
#pragma unroll
        for (unsigned k = 0; k < P::kTopK; ++k) e[k] = P::kExperts, w[k] = 0.0F;
    }
    if (bad) {
        if constexpr (P::kDropInvalid) {
#pragma unroll
            for (unsigned k = 0; k < P::kTopK; ++k) e[k] = P::kExperts;
        } else {
            P::fallback(e, w);
        }
    }
    return bad;
}

template <class P> __global__ __launch_bounds__(P::kRows) void rank_kernel(Args<P> p) {
    constexpr unsigned E = P::kExperts, K = P::kTopK, W = P::kRows / 32U;
    __shared__ unsigned cnt[W][E];
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U, r = blockIdx.x * P::kRows + tid;
    for (unsigned i = tid; i < W * E; i += P::kRows) (&cnt[0][0])[i] = 0;
    unsigned e[K];
    float w[K];
    if (load_row(p, r, e, w)) atomicOr(p.flag, 1U);
    __syncthreads();
    // Wave-local ranks: one pass per distinct expert held by any lane; lanes below hold earlier rows.
    unsigned rk[K] = {}, done = 0;
    constexpr unsigned all = (1U << K) - 1U;
#pragma unroll
    for (unsigned k = 0; k < K; ++k) if (e[k] >= E) done |= 1U << k;
    for (;;) {
        const unsigned pending = static_cast<unsigned>(__ballot(done != all));
        if (!pending) break;
        const unsigned leader = static_cast<unsigned>(__builtin_ctz(pending));
        unsigned mine = E;
#pragma unroll
        for (unsigned k = K; k-- > 0;) if (!(done >> k & 1U)) mine = e[k];
        const unsigned x = static_cast<unsigned>(__shfl(static_cast<int>(mine), static_cast<int>(leader)));
        unsigned slot = K;
#pragma unroll
        for (unsigned k = 0; k < K; ++k) if (e[k] == x) slot = k;
        const unsigned holders = static_cast<unsigned>(__ballot(slot < K));
        if (slot < K) {
            rk[slot] = cnt[wave][x] + static_cast<unsigned>(__builtin_popcount(holders & ((1U << lane) - 1U)));
            done |= 1U << slot;
        }
        if (lane == leader) cnt[wave][x] += static_cast<unsigned>(__builtin_popcount(holders));
    }
    __syncthreads();
    // Waves cover consecutive rows: prefix the wave counts per expert, publish the block's counts.
    for (unsigned x = tid; x < E; x += P::kRows) {
        unsigned run = 0;
        for (unsigned v = 0; v < W; ++v) { const unsigned c = cnt[v][x]; cnt[v][x] = run; run += c; }
        p.blk[std::size_t{blockIdx.x} * E + x] = run;
    }
    __syncthreads();
    if (r < p.rows)
#pragma unroll
        for (unsigned k = 0; k < K; ++k) p.rank[std::size_t{r} * K + k] = e[k] < E ? rk[k] + cnt[wave][e[k]] : 0U;
}

template <class P> __global__ __launch_bounds__(1024) void scan_kernel(Args<P> p, unsigned blocks) {
    constexpr unsigned E = P::kExperts;
    __shared__ unsigned count[E + 1], start[E + 1];
    const unsigned tid = threadIdx.x;
    for (unsigned x = tid; x < E; x += blockDim.x) {
        unsigned run = 0;
        for (unsigned b = 0; b < blocks; ++b) {
            unsigned* c = p.blk + std::size_t{b} * E + x;
            const unsigned v = *c;
            *c = run;
            run += v;
        }
        count[x] = run;
    }
    if (tid == 0) count[E] = p.rows;
    __syncthreads();
    if (tid == 0) {
        unsigned n = 0;
        for (unsigned x = 0; x <= E; ++x) {
            start[x] = n;
            p.off[x] = n;
            n += count[x];
        }
    }
    __syncthreads();
    P::finish(p, count, start);
}

template <class P> __global__ __launch_bounds__(P::kRows) void scatter_kernel(Args<P> p) {
    constexpr unsigned E = P::kExperts, K = P::kTopK;
    const unsigned r = blockIdx.x * P::kRows + threadIdx.x;
    if (r >= p.rows) return;
    unsigned e[K];
    float w[K];
    const bool bad = load_row(p, r, e, w);
    if (!(bad && P::kDropInvalid))
#pragma unroll
        for (unsigned k = 0; k < K; ++k)
            P::routed(p, r, k, e[k], w[k], p.off[e[k]] + p.blk[std::size_t{blockIdx.x} * E + e[k]] + p.rank[std::size_t{r} * K + k]);
    P::shared(p, r, p.off[E] + r);
}

// The three planner launches (rows > 0). Calls sharing scratch/tables must be ordered on `stream`.
template <class P> inline hipError_t enqueue(const Args<P>& a, hipStream_t stream) {
    const auto blocks = static_cast<unsigned>(blocks_for<P>(a.rows));
    rank_kernel<P><<<blocks, P::kRows, 0, stream>>>(a);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return e;
    scan_kernel<P><<<1, 1024, 0, stream>>>(a, blocks);
    if ((e = hipGetLastError()) != hipSuccess) return e;
    scatter_kernel<P><<<blocks, P::kRows, 0, stream>>>(a);
    return hipGetLastError();
}

// ---------------------------------------------------------------- device item queue
// A grouped launch whose item table (cta_begin prefixes ascending, `*count` items, `*total` CTAs) was built on the
// device. Persistent: every CTA takes virtual CTA indices from `*next` (zeroed by the planner) until `*total`; the grid
// is the resident capacity, not the (device-only) CTA count. Direct: virtual CTA = blockIdx.x, for launches whose grid
// is a proven bound of `*total` (surplus CTAs return at once; the planner flags a total above the bound). Each virtual
// CTA runs exactly the work of the host-planned launch's CTA of the same index, so outputs are bitwise those of the
// host table. The virtual index is wave-uniform (readfirstlane), so the item is read with scalar loads as in the host
// kernels and the body's register allocation is unchanged.
template <class ItemT> struct ItemQueue {
    const ItemT* items;
    const unsigned* count;
    const unsigned* total;
    unsigned* next;
};

template <class ItemT>
__device__ __forceinline__ const ItemT& find_item(const ItemT* items, unsigned count, unsigned vb) {
    unsigned lo = 0, hi = count;  // last item with cta_begin <= vb
    while (hi - lo > 1) {
        const unsigned mid = (lo + hi) / 2;
        if (items[mid].cta_begin <= vb) lo = mid;
        else hi = mid;
    }
    return items[lo];
}

template <class ItemT, class Body>
__device__ __forceinline__ void run_item(const ItemQueue<ItemT>& q, unsigned count, unsigned vb, Body&& body) {
    const ItemT it = find_item(q.items, count, vb);
    const unsigned local = vb - it.cta_begin;
    body(it, local % it.gx, (local / it.gx) % it.gy, local / (it.gx * it.gy));
}

// body(item, bx, by, bz) with the host kernels' decomposition of the CTA index inside the item.
template <bool Persistent, class ItemT, class Body>
__device__ __forceinline__ void run_items(const ItemQueue<ItemT>& q, Body&& body) {
    const unsigned total = *q.total;
    if constexpr (Persistent) {
        __shared__ unsigned s_vb;
        const unsigned count = *q.count;
        for (;;) {
            if (threadIdx.x == 0) s_vb = atomicAdd(q.next, 1U);
            __syncthreads();
            const unsigned vb = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(s_vb)));
            if (vb >= total) return;
            run_item(q, count, vb, body);
            __syncthreads();  // s_vb and the body's LDS are reused by the next virtual CTA
        }
    } else {
        if (blockIdx.x >= total) return;
        run_item(q, *q.count, blockIdx.x, body);
    }
}

}  // namespace strix::moe_core::device_plan
