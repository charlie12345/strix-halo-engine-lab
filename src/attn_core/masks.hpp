#pragma once
// attn_core masks and block skip: cores call these, never re-derive them. Policies keep live(kt) (idx, MSA selected
// blocks) as their own block-skip hook on top.
#include <hip/hip_runtime.h>

namespace strix::attn_core::mask {

// Sliding window: keys p - window + 1 .. p; window 0 = full causal.
__host__ __device__ inline unsigned window_lo(unsigned p, unsigned window) { return window && p >= window ? p - window + 1U : 0U; }
__host__ __device__ inline bool causal_ok(unsigned key, unsigned pos, unsigned lo) { return key <= pos && key >= lo; }
// Keys a run of consecutive query positions [first, last] can see: [begin, end) (block skip outside it).
struct Range {
    unsigned begin, end;
};
__host__ __device__ inline Range tile_range(unsigned first, unsigned last, unsigned window) { return {window_lo(first, window), last + 1U}; }
// Per-row key lists: entry j of a list of n is live when j < n; dead entries load entry n - 1 (in bounds, finite)
// and are masked to -inf (n >= 1).
__host__ __device__ inline bool list_ok(unsigned j, unsigned n) { return j < n; }
__host__ __device__ inline unsigned list_clamp(unsigned j, unsigned n) { return j < n ? j : n - 1U; }

}  // namespace strix::attn_core::mask
