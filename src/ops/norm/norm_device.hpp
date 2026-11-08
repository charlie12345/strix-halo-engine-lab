#pragma once

#include "norm.hpp"

#include <hip/hip_runtime.h>
#include <cstddef>
#include <type_traits>

namespace strix::ops::norm {

// One block per row; L lane-strided FMA chains, then the engine's LDS
// halving tree. Traits adapters must inline without changing evaluation order.
template <class Traits, unsigned L>
__global__ void rmsnorm_kernel(const typename Traits::Input* x, const typename Traits::Scale* w,
                              typename Traits::ScaleFormat wt, unsigned n, float eps,
                              typename Traits::Output* y) {
    static_assert(Traits::reduction_order == ReductionOrder::lane_strided_fma_lds_halving);
    static_assert(std::is_same_v<typename Traits::Input, float>);
    static_assert(std::is_same_v<typename Traits::Output, float>);
    __shared__ float s[L];
    const float* xr = x + static_cast<std::size_t>(blockIdx.x) * n;
    float* yr = y + static_cast<std::size_t>(blockIdx.x) * n;
    float a = 0.0F;
    for (unsigned i = threadIdx.x; i < n; i += L) a = __builtin_fmaf(xr[i], xr[i], a);
    const float ss = Traits::template reduce<L>(s, a);
    const float r = Traits::inv_sqrt(ss / static_cast<float>(n) + eps);
    for (unsigned i = threadIdx.x; i < n; i += L) yr[i] = w ? (xr[i] * r) * Traits::load(w, wt, i) : xr[i] * r;
}

}  // namespace strix::ops::norm
