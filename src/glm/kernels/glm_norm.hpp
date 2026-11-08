#pragma once

#include "glm_tier_a.hpp"
#include "../../ops/norm/norm.hpp"

namespace strix::glm::dev {

// GLM R2/R3 identity: FP32 I/O, F16/BF16/F32 direct scales, 128 lanes
// below width 512, otherwise 256. inv_sqrt is GLM's declared sqrt + divide,
// not a hardware approximate rsqrt. All engine call sites use kEps and BF16.
struct GlmNormTraits {
    using Input = float;
    using Output = float;
    using Scale = void;
    using ScaleFormat = WType;
    using Stream = hipStream_t;
    static constexpr float epsilon = kEps;
    static constexpr auto reduction_order = ops::norm::ReductionOrder::lane_strided_fma_lds_halving;
    static constexpr unsigned lanes(unsigned n) { return norm_lanes(n); }
    static void launch(const float* x, const void* w, WType wt, unsigned n, float eps,
                       float* y, unsigned rows, hipStream_t stream) {
        rmsnorm(x, w, wt, n, eps, y, rows, stream);
    }
};

using GlmNorm = ops::norm::Norm<GlmNormTraits>;

}  // namespace strix::glm::dev
