#pragma once

namespace strix::ops::norm {

enum class ReductionOrder { lane_strided_fma_lds_halving };

// Contiguous [rows][n] RMSNorm, shared scale[n] (or null for unit scale),
// no persistent state. Input/output may alias. The engine owns launch/error
// handling, scale decoding and the declared reciprocal-square-root operation.
// Epsilon is added AFTER FP32 sumsq/n; output is (x*r)*scale, never x*(r*scale).
// Only the reduction below is implemented; other orders need a separate body.
template <class Traits>
struct Norm {
    using Input = typename Traits::Input;
    using Output = typename Traits::Output;
    using Scale = typename Traits::Scale;
    using ScaleFormat = typename Traits::ScaleFormat;
    using Stream = typename Traits::Stream;
    static constexpr float epsilon = Traits::epsilon;
    static constexpr ReductionOrder reduction_order = Traits::reduction_order;

    static constexpr unsigned lanes(unsigned n) { return Traits::lanes(n); }

    // Explicit epsilon retains callers' existing arithmetic identity. Geometry,
    // pointer validity and stream lifetime remain the engine's responsibility.
    static void run(const Input* x, const Scale* w, ScaleFormat wt, unsigned n,
                    float eps, Output* y, unsigned rows, Stream stream) {
        Traits::launch(x, w, wt, n, eps, y, rows, stream);
    }
};

}  // namespace strix::ops::norm
