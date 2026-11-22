#pragma once

#include "ops/head/screened_head.hpp"

#include <cuda_runtime.h>

namespace strix::cuda::core::head {
namespace screen = ::strix::ops::screened_head;

struct ScreenedBuffers {
    // Weights, packed plane and certificates must describe the same immutable
    // rows. The activation, workspace and result remain live through the launch.
    const std::uint16_t* weights;
    const float* activation;
    const std::uint8_t* packed;
    const screen::RowCertificate* certificates;
    float* activation_max;            // one float
    screen::ha::Pick2* tile_picks;    // plan.partial_count() disjoint partials
    screen::Result* out;              // one result
};

namespace screened_detail {
inline constexpr unsigned kWarp = 32;
inline constexpr unsigned kThreads = 256;
inline constexpr unsigned kWarps = kThreads / kWarp;
inline constexpr unsigned kMask = 0xffffffffU;

struct DeviceOutward {
    __device__ static double add_up(double a, double b) { return __dadd_ru(a, b); }
    __device__ static double add_down(double a, double b) { return __dadd_rd(a, b); }
    __device__ static double mul_up(double a, double b) { return __dmul_ru(a, b); }
    __device__ static double mul_down(double a, double b) { return __dmul_rd(a, b); }
};

__device__ inline float warp_max(float value) {
    for (int offset = 16; offset != 0; offset /= 2) {
        value = fmaxf(value, __shfl_xor_sync(kMask, value, offset));
    }
    return value;
}

static __global__ void activation_max(const float* activation, unsigned width, float* out) {
    __shared__ float warp_peaks[kWarps];
    float peak = 0;
    for (unsigned col = threadIdx.x; col < width; col += kThreads) {
        const float x = activation[col];
        peak = fmaxf(peak, isfinite(x) ? fabsf(x) : static_cast<float>(screen::kInfinity));
    }
    peak = warp_max(peak);
    if (threadIdx.x % kWarp == 0) warp_peaks[threadIdx.x / kWarp] = peak;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (const float warp_peak : warp_peaks) peak = fmaxf(peak, warp_peak);
        *out = peak;
    }
}

__device__ inline screen::Interval screen_row(const std::uint8_t* packed, const float* activation,
                                               unsigned width, unsigned lane) {
    // An integral signed nibble times FP32 fits exactly in binary64. Round
    // only the sums outwards, including the sums in the warp reduction.
    screen::Interval dot{0, 0};
    for (unsigned col = lane; col < width; col += kWarp) {
        const double product = static_cast<double>(screen::unpack(packed[col / 2], col)) * activation[col];
        dot.lower = DeviceOutward::add_down(dot.lower, product);
        dot.upper = DeviceOutward::add_up(dot.upper, product);
    }
    for (int offset = 16; offset != 0; offset /= 2) {
        dot.lower = DeviceOutward::add_down(dot.lower, __shfl_xor_sync(kMask, dot.lower, offset));
        dot.upper = DeviceOutward::add_up(dot.upper, __shfl_xor_sync(kMask, dot.upper, offset));
    }
    return dot;
}

template <class Exact>
__global__ void tiles(ScreenedBuffers buffers, unsigned width, unsigned vocab, unsigned first,
                      double relative, double underflow, Exact exact) {
    __shared__ screen::Interval ranges[screen::kTileRows];
    __shared__ double cutoff;
    __shared__ screen::ha::Pick2 picks[kWarps];
    const unsigned begin = blockIdx.x * screen::kTileRows;
    const unsigned count = min(vocab - begin, screen::kTileRows);
    const unsigned warp = threadIdx.x / kWarp;
    const unsigned lane = threadIdx.x % kWarp;
    const double x_max = *buffers.activation_max;
    for (unsigned local = warp; local < count; local += kWarps) {
        const unsigned row = begin + local;
        const auto cert = buffers.certificates[row];
        screen::Interval range;
        if (x_max != screen::kInfinity && cert.weight_l1 != screen::kInfinity) {
            const auto* packed = buffers.packed + std::size_t{row} * (width / 2U);
            const auto dot = screen_row(packed, buffers.activation, width, lane);
            if (lane == 0) range = screen::certify<DeviceOutward>(cert, x_max, relative, underflow, dot);
        }
        if (lane == 0) ranges[local] = range;
    }
    __syncthreads();
    if (threadIdx.x == 0) cutoff = screen::second_lower(ranges, count);
    __syncthreads();

    screen::ha::Pick2 pick;
    for (unsigned local = warp; local < count; local += kWarps) {
        if (!screen::survives(ranges[local], cutoff)) continue;
        const unsigned row = begin + local;
        const auto* weights = buffers.weights + std::size_t{row} * width;
        const float score = exact(weights, buffers.activation, width, lane);
        if (lane == 0) {
            if (isfinite(score)) screen::ha::push2(pick, score, first + row);
            else ++pick.first.nonfinite;
        }
    }
    if (lane == 0) picks[warp] = pick;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned i = 1; i < kWarps; ++i) pick = screen::ha::merge2(pick, picks[i]);
        buffers.tile_picks[blockIdx.x] = pick;
    }
}

static __global__ void finish(const screen::ha::Pick2* partials, unsigned count, screen::Result* out) {
    // One block reads compact top-two partials. All pruned logits were finite.
    __shared__ screen::ha::Pick2 picks[kThreads];
    screen::ha::Pick2 pick;
    for (unsigned i = threadIdx.x; i < count; i += kThreads) pick = screen::ha::merge2(pick, partials[i]);
    picks[threadIdx.x] = pick;
    for (unsigned step = kThreads / 2; step != 0; step /= 2) {
        __syncthreads();
        if (threadIdx.x < step) picks[threadIdx.x] = screen::ha::merge2(picks[threadIdx.x], picks[threadIdx.x + step]);
    }
    if (threadIdx.x == 0) *out = screen::emit(picks[0]);
}
}  // namespace screened_detail

// Exact is the incumbent raw BF16/FP32 row arithmetic bound by the caller.
// It is called by a full warp and lane zero supplies the score.
template <class Exact>
cudaError_t enqueue_screened(const screen::Plan& plan, ScreenedBuffers buffers, Exact exact, cudaStream_t stream) {
    namespace detail = screened_detail;
    if (!buffers.weights || !buffers.activation || !buffers.packed || !buffers.certificates ||
        !buffers.activation_max || !buffers.tile_picks || !buffers.out ||
        reinterpret_cast<std::uintptr_t>(buffers.weights) % 16 != 0 ||
        reinterpret_cast<std::uintptr_t>(buffers.activation) % 16 != 0) return cudaErrorInvalidValue;

    detail::activation_max<<<1, detail::kThreads, 0, stream>>>(buffers.activation, plan.width(), buffers.activation_max);
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
    detail::tiles<<<plan.tiles(), detail::kThreads, 0, stream>>>(
        buffers, plan.width(), plan.vocab(), plan.first(), plan.relative_error(), plan.underflow_error(), exact);
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
    detail::finish<<<1, detail::kThreads, 0, stream>>>(buffers.tile_picks, plan.tiles(), buffers.out);
    return cudaGetLastError();
}

}  // namespace strix::cuda::core::head
