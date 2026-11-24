#pragma once

#include "core/dense_linear.hpp"
#include "core/dense_row_plan.hpp"
#include "core/f16_tile_gemm.hpp"
#include "core/fma_rows.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <algorithm>
#include <limits>

namespace strix::core::fma_rows {

// Small NK projections use the decode sweep. Wide projections stage RNE BF16
// activations and resident BF16 weights in the shared WMMA tile.
// A round_bf16 projection group of at most gemv_rows::max_rows rows runs the dense_linear tight-row body, whose
// store_tight_rows_bf16 epilogue rounds each completed FP32 sum: the value a later Bf16Round pass over the stored sum
// writes (a NaN stays NaN; only its low payload bits may differ). Larger groups keep their separate round.
inline constexpr bool rounds_in_store(unsigned rows) noexcept { return rows <= gemv_rows::max_rows; }

inline hipError_t enqueue_bf16_rows(const float* input, unsigned input_width, unsigned rows,
                                    std::span<const Bf16RowProjection> projections, unsigned max_outputs,
                                    hipStream_t stream) {
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!input || rows > std::numeric_limits<unsigned>::max() - 128U || !admits(input_width, rows, 16U) ||
        input_width > max_elements / rows || reinterpret_cast<std::uintptr_t>(input) % 16U != 0 ||
        projections.empty() || projections.size() > kGroupMax) return hipErrorInvalidValue;

    const bool round_bf16 = projections.front().round_bf16;
    const auto natural_row_bytes = std::size_t{input_width} * sizeof(std::uint16_t);
    const auto effective_pitch = [natural_row_bytes](const Bf16RowProjection& projection) {
        return projection.row_bytes != 0U ? projection.row_bytes : natural_row_bytes;
    };
    Group group{};
    group.count = static_cast<unsigned>(projections.size());
    for (unsigned p = 0; p < group.count; ++p) {
        const auto& projection = projections[p];
        const auto pitch = effective_pitch(projection);
        if (projection.round_bf16 != round_bf16 || !projection.weights || !projection.output || projection.outputs == 0 || projection.outputs > max_outputs ||
            projection.outputs > max_elements / rows ||
            projection.outputs > std::numeric_limits<unsigned>::max() - 128U ||
            pitch < natural_row_bytes || pitch % 16U != 0U ||
            projection.outputs > std::numeric_limits<std::size_t>::max() / pitch)
            return hipErrorInvalidValue;
    }

    if (rows <= gemv_rows::max_rows) {
        // Same-input projections share one launch; each output keeps its single-launch instructions (bitwise).
        static_assert(kGroupMax <= dense_linear::kTightGroupMax);
        dense_linear::View views[dense_linear::kTightGroupMax]{};
        float* outputs[dense_linear::kTightGroupMax]{};
        for (unsigned p = 0; p < group.count; ++p) {
            views[p] = {reinterpret_cast<const std::uint8_t*>(projections[p].weights), input_width,
                        projections[p].outputs, effective_pitch(projections[p])};
            outputs[p] = projections[p].output;
        }
        if (group.count == 1U)
            return round_bf16 ? dense_linear::enqueue_bf16_tight_rows<true>(views[0], input, outputs[0], rows, stream)
                              : dense_linear::enqueue_bf16_tight_rows(views[0], input, outputs[0], rows, stream);
        return round_bf16
            ? dense_linear::enqueue_bf16_tight_rows_group<true>(views, outputs, group.count, input, rows, stream)
            : dense_linear::enqueue_bf16_tight_rows_group(views, outputs, group.count, input, rows, stream);
    }

    const auto aligned_weights = [](const Bf16RowProjection& projection) {
        return reinterpret_cast<std::uintptr_t>(projection.weights) % 16U == 0U;
    };
    if (rows <= 65535U * f16_tile_gemm::kBM &&
        input_width <= std::numeric_limits<unsigned>::max() - f16_tile_gemm::kBK &&
        std::ranges::all_of(projections, aligned_weights)) {
        f16_tile_gemm::Bf16ProjectionGroup<kGroupMax> wmma_group{};
        const auto pitch = effective_pitch(projections.front());
        for (const auto& projection : projections) {
            if (effective_pitch(projection) != pitch) return hipErrorInvalidValue;
            wmma_group.append({projection.weights, projection.outputs, projection.output, projection.round_bf16});
        }
        if (pitch == natural_row_bytes)
            f16_tile_gemm::gemm_bf16_projections(input, rows, input_width, wmma_group, stream);
        else
            f16_tile_gemm::gemm_bf16_projections(input, rows, input_width, wmma_group, stream,
                                                 pitch / sizeof(std::uint16_t));
        return hipGetLastError();
    }

    // Preserve existing admission for callers whose weights cannot use vector
    // loads or whose row grid exceeds the WMMA launch limit.
    for (unsigned p = 0; p < group.count; ++p) {
        if (!projections[p].contiguous(input_width)) return hipErrorInvalidValue;
        group.w[p] = projections[p].weights;
        group.y[p] = projections[p].output;
        group.n[p] = projections[p].outputs;
    }
    if (round_bf16) {
        if (admits(input_width, rows, 32U)) launch_auto<true>(input, Weight::BF16, input_width, rows, group, stream);
        else launch<128, 64, 1, 16, true, true>(input, Weight::BF16, input_width, rows, group, stream);
    } else if (admits(input_width, rows, 32U)) {
        launch_auto(input, Weight::BF16, input_width, rows, group, stream);
    } else {
        launch<128, 64, 1, 16, true>(input, Weight::BF16, input_width, rows, group, stream);
    }
    return hipGetLastError();
}

// Rows whose activations arrive as a BF16 operand image (u16 [rows][input_width], element = the RNE BF16 encoding of
// the FP32 value): the shared WMMA tile reads the image as is, the operand bits its FP32 staging makes of those values,
// so each output keeps the bits enqueue_bf16_rows stores for the FP32 rows; half the activation bytes. The tile's
// prefill shapes only (decode rows keep their FP32 sweep): callers check admits_bf16_image_projections before they
// store the image.
inline constexpr bool admits_bf16_image_rows(unsigned input_width, unsigned rows) noexcept {
    return rows > gemv_rows::max_rows && rows <= 65535U * f16_tile_gemm::kBM && input_width >= f16_tile_gemm::kBK &&
           input_width % f16_tile_gemm::kBK == 0U &&
           input_width <= std::numeric_limits<unsigned>::max() - f16_tile_gemm::kBK;
}

// The image route for these projections, asked before the image is stored: the shape (admits_bf16_image_rows) and
// every weight the image launch reads (contiguous rows, 16-byte aligned, outputs within max_outputs, one rounding
// class). A projection outside it keeps its FP32 rows, where enqueue_bf16_rows serves pitched and unaligned weights.
inline bool admits_bf16_image_projections(unsigned input_width, unsigned rows,
                                          std::span<const Bf16RowProjection> projections,
                                          unsigned max_outputs) noexcept {
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!admits_bf16_image_rows(input_width, rows) || input_width > max_elements / rows || projections.empty() ||
        projections.size() > kGroupMax)
        return false;
    const bool round_bf16 = projections.front().round_bf16;
    return std::ranges::all_of(projections, [&](const Bf16RowProjection& projection) {
        return projection.weights && projection.output && projection.outputs != 0U &&
               projection.outputs <= max_outputs && projection.outputs <= max_elements / rows &&
               projection.contiguous(input_width) &&
               reinterpret_cast<std::uintptr_t>(projection.weights) % 16U == 0U && projection.round_bf16 == round_bf16;
    });
}

// Image rows from this height up take the K64 packed schedule where the shape admits it (measured faster than the K32
// image tile at 4,096 and 8,192 rows on every wide shape; lower heights keep the K32 image tile).
inline constexpr unsigned kPackedImageMinRows = 1024U;

inline hipError_t enqueue_bf16_image_rows(const std::uint16_t* input, unsigned input_width, unsigned rows,
                                          std::span<const Bf16RowProjection> projections, unsigned max_outputs,
                                          hipStream_t stream) {
    if (!input || reinterpret_cast<std::uintptr_t>(input) % 16U != 0U ||
        !admits_bf16_image_projections(input_width, rows, projections, max_outputs))
        return hipErrorInvalidValue;
    f16_tile_gemm::Bf16ProjectionGroup<kGroupMax> group{};
    for (const auto& projection : projections)
        group.append({projection.weights, projection.outputs, projection.output, projection.round_bf16});
    // Prefill heights on whole 128-column projections take the packed-operand core's K64 schedule (same bits).
    if (rows >= kPackedImageMinRows && f16_tile_gemm::admits_packed_image_projections(input_width, group))
        return f16_tile_gemm::gemm_bf16_projections_image_packed(input, rows, input_width, group, stream);
    f16_tile_gemm::gemm_bf16_projections_image(input, rows, input_width, group, stream);
    return hipGetLastError();
}

namespace detail {
// One thread per 8 values: image[i] = bf16_encode(input[i]) (RNE, a NaN stays NaN), the operand bits the projection
// tile's FP32 staging (pack2<bf16>) makes of the same values.
static __global__ __launch_bounds__(256) void bf16_operand_image_kernel(const float* input, std::uint16_t* image,
                                                                 std::size_t groups) {
    const std::size_t g = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const auto* in = reinterpret_cast<const float4*>(input) + 2U * g;
    const float4 x = in[0], y = in[1];
    const auto pack = [](float lo, float hi) {
        return std::uint32_t{::strix::weight::scalar_float::bf16_encode(lo)} |
               (std::uint32_t{::strix::weight::scalar_float::bf16_encode(hi)} << 16U);
    };
    reinterpret_cast<uint4*>(image)[g] = make_uint4(pack(x.x, x.y), pack(x.z, x.w), pack(y.x, y.y), pack(y.z, y.w));
}
}  // namespace detail

// The BF16 operand image of contiguous FP32 rows ([rows][width] -> [rows][width] u16) for enqueue_bf16_image_rows,
// which then stores the bits enqueue_bf16_rows stores on the FP32 rows. Both pointers 16-byte aligned, width % 8 == 0,
// the two byte ranges disjoint (blocks run in no order: an image over its own input would depend on the schedule) and
// the grid within the launch limit; anything else is refused before a launch.
inline hipError_t enqueue_bf16_operand_image(const float* input, unsigned width, unsigned rows, std::uint16_t* image,
                                             hipStream_t stream) {
    if (!input || !image || width % 8U != 0U || reinterpret_cast<std::uintptr_t>(input) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(image) % 16U != 0U)
        return hipErrorInvalidValue;
    const std::uint64_t values = std::uint64_t{rows} * width;  // both factors < 2^32: no wrap
    const std::uint64_t groups = values / 8U;
    if (groups == 0U) return hipSuccess;
    const std::uint64_t blocks = (groups + 255U) / 256U;
    if (blocks > std::uint64_t{std::numeric_limits<int>::max()} ||
        values > std::uint64_t{std::numeric_limits<std::uintptr_t>::max() / 4U})
        return hipErrorInvalidValue;
    const auto in_lo = reinterpret_cast<std::uintptr_t>(input), out_lo = reinterpret_cast<std::uintptr_t>(image);
    const auto in_bytes = static_cast<std::uintptr_t>(values) * 4U, out_bytes = static_cast<std::uintptr_t>(values) * 2U;
    if (in_lo < out_lo + out_bytes && out_lo < in_lo + in_bytes) return hipErrorInvalidValue;
    detail::bf16_operand_image_kernel<<<static_cast<unsigned>(blocks), 256, 0, stream>>>(input, image,
                                                                                       static_cast<std::size_t>(groups));
    return hipGetLastError();
}

// The image routes of the shared tile on gfx11 (backend data, measured at 8,192 rows, K 2,304: FP32 route 0.150 ms per
// 128-column tile, packed K64 0.109 ms, image pass 0.48 ms -> the image pays from ~12 packed tiles in one call; 16 kept).
inline constexpr ImageRoutePolicy kImageRoutePolicy{kPackedImageMinRows, f16_tile_gemm::ImageProjectionSchedule::kStageK,
                                                    f16_tile_gemm::kBN, 16U * f16_tile_gemm::kBN};
inline constexpr std::size_t kImageRouteMaxProjections = 8U;

// enqueue_bf16_rows for every projection of one call on the same FP32 rows (any count, consecutive runs of kGroupMax),
// where a call at prefill height with enough whole-tile columns stores the rows' BF16 operand image once (in image, at
// least rows x input_width u16, 16-byte aligned) and runs the packed K64 schedule (pitched weights included), the K32
// image tile for the narrow contiguous rest and the FP32 rows for what neither reads. Every output keeps the bits
// enqueue_bf16_rows stores; a call the plan keeps on its rows makes the launches enqueue_bf16_rows makes.
inline hipError_t enqueue_bf16_rows_routed(const float* input, unsigned input_width, unsigned rows,
                                           std::span<const Bf16RowProjection> projections, unsigned max_outputs,
                                           std::span<std::uint16_t> image, hipStream_t stream) {
    const auto fp32_rows = [&]() {
        return for_each_bf16_projection_group<kGroupMax>(projections, hipSuccess, [&](auto group) {
            return enqueue_bf16_rows(input, input_width, rows, group, max_outputs, stream);
        });
    };
    ImageRoute routes[kImageRouteMaxProjections]{};
    if (!input || !image.data() || projections.size() > kImageRouteMaxProjections || input_width == 0U ||
        input_width % 8U != 0U ||
        image.size() / input_width < rows || reinterpret_cast<std::uintptr_t>(input) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(image.data()) % 16U != 0U ||
        !plan_image_routes(kImageRoutePolicy, input_width, rows, projections, max_outputs, routes,
                           [&](const Bf16RowProjection& projection) {
                               return admits_bf16_image_projections(input_width, rows, {&projection, 1U}, max_outputs);
                           }))
        return fp32_rows();
    if (const auto status = enqueue_bf16_operand_image(input, input_width, rows, image.data(), stream);
        status != hipSuccess)
        return status;
    const auto natural = std::size_t{input_width} * sizeof(std::uint16_t);
    return for_each_image_route_run<kGroupMax>(projections, std::span<const ImageRoute>(routes, projections.size()),
        input_width, hipSuccess, [&](ImageRoute route, std::span<const Bf16RowProjection> run) {
            if (route == ImageRoute::fp32_rows)
                return enqueue_bf16_rows(input, input_width, rows, run, max_outputs, stream);
            if (route == ImageRoute::image)
                return enqueue_bf16_image_rows(image.data(), input_width, rows, run, max_outputs, stream);
            f16_tile_gemm::Bf16ProjectionGroup<kGroupMax> group{};
            for (const auto& projection : run)
                group.append({projection.weights, projection.outputs, projection.output, projection.round_bf16});
            const auto pitch = run.front().row_bytes != 0U ? run.front().row_bytes : natural;
            return f16_tile_gemm::gemm_bf16_projections_image_packed(image.data(), rows, input_width, group, stream,
                                                                     pitch / sizeof(std::uint16_t));
        });
}

// The [forget, gain] pair of a low-rank controller with the controller as the tile epilogue:
// output[r][n] = controller_mix(current[r][n], message[r][n], forget sum, gain sum), the same bits as
// enqueue_bf16_rows (FP32 sums) followed by the pointwise controller. hipErrorNotSupported where the shared WMMA tile
// does not take the pair (decode rows, unequal widths, rounded sums, unaligned operands): the caller runs the two-step
// form instead.
inline hipError_t enqueue_bf16_controller_rows(const float* input, unsigned input_width, unsigned rows,
                                               std::span<const Bf16RowProjection> gates, const float* current,
                                               const float* message, float* output, hipStream_t stream) {
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    for (const auto& gate : gates)
        if (!gate.contiguous(input_width)) return hipErrorInvalidValue;
    if (gates.size() != 2U || rows <= gemv_rows::max_rows) return hipErrorNotSupported;
    const auto& forget = gates[0];
    const auto& gain = gates[1];
    const auto aligned = [](const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0U; };
    if (forget.outputs != gain.outputs || forget.round_bf16 || gain.round_bf16 || forget.outputs == 0U ||
        rows > 65535U * f16_tile_gemm::kBM || input_width == 0U || input_width % 16U != 0U ||
        input_width > std::numeric_limits<unsigned>::max() - f16_tile_gemm::kBK || !aligned(input) ||
        !aligned(forget.weights) || !aligned(gain.weights) || forget.outputs > max_elements / rows)
        return hipErrorNotSupported;
    if (!current || !message || !output || !forget.weights || !gain.weights) return hipErrorInvalidValue;
    f16_tile_gemm::Bf16ProjectionGroup<2U> group{};
    group.append({forget.weights, forget.outputs, forget.output, forget.round_bf16});
    group.append({gain.weights, gain.outputs, gain.output, gain.round_bf16});
    f16_tile_gemm::gemm_bf16_controller(input, rows, input_width, group, {current, message, output, forget.outputs}, stream);
    return hipGetLastError();
}

}  // namespace strix::core::fma_rows
