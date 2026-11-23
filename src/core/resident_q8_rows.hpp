#pragma once

#include "core/bf16_lm_head.hpp"
#include "core/resident_row_plan.hpp"

#if defined(__HIPCC__)
#include "core/dense_linear.hpp"
#include "core/f16_tile_gemm_small.hpp"
#else
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/dense_big.hpp"
#endif

namespace strix::core {

#if defined(__HIPCC__)
namespace detail {
template <unsigned Bucket, weight_decode::Format Format = weight_decode::Format::q8_0>
void launch_q8_tight_rows(dense_linear::View weights, const float* input, float* output,
                         unsigned rows, row_transforms::Stream stream) {
    constexpr unsigned tile = Bucket <= 8U ? 2U : 4U;
    constexpr unsigned block_outputs = 4U * tile;
    dense_linear::kernel<Bucket, tile, Format, dense_linear::Epilogue::store_tight_rows>
        <<<(weights.outputs + block_outputs - 1U) / block_outputs, 128, 0, stream>>>(
            weights, weights, input, output, rows);
}
template <weight_decode::Format Format>
hipError_t launch_lowbit_rows(dense_linear::View weights, const float* input, float* output, unsigned rows,
                              row_transforms::Stream stream) {
    if (rows == 1U) launch_q8_tight_rows<1, Format>(weights, input, output, rows, stream);
    else if (rows <= 2U) launch_q8_tight_rows<2, Format>(weights, input, output, rows, stream);
    else if (rows <= 4U) launch_q8_tight_rows<4, Format>(weights, input, output, rows, stream);
    else if (rows <= 8U) launch_q8_tight_rows<8, Format>(weights, input, output, rows, stream);
    else launch_q8_tight_rows<16, Format>(weights, input, output, rows, stream);
    return hipGetLastError();
}
// Low-bit projections sharing one input, rows above the sweep: one launch of the planned WMMA route.
template <class Storage, unsigned Capacity>
hipError_t launch_lowbit_group(const float* input, unsigned rows, unsigned width,
                               const f16_tile_gemm::Q8ProjectionGroup<Capacity>& group, hipStream_t stream) {
    const auto plan = plan_lowbit_rows(rows, group.first_panel[group.count], width / Storage::group);
    if (plan.route == LowBitRoute::small_tile)
        return f16_tile_gemm::small_m::gemm_lowbit<Storage>(input, rows, width, group, plan.splits, stream);
    return f16_tile_gemm::gemm_lowbit_projections<Storage>(input, rows, width, group, stream);
}
}  // namespace detail
#endif

// Low-bit resident rows (q4_g64 / q6_g64), one launch of the real row count on the planned route (plan_lowbit_rows):
// the dense rows sweep, the small-M WMMA schedule or the 128-row projection tile. HIP only today; the CUDA dense family
// has no decoder for these storages.
inline row_transforms::Status enqueue_resident_lowbit_rows(
    const weight::ResidentMatrixView& matrix, unsigned live_outputs, const float* input,
    unsigned rows, float* output, row_transforms::Stream stream) {
    if (!valid_resident_lowbit_rows(matrix, live_outputs, input, rows, output)) return row_transforms::invalid;
#if defined(__HIPCC__)
    const bool q4 = matrix.storage == weight::kQ4G64WeightStorage;
    const auto width = static_cast<unsigned>(matrix.columns);
    const unsigned panels = (live_outputs + kLowBitPanelColumns - 1U) / kLowBitPanelColumns;
    if (plan_lowbit_rows(rows, panels, width / weight::Q4G64Storage::group).route != LowBitRoute::rows) {
        f16_tile_gemm::Q8ProjectionGroup<1> group{};
        group.append({matrix.bytes.data(), live_outputs, output});
        return q4 ? detail::launch_lowbit_group<weight::Q4G64Storage>(input, rows, width, group, stream)
                  : detail::launch_lowbit_group<weight::Q6G64Storage>(input, rows, width, group, stream);
    }
    const dense_linear::View weights{matrix.bytes.data(), width, live_outputs, static_cast<std::size_t>(matrix.row_bytes)};
    return q4
               ? detail::launch_lowbit_rows<weight_decode::Format::q4_g64>(weights, input, output, rows, stream)
               : detail::launch_lowbit_rows<weight_decode::Format::q6_g64>(weights, input, output, rows, stream);
#else
    (void)stream;
    return row_transforms::invalid;
#endif
}

// Bind compact resident weights to the native dense family. Only live output
// rows are projected. The caller owns the activation/output BF16 boundaries.
// CUDA callers can supply a reserved scratch context for aligned tall shapes.
inline row_transforms::Status enqueue_resident_q8_rows(
    const weight::ResidentMatrixView& matrix, unsigned live_outputs, const float* input,
    unsigned rows, float* output, row_transforms::Stream stream
#if !defined(__HIPCC__)
    , cuda_core::BigCtx* context = nullptr
#endif
) {
    if (matrix.storage == weight::kQ4G64WeightStorage || matrix.storage == weight::kQ6G64WeightStorage)
        return enqueue_resident_lowbit_rows(matrix, live_outputs, input, rows, output, stream);
    if (!valid_resident_q8_rows(matrix, live_outputs, input, rows, output)) return row_transforms::invalid;

    const auto width = static_cast<unsigned>(matrix.columns);
#if defined(__HIPCC__)
    if (use_resident_q8_tile(rows)) {
        f16_tile_gemm::Q8ProjectionGroup<1> group{};
        group.append({matrix.bytes.data(), live_outputs, output});
        return f16_tile_gemm::gemm_q8_projections(input, rows, width, group, stream);
    }
    const dense_linear::View weights{matrix.bytes.data(), width, live_outputs, static_cast<std::size_t>(matrix.row_bytes)};
    return for_each_dense_chunk<gemv_rows::max_rows>(rows, hipSuccess, [&](unsigned first, unsigned count) {
        const auto* x = input + std::size_t{first} * width;
        auto* y = output + std::size_t{first} * live_outputs;
        if (count == 1U) detail::launch_q8_tight_rows<1>(weights, x, y, count, stream);
        else if (count <= 2U) detail::launch_q8_tight_rows<2>(weights, x, y, count, stream);
        else if (count <= 4U) detail::launch_q8_tight_rows<4>(weights, x, y, count, stream);
        else if (count <= 8U) detail::launch_q8_tight_rows<8>(weights, x, y, count, stream);
        else detail::launch_q8_tight_rows<16>(weights, x, y, count, stream);
        return hipGetLastError();
    });
#else
    const cuda_core::DenseWeight weights{matrix.bytes.data(), cuda_core::WeightFormat::q8_0, live_outputs, width};
    if (context && rows > cuda_core::kMaxRows && cuda_core::dense_big_supported(weights)) {
        cuda_core::BigAct activation{};
        const auto status = cuda_core::dense_big_split(input, rows, width, weights.format, 3, &activation, stream, context);
        if (status != cudaSuccess) return status;
        return cuda_core::dense_big_gemm(weights, activation, output, nullptr, cuda_core::BigEpi::none, stream, context);
    }
    return for_each_dense_chunk<cuda_core::kMaxRows>(rows, cudaSuccess, [&](unsigned first, unsigned count) {
        return cuda_core::dense_rows(weights, input + std::size_t{first} * width,
            output + std::size_t{first} * live_outputs, count, nullptr, stream, cuda_core::DenseArith::mma3);
    });
#endif
}

// Resident linear projections have a common input and no stored output padding.
// The standalone matrix entry also supports projecting a live prefix (e.g. the LM head).
template <unsigned Capacity>
inline row_transforms::Status enqueue_resident_q8_projections(
    const float* input, unsigned width, unsigned rows, std::span<const ResidentRowProjection> projections,
    row_transforms::Stream stream
#if !defined(__HIPCC__)
    , cuda_core::BigCtx* context = nullptr
#endif
) {
    if (projections.empty() || projections.size() > Capacity) return row_transforms::invalid;
#if defined(__HIPCC__)
    constexpr auto success = hipSuccess;
    // Projections of one low-bit storage above the sweep: one launch over all their panels (a mix of storages runs
    // each projection on its own route below).
    const auto* first = std::get_if<weight::ResidentMatrixView>(&projections.front().weights);
    bool lowbit = first && (first->storage == weight::kQ4G64WeightStorage || first->storage == weight::kQ6G64WeightStorage);
    for (const auto& projection : projections) {
        const auto* matrix = std::get_if<weight::ResidentMatrixView>(&projection.weights);
        lowbit = lowbit && matrix && matrix->storage == first->storage;
    }
    unsigned panels = 0U;
    for (const auto& projection : projections) panels += (projection.outputs + kLowBitPanelColumns - 1U) / kLowBitPanelColumns;
    if (lowbit && plan_lowbit_rows(rows, panels, width / weight::Q4G64Storage::group).route != LowBitRoute::rows) {
        f16_tile_gemm::Q8ProjectionGroup<Capacity> group{};
        for (const auto& projection : projections) {
            const auto* matrix = std::get_if<weight::ResidentMatrixView>(&projection.weights);
            if (!matrix || matrix->columns != width ||
                matrix->rows != projection.outputs ||
                !valid_resident_lowbit_rows(*matrix, projection.outputs, input, rows, projection.output))
                return row_transforms::invalid;
            group.append({matrix->bytes.data(), projection.outputs, projection.output});
        }
        return first->storage == weight::kQ4G64WeightStorage
                   ? detail::launch_lowbit_group<weight::Q4G64Storage>(input, rows, width, group, stream)
                   : detail::launch_lowbit_group<weight::Q6G64Storage>(input, rows, width, group, stream);
    }
    if (use_resident_q8_tile(rows)) {
        f16_tile_gemm::Q8ProjectionGroup<Capacity> group{};
        for (const auto& projection : projections) {
            const auto* matrix = std::get_if<weight::ResidentMatrixView>(&projection.weights);
            if (!matrix || matrix->columns != width || matrix->rows != projection.outputs ||
                !valid_resident_q8_rows(*matrix, projection.outputs, input, rows, projection.output))
                return row_transforms::invalid;
            group.append({matrix->bytes.data(), projection.outputs, projection.output});
        }
        return f16_tile_gemm::gemm_q8_projections(input, rows, width, group, stream);
    }
#else
    constexpr auto success = cudaSuccess;
#endif
    for (const auto& projection : projections) {
        const auto* matrix = std::get_if<weight::ResidentMatrixView>(&projection.weights);
        if (!matrix || matrix->columns != width || matrix->rows != projection.outputs) return row_transforms::invalid;
        const auto status = enqueue_resident_q8_rows(*matrix, projection.outputs, input, rows, projection.output, stream
#if !defined(__HIPCC__)
                                                     , context
#endif
        );
        if (status != success) return status;
    }
    return success;
}

inline row_transforms::Status enqueue_resident_q8_softcap_head(
    const weight::ResidentMatrixView& matrix, unsigned live_outputs, const float* input,
    unsigned rows, std::span<float> logits, float cap, row_transforms::Stream stream) {
    if (live_outputs == 0 || rows > logits.size() / live_outputs || !(cap > 0.0F))
        return row_transforms::invalid;
    const auto status = enqueue_resident_q8_rows(matrix, live_outputs, input, rows, logits.data(), stream);
#if defined(__HIPCC__)
    if (status != hipSuccess) return status;
#else
    if (status != cudaSuccess) return status;
#endif
    return enqueue_bf16_head_softcap(rows, live_outputs, logits.data(), cap, stream);
}

}  // namespace strix::core
