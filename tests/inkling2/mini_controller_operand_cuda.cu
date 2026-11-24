// The controller's declared operand class on the tall CUDA dense route (cuda_core::enqueue_bf16_dense_rows, mma3):
// down (K = 2 x 2304 RMS-normalised BF16 rows -> rank 64) and the forget/gain pair (K = 64 BF16 latent -> 2304), one
// plane (bf16_exact) against three planes (fp32 operand) at 2048 rows. Finite rows and finite weights: same bits.
// An infinite forget weight: the two classes differ (one plane inf, three planes NaN from 0 x inf), and the load-time
// guard (core::finite_magnitude_max + core::low_rank_operands_bounded) refuses the declared class for it.
#include "core/dense_row_plan.hpp"
#include "cuda/core/dense/bf16_rows.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

namespace {

namespace cc = strix::cuda_core;
namespace core = strix::core;

constexpr unsigned kWidth = 2304, kRank = 64, kRows = 2048;

void require(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::cerr << cudaGetErrorString(status) << '\n';
    std::exit(1);
}

template <class T>
struct Buffer {
    explicit Buffer(std::size_t count) : size(count) { require(cudaMallocManaged(&data, count * sizeof(T))); }
    ~Buffer() { (void)cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* data = nullptr;
    std::size_t size;
};

std::uint32_t seed = 0x5EED1238U;
std::uint32_t next() { return seed = 1664525U * seed + 1013904223U; }

float bf16_round(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    bits = (bits + 0x7FFFU + ((bits >> 16U) & 1U)) & 0xFFFF0000U;  // nearest even, finite inputs
    std::memcpy(&value, &bits, sizeof bits);
    return value;
}

// RMS-normalised BF16 rows (the controller's down operands) or small BF16 latent values.
void fill_rows(float* rows, std::size_t count, unsigned width, bool normalise) {
    std::vector<float> row(width);
    for (std::size_t r = 0; r < count; ++r) {
        double squares = 0.0;
        for (auto& x : row) {
            x = static_cast<float>(static_cast<int>(next() >> 8U) - 8388608) * 0x1p-21F;
            squares += double{x} * x;
        }
        const float scale = normalise ? static_cast<float>(1.0 / std::sqrt(squares / width + 1e-6)) : 1.0F;
        for (unsigned i = 0; i < width; ++i) rows[r * width + i] = bf16_round(row[i] * scale);
    }
}

void fill_weights(std::uint16_t* weights, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        const auto bits = next();
        weights[i] = static_cast<std::uint16_t>(0x3C00U | ((bits >> 16U) & 0x8000U) | ((bits >> 12U) & 0x01FFU));
    }
}

std::vector<float> widen(const std::uint16_t* weights, std::size_t count) {
    std::vector<float> out(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint32_t bits = std::uint32_t{weights[i]} << 16U;
        std::memcpy(&out[i], &bits, sizeof bits);
    }
    return out;
}

// One projection group on both operand classes; returns true when the outputs are bitwise equal.
bool same_bits(const float* input, unsigned width, std::span<const core::Bf16RowProjection> one,
               std::span<const core::Bf16RowProjection> three, std::size_t elements, cc::BigCtx* context) {
    require(cc::enqueue_bf16_dense_rows(input, width, kRows, one, nullptr, cc::DenseArith::mma3, context, nullptr,
                                        core::Bf16Operand::bf16_exact));
    require(cc::enqueue_bf16_dense_rows(input, width, kRows, three, nullptr, cc::DenseArith::mma3, context, nullptr,
                                        core::Bf16Operand::fp32));
    require(cudaDeviceSynchronize());
    for (std::size_t p = 0; p < one.size(); ++p)
        if (std::memcmp(one[p].output, three[p].output, elements * sizeof(float))) return false;
    return true;
}

}  // namespace

int main() {
    std::unique_ptr<cc::BigCtx, decltype(&cc::dense_big_ctx_destroy)> ctx(cc::dense_big_ctx_create(),
                                                                        cc::dense_big_ctx_destroy);
    if (!ctx) return 1;
    const unsigned down_k = 2U * kWidth;
    Buffer<float> operands(std::size_t{kRows} * down_k), latent(std::size_t{kRows} * kRank);
    fill_rows(operands.data, std::size_t{kRows} * 2U, kWidth, true);  // raw and message halves, each normalised
    fill_rows(latent.data, kRows, kRank, false);
    Buffer<std::uint16_t> down(std::size_t{kRank} * down_k), forget(std::size_t{kWidth} * kRank),
        gain(std::size_t{kWidth} * kRank);
    fill_weights(down.data, down.size);
    fill_weights(forget.data, forget.size);
    fill_weights(gain.data, gain.size);
    Buffer<float> down_one(std::size_t{kRows} * kRank), down_three(std::size_t{kRows} * kRank);
    Buffer<float> f_one(std::size_t{kRows} * kWidth), f_three(std::size_t{kRows} * kWidth);
    Buffer<float> g_one(std::size_t{kRows} * kWidth), g_three(std::size_t{kRows} * kWidth);

    const core::Bf16RowProjection d1[] = {{down.data, kRank, down_one.data, true}};
    const core::Bf16RowProjection d3[] = {{down.data, kRank, down_three.data, true}};
    const bool down_equal = same_bits(operands.data, down_k, d1, d3, std::size_t{kRows} * kRank, ctx.get());
    const core::Bf16RowProjection p1[] = {{forget.data, kWidth, f_one.data, false}, {gain.data, kWidth, g_one.data, false}};
    const core::Bf16RowProjection p3[] = {{forget.data, kWidth, f_three.data, false}, {gain.data, kWidth, g_three.data, false}};
    const bool gates_equal = same_bits(latent.data, kRank, p1, p3, std::size_t{kRows} * kWidth, ctx.get());
    const float down_max = core::finite_magnitude_max(widen(down.data, down.size));
    const float gate_max = std::max(core::finite_magnitude_max(widen(forget.data, forget.size)),
                                    core::finite_magnitude_max(widen(gain.data, gain.size)));
    const bool admitted = core::low_rank_operands_bounded(kWidth, 2U, kRank, down_max, gate_max);

    // An infinite forget weight hit by a positive latent: one plane gives inf, three planes NaN (0 x inf on the
    // zero planes). The guard sees the non-finite weight and keeps the FP32 class.
    forget.data[0] = 0x7F80U;
    for (unsigned r = 0; r < kRows; ++r) latent.data[std::size_t{r} * kRank] = 1.0F;
    const bool inf_differs = !same_bits(latent.data, kRank, p1, p3, std::size_t{kRows} * kWidth, ctx.get());
    const bool refused = !core::low_rank_operands_bounded(
        kWidth, 2U, kRank, down_max, core::finite_magnitude_max(widen(forget.data, forget.size)));

    std::cout << (down_equal ? "PASS" : "FAIL") << " down 2048 x 4608 -> 64: one plane == three planes\n"
              << (gates_equal ? "PASS" : "FAIL") << " forget/gain 2048 x 64 -> 2304: one plane == three planes\n"
              << (admitted ? "PASS" : "FAIL") << " finite weights admitted by the bound\n"
              << (inf_differs ? "INFO" : "INFO (no difference on this route)")
              << " infinite forget weight: one plane vs three planes differ (why the guard exists)\n"
              << (refused ? "PASS" : "FAIL") << " infinite forget weight refused by the guard\n";
    return down_equal && gates_equal && admitted && refused ? 0 : 1;
}
