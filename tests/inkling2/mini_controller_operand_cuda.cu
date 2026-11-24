// The controller's declared operand class on the tall CUDA dense dispatch the resident backend uses
// (inkling2::enqueue_controller_dense, mma3, 2048 rows): down (K = 2 x 2304 RMS-normalised BF16 rows -> rank 64) and
// the forget/gain pair (K = 64 BF16 latent -> 2304). The operand class that runs comes from the production
// certificate (MiniControllerCertificate, filled from the staged weight bytes exactly as
// MiniResidentWeights::certify_controllers fills it), and the outputs are compared bit for bit with a three-plane
// (fp32 operand) baseline through the same kernels: ordinary weights run one plane, an infinite forget weight and
// oversized finite down weights (latent overflow) run three planes. The latent of the gate stage is the output of
// the dispatched down stage.
#include "inkling2/inkling2_mini_controller_operands_cuda.hpp"

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

namespace inkling = strix::inkling2;

// One projection group through the production dispatch (declared one plane) and the three-plane baseline.
struct Pair {
    bool equal;
    core::Bf16Operand route;
};

Pair run_pair(const inkling::MiniControllerCertificate& certificate, const float* input, unsigned width,
              std::span<const core::Bf16RowProjection> dispatched, std::span<const core::Bf16RowProjection> baseline,
              std::size_t elements, cc::BigCtx* context) {
    require(inkling::enqueue_controller_dense(certificate, input, width, kRows, dispatched, nullptr, context,
                                              core::Bf16Operand::bf16_exact));
    require(cc::enqueue_bf16_dense_rows(input, width, kRows, baseline, nullptr, cc::DenseArith::mma3, context, nullptr,
                                        core::Bf16Operand::fp32));
    require(cudaDeviceSynchronize());
    Pair result{true, certificate.route(core::Bf16Operand::bf16_exact)};
    for (std::size_t p = 0; p < dispatched.size(); ++p)
        if (std::memcmp(dispatched[p].output, baseline[p].output, elements * sizeof(float))) result.equal = false;
    return result;
}

std::span<const std::uint16_t> bits(const Buffer<std::uint16_t>& weights) { return {weights.data, weights.size}; }

struct Outcome {
    Pair down, gates;
};

// The chain on the given weights: down on the normalised operands, then the gates on the dispatched down latent.
Outcome run_chain(Buffer<std::uint16_t>& down, Buffer<std::uint16_t>& forget, Buffer<std::uint16_t>& gain,
                  const float* operands, cc::BigCtx* context) {
    inkling::MiniControllerCertificate certificate;
    certificate.add_down(bits(down));
    certificate.add_gate(bits(forget));
    certificate.add_gate(bits(gain));
    Buffer<float> latent(std::size_t{kRows} * kRank), latent_baseline(std::size_t{kRows} * kRank);
    const core::Bf16RowProjection d1[] = {{down.data, kRank, latent.data, true}};
    const core::Bf16RowProjection d3[] = {{down.data, kRank, latent_baseline.data, true}};
    Outcome outcome{};
    outcome.down = run_pair(certificate, operands, 2U * kWidth, d1, d3, std::size_t{kRows} * kRank, context);
    Buffer<float> f1(std::size_t{kRows} * kWidth), f3(std::size_t{kRows} * kWidth), g1(std::size_t{kRows} * kWidth),
        g3(std::size_t{kRows} * kWidth);
    const core::Bf16RowProjection p1[] = {{forget.data, kWidth, f1.data, false}, {gain.data, kWidth, g1.data, false}};
    const core::Bf16RowProjection p3[] = {{forget.data, kWidth, f3.data, false}, {gain.data, kWidth, g3.data, false}};
    // Both gate runs read the dispatched down latent.
    outcome.gates = run_pair(certificate, latent.data, kRank, p1, p3, std::size_t{kRows} * kWidth, context);
    return outcome;
}

}  // namespace

int main() {
    std::unique_ptr<cc::BigCtx, decltype(&cc::dense_big_ctx_destroy)> ctx(cc::dense_big_ctx_create(),
                                                                        cc::dense_big_ctx_destroy);
    if (!ctx) return 1;
    const unsigned down_k = 2U * kWidth;
    Buffer<float> operands(std::size_t{kRows} * down_k);
    fill_rows(operands.data, std::size_t{kRows} * 2U, kWidth, true);  // raw and message halves, each normalised
    Buffer<std::uint16_t> down(std::size_t{kRank} * down_k), forget(std::size_t{kWidth} * kRank),
        gain(std::size_t{kWidth} * kRank);
    fill_weights(down.data, down.size);
    fill_weights(forget.data, forget.size);
    fill_weights(gain.data, gain.size);

    // 1. Ordinary finite weights: one plane runs, same bits as three planes.
    const auto finite = run_chain(down, forget, gain, operands.data, ctx.get());
    const bool finite_one_plane = finite.down.route == core::Bf16Operand::bf16_exact;

    // 2. An infinite forget weight hit by a positive latent value. One plane would give inf where three planes give NaN
    // (0 x inf on the zero planes): the certificate refuses, the dispatch runs three planes.
    forget.data[0] = 0x7F80U;
    const auto infinite = run_chain(down, forget, gain, operands.data, ctx.get());
    const bool infinite_three_planes = infinite.gates.route == core::Bf16Operand::fp32;
    {
        // INFO only: forcing one plane on these weights really differs from the baseline.
        Buffer<float> latent(std::size_t{kRows} * kRank), f1(std::size_t{kRows} * kWidth), f3(std::size_t{kRows} * kWidth);
        for (unsigned r = 0; r < kRows; ++r)
            for (unsigned i = 0; i < kRank; ++i) latent.data[std::size_t{r} * kRank + i] = 1.0F;
        const core::Bf16RowProjection q1[] = {{forget.data, kWidth, f1.data, false}};
        const core::Bf16RowProjection q3[] = {{forget.data, kWidth, f3.data, false}};
        require(cc::enqueue_bf16_dense_rows(latent.data, kRank, kRows, q1, nullptr, cc::DenseArith::mma3, ctx.get(),
                                            nullptr, core::Bf16Operand::bf16_exact));
        require(cc::enqueue_bf16_dense_rows(latent.data, kRank, kRows, q3, nullptr, cc::DenseArith::mma3, ctx.get(),
                                            nullptr, core::Bf16Operand::fp32));
        require(cudaDeviceSynchronize());
        std::cout << (std::memcmp(f1.data, f3.data, f1.size * sizeof(float)) ? "INFO" : "INFO (no difference)")
                  << " forced one plane on an infinite forget weight differs from three planes (why the guard exists)\n";
    }
    forget.data[0] = 0x3F80U;

    // 3. Oversized finite down weights (2^127) under normalised operands: the latent overflows, the certificate refuses.
    for (std::size_t i = 0; i < down.size; ++i) down.data[i] = 0x7F00U;
    const auto oversized = run_chain(down, forget, gain, operands.data, ctx.get());
    const bool oversized_three_planes = oversized.down.route == core::Bf16Operand::fp32;

    std::cout << (finite.down.equal && finite.gates.equal && finite_one_plane ? "PASS" : "FAIL")
              << " finite weights: certified one plane, down and gates bitwise == three planes\n"
              << (infinite.down.equal && infinite.gates.equal && infinite_three_planes ? "PASS" : "FAIL")
              << " infinite forget weight: refused, dispatch bitwise == three planes\n"
              << (oversized.down.equal && oversized.gates.equal && oversized_three_planes ? "PASS" : "FAIL")
              << " oversized finite down weights: refused, dispatch bitwise == three planes\n";
    return finite.down.equal && finite.gates.equal && finite_one_plane && infinite.down.equal && infinite.gates.equal &&
                   infinite_three_planes && oversized.down.equal && oversized.gates.equal && oversized_three_planes
               ? 0
               : 1;
}
