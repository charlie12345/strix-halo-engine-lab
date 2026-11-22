// Test-only inclusion lets the fixture bind the private incumbent dot directly
// and compare against dense_rows without introducing a copy of its arithmetic.
#include "cuda/core/dense/dense.cu"
#include "cuda/core/head_screened.cuh"
#include "screened_head_cases.hpp"

#include <cstdio>

namespace strix::cuda_core {
namespace {
struct IncumbentBf16R1 {
    unsigned* calls;

    __device__ float operator()(const std::uint16_t* weights, const float* x, unsigned width, unsigned lane) const {
        if (calls && lane == 0) atomicAdd(calls, 1U);
        float accum[1][1]{{0.0F}};
        dot_bf16<1, 1>(weights, width, x, width, 1U, static_cast<int>(lane), accum);
        return warp_sum_xor(accum[0][0]);
    }
};
}  // namespace
}  // namespace strix::cuda_core

namespace {
using namespace b7_fixture;
namespace cuda_head = strix::cuda::core::head;
namespace dense = strix::cuda_core;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class T>
class Device {
public:
    explicit Device(std::size_t count) : size(count) { check(cudaMalloc(&data, size * sizeof(T))); }
    ~Device() { cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    void upload(std::span<const T> source) const {
        require(source.size() == size, "upload size");
        check(cudaMemcpy(data, source.data(), source.size_bytes(), cudaMemcpyHostToDevice));
    }

    std::vector<T> download() const {
        std::vector<T> result(size);
        check(cudaMemcpy(result.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost));
        return result;
    }

    T* data = nullptr;

private:
    std::size_t size;
};

void check_case(const Case& test) {
    const auto& plan = test.plan;
    const auto screen = head::build_screen(plan, test.weights);
    Device<std::uint16_t> weights(plan.weight_count());
    Device<float> activation(plan.width()), maximum(1), incumbent(plan.vocab());
    Device<std::uint8_t> packed(plan.packed_bytes());
    Device<head::RowCertificate> certificates(plan.vocab());
    Device<head::ha::Pick2> partials(plan.partial_count());
    Device<head::Result> result(1);
    Device<unsigned> calls(1);
    weights.upload(test.weights);
    activation.upload(test.activation);
    packed.upload(screen.packed);
    certificates.upload(screen.certificates);
    check(cudaMemset(calls.data, 0, sizeof(unsigned)));

    const cuda_head::ScreenedBuffers buffers{weights.data, activation.data, packed.data, certificates.data,
                                             maximum.data, partials.data, result.data};
    // Use a counter only on the cases with a prescribed pruning outcome.
    const std::string_view name(test.name);
    const bool witness_pruning = name == "pruning-ties";
    const bool witness_fallback = name == "nonfinite-activation";
    check(cuda_head::enqueue_screened(plan, buffers,
        dense::IncumbentBf16R1{witness_pruning || witness_fallback ? calls.data : nullptr}, nullptr));
    const dense::DenseWeight view{weights.data, dense::WeightFormat::bf16, plan.vocab(), plan.width()};
    check(dense::dense_rows(view, activation.data, incumbent.data, 1, nullptr, nullptr, dense::DenseArith::fp32));

    const auto old_logits = incumbent.download();
    const auto actual = result.download().front();
    const std::span host_weights(test.weights);
    std::vector<float> reference(plan.vocab());
    for (unsigned row = 0; row < plan.vocab(); ++row) {
        reference[row] = incumbent_score(host_weights.subspan(std::size_t{row} * plan.width(), plan.width()), test.activation);
        if (std::isfinite(reference[row])) {
            require(std::bit_cast<std::uint32_t>(old_logits[row]) == std::bit_cast<std::uint32_t>(reference[row]),
                    "incumbent host/device score bits");
        } else {
            require(!std::isfinite(old_logits[row]), "incumbent host/device nonfinite classification");
        }
    }
    require(same(actual, head::emit(head::ha::reference2(old_logits, plan.first()))), "screened CUDA parity");
    if (witness_pruning) require(calls.download().front() < plan.vocab(), "screen pruned no rows");
    if (witness_fallback) require(calls.download().front() == plan.vocab(), "screen pruned an uncertified row");
    std::printf("b7 cuda %s PASS\n", test.name);
}
}  // namespace

int main() {
    try {
        for (const auto& test : b7_fixture::cases()) check_case(test);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "b7 cuda FAIL %s\n", e.what());
        return 1;
    }
}
