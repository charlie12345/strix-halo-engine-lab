#include "cuda/core/attn_rowgroup_launch.cuh"
#include "strix/cuda/qwen_qsa.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace rg = strix::cuda::core::rowgroup;
namespace qsa = strix::cuda::qwen_qsa;

namespace {

__device__ __forceinline__ float round_bf16(float v) {
    const auto b = __float_as_uint(v);
    const auto rounded = static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U);
    return __uint_as_float(static_cast<std::uint32_t>(rounded) << 16U);
}

// Test-only twin of QsaRowgroup in qwen_qsa.cu, including its argument ABI.
struct QwenRowgroup {
    static constexpr unsigned D = qsa::kHeadDim, Heads = qsa::kHeads / qsa::kKvHeads, Rows = 4, Cap = qsa::kMaxSelected;
    static constexpr float kScaleLog2 = 0.0625F * 1.44269504088896341F;
    const float* q_gate;
    std::uint64_t first_position;
    const std::uint32_t* selected;
    const std::uint16_t* kv;
    float* output;
    std::uint32_t rows;

    __device__ unsigned count(unsigned row) const { return qsa::selected_count(first_position + row); }
    __device__ std::uint32_t key(unsigned row, unsigned j) const { return selected[static_cast<std::uint64_t>(row) * Cap + j]; }
    __device__ const std::uint16_t* key_row(std::uint32_t token, unsigned kvh) const {
        return kv + static_cast<std::uint64_t>(token) * (qsa::kKvWidth * 2ULL) + static_cast<std::uint64_t>(kvh) * D;
    }
    __device__ const std::uint16_t* value_row(std::uint32_t token, unsigned kvh) const { return key_row(token, kvh) + qsa::kKvWidth; }
    __device__ float query(unsigned row, unsigned kvh, unsigned h, unsigned d) const {
        return q_gate[static_cast<std::uint64_t>(row) * qsa::kQGateWidth + static_cast<std::uint64_t>(kvh * Heads + h) * D * 2ULL + d];
    }
    __device__ void store2(unsigned row, unsigned kvh, unsigned h, unsigned d, float a0, float a1) const {
        const auto head = kvh * Heads + h;
        const float* g = q_gate + static_cast<std::uint64_t>(row) * qsa::kQGateWidth + static_cast<std::uint64_t>(head) * D * 2ULL + D + d;
        const float g0 = round_bf16(1.0F / (1.0F + __expf(-g[0]))), g1 = round_bf16(1.0F / (1.0F + __expf(-g[1])));
        *reinterpret_cast<float2*>(output + static_cast<std::uint64_t>(row) * qsa::kAttnWidth + static_cast<std::uint64_t>(head) * D + d) =
            make_float2(round_bf16(round_bf16(a0) * g0), round_bf16(round_bf16(a1) * g1));
    }
};

using Shape = rg::Shape<QwenRowgroup>;
static_assert(Shape::Threads == 128 && Shape::UnionSmem == 49232 && Shape::FlashSmem == 33792);

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class T> struct Buffer {
    T* ptr = nullptr;
    std::size_t bytes;
    explicit Buffer(std::size_t count) : bytes(count * sizeof(T)) { check(cudaMalloc(&ptr, bytes)); }
    explicit Buffer(const std::vector<T>& src) : Buffer(src.size()) {
        check(cudaMemcpy(ptr, src.data(), bytes, cudaMemcpyHostToDevice));
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { cudaFree(ptr); }
    std::vector<T> read() const {
        std::vector<T> dst(bytes / sizeof(T));
        check(cudaMemcpy(dst.data(), ptr, bytes, cudaMemcpyDeviceToHost));
        return dst;
    }
};

int compare(cudaStream_t stream) {
    constexpr unsigned rows = 7, first = 15;
    constexpr auto plan = *rg::make_launch_plan<QwenRowgroup>(rows, qsa::kKvHeads);
    constexpr auto scratch_bytes = rg::workspace_bytes<QwenRowgroup>(plan.tiles);
    static_assert(scratch_bytes == 82048);
    std::vector<std::uint32_t> selected(std::size_t{rows} * qsa::kMaxSelected);
    for (unsigned row = 0; row < rows; ++row) {
        auto* list = selected.data() + std::size_t{row} * qsa::kMaxSelected;
        for (unsigned j = 0; j < qsa::selected_count(first + row); ++j) list[j] = j;
        if (row == 5) list[3] = 1;  // The second tile uses concatenation and retains a duplicate.
    }
    std::vector<float> q_gate(std::size_t{rows} * qsa::kQGateWidth);
    for (std::size_t i = 0; i < q_gate.size(); ++i)
        q_gate[i] = static_cast<float>(static_cast<int>((i * 7U) % 29U) - 14) / 16.0F;
    constexpr std::uint16_t values[] = {0x3d80, 0x3e00, 0x3e80, 0xbe80, 0x3f00, 0xbf00, 0x3f80};
    std::vector<std::uint16_t> kv(std::size_t{first + rows} * 2U * qsa::kKvWidth);
    for (std::size_t i = 0; i < kv.size(); ++i) kv[i] = values[i % 7U];

    const Buffer<float> queries(q_gate);
    const Buffer<std::uint32_t> lists(selected);
    const Buffer<std::uint16_t> cache(kv);
    const Buffer<std::uint8_t> private_scratch(scratch_bytes), shared_scratch(scratch_bytes);
    const Buffer<float> private_out(rows * qsa::kAttnWidth), shared_out(rows * qsa::kAttnWidth);
    check(cudaMemsetAsync(private_scratch.ptr, 0xa5, private_scratch.bytes, stream));
    check(cudaMemsetAsync(shared_scratch.ptr, 0xa5, shared_scratch.bytes, stream));
    check(cudaMemsetAsync(private_out.ptr, 0xff, private_out.bytes, stream));
    check(cudaMemsetAsync(shared_out.ptr, 0x7f, shared_out.bytes, stream));

    qsa::attention_mma(queries.ptr, rows, first, lists.ptr, cache.ptr, private_scratch.ptr, private_out.ptr, stream);
    rg::enqueue_rowgroup(QwenRowgroup{queries.ptr, first, lists.ptr, cache.ptr, shared_out.ptr, rows},
                         qsa::kKvHeads, shared_scratch.ptr, shared_scratch.bytes, stream);
    check(cudaStreamSynchronize(stream));

    const auto private_union = private_scratch.read();
    auto shared_union = shared_scratch.read();
    std::uint32_t counts[2]{};
    std::memcpy(counts, rg::carve<QwenRowgroup>(shared_union.data(), plan.tiles).count, sizeof(counts));
    const bool union_ok = private_union == shared_union && counts[0] == 19U && counts[1] == 63U;
    const auto expected = private_out.read(), actual = shared_out.read();
    const bool output_ok = std::memcmp(expected.data(), actual.data(), shared_out.bytes) == 0 &&
        std::all_of(actual.begin(), actual.end(), [](float x) { return std::isfinite(x); });
    std::printf("QSA union %s: 2 tiles, counts 19 and 63\n", union_ok ? "PASS" : "FAIL");
    std::printf("QSA flash %s: 43008 outputs bitwise\n", output_ok ? "PASS" : "FAIL");
    return union_ok && output_ok ? 0 : 1;
}
}  // namespace

int main() {
    cudaStream_t stream = nullptr;
    try {
        rg::prepare_rowgroup<QwenRowgroup>();
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const int result = compare(stream);
        check(cudaStreamDestroy(stream));
        return result;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "QSA fixture FAIL: %s\n", e.what());
        if (stream != nullptr) cudaStreamDestroy(stream);
        return 1;
    }
}
