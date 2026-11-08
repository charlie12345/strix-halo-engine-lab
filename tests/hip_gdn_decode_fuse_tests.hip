#include "strix/hip_gdn_decode_fuse.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
struct Buffer {
    void* p = nullptr;
    explicit Buffer(std::size_t bytes) { check(hipMalloc(&p, bytes)); }
    ~Buffer() { if (p) (void)hipFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* as() { return static_cast<T*>(p); }
    template<class T> void put(const std::vector<T>& v) {
        check(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    template<class T> std::vector<T> get(std::size_t n) {
        std::vector<T> v(n);
        check(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
        return v;
    }
};
template<class T> void equal(const std::vector<T>& a, const std::vector<T>& b, const char* label) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)))
        throw std::runtime_error(std::string("byte mismatch: ") + label);
}
std::uint16_t bf16(float v) {
    auto b = std::bit_cast<std::uint32_t>(v);
    return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U);
}
std::vector<float> values(std::size_t n, unsigned seed, float scale = 0.125F) {
    std::vector<float> v(n);
    for (auto& x : v) {
        seed = seed * 1664525U + 1013904223U;
        x = static_cast<float>(static_cast<int>((seed >> 16U) & 255U) - 128) * scale / 128.0F;
    }
    return v;
}
struct Weights {
    Buffer data;
    explicit Weights(const std::vector<float>& v, unsigned type) : data(v.size() * (type ? 2U : 4U)) {
        if (!type) data.put(v);
        else {
            std::vector<std::uint16_t> bits(v.size());
            for (std::size_t i = 0; i < v.size(); ++i) bits[i] = type == 30 ? bf16(v[i]) :
                std::bit_cast<std::uint16_t>(__float2half(v[i]));
            data.put(bits);
        }
    }
};

// Production GDN workspace order: qkv | z | beta | gate | core.
void run(unsigned type, unsigned kh, unsigned vh, bool guarded, bool graph, hipStream_t stream) {
    constexpr unsigned d = 128, kernel = 4;
    const unsigned channels = (2 * kh + vh) * d, width = vh * d;
    const unsigned words = channels + 2 * width + 2 * vh;
    auto conv = values(channels * kernel, 3, 0.5F);
    auto dt = values(vh, 7);
    auto a = values(vh, 11, 0.5F);
    for (unsigned h = 0; h < vh; ++h) { a[h] = -std::fabs(a[h]); dt[h] = h % 2 ? -32.0F : 32.0F; }
    auto norm = values(d, 13);
    for (auto& x : norm) x += 1.0F;
    Weights cw(conv, type), dw(dt, type), aw(a, type), nw(norm, type);
    std::array<Buffer, 2> work{Buffer(words * 4ULL), Buffer(words * 4ULL)};
    std::array<Buffer, 2> history{Buffer(channels * kernel * 2ULL), Buffer(channels * kernel * 2ULL)};
    std::array<Buffer, 2> state{Buffer(vh * d * d * 4ULL), Buffer(vh * d * d * 4ULL)};
    std::array<Buffer, 2> status{Buffer(4), Buffer(4)};
    const std::vector<float> zero_state(vh * d * d, 0);
    const std::vector<std::uint16_t> zero_history(channels * kernel, 0);
    for (unsigned arm = 0; arm < 2; ++arm) { history[arm].put(zero_history); state[arm].put(zero_state); }
    // Position does not enter this post-projection layer body. The sequence
    // exercises first-token zero state and carried state with distinct projected
    // images for three layers and short/long position labels.
    constexpr std::array positions{0U, 1U, 1023U, 32767U, 131071U};
    for (unsigned layer = 0; layer < 3; ++layer) for (unsigned pos : positions) {
        auto input = values(words, 17U + layer * 181U + pos);
        for (unsigned arm = 0; arm < 2; ++arm) {
            work[arm].put(input); status[arm].put(std::vector<unsigned>{0});
            strix::HipGdnDecodeFuseFixture v{};
            v.qkv = work[arm].as<float>(); v.z = v.qkv + channels;
            v.beta = v.qkv + channels + width; v.gate = v.beta + vh; v.core = v.gate + vh;
            v.matrix = state[arm].as<float>(); v.conv_state = history[arm].as<std::uint16_t>();
            v.conv = cw.data.p; v.dt = dw.data.p; v.a = aw.data.p; v.norm = nw.data.p;
            v.small_type = type; v.key_heads = kh; v.value_heads = vh;
            hipGraph_t captured = nullptr;
            hipGraphExec_t executable = nullptr;
            if (graph) check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
            strix::hip_gdn_decode_fuse_fixture(v, arm == 1, guarded ? status[arm].as<unsigned>() : nullptr, stream);
            if (graph) {
                check(hipStreamEndCapture(stream, &captured));
                check(hipGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
                check(hipGraphLaunch(executable, stream));
            }
            check(hipStreamSynchronize(stream));
            if (graph) { check(hipGraphExecDestroy(executable)); check(hipGraphDestroy(captured)); }
        }
        equal(work[0].get<float>(words), work[1].get<float>(words), "finite workspace/core");
        equal(history[0].get<std::uint16_t>(channels * kernel), history[1].get<std::uint16_t>(channels * kernel), "convolution history");
        equal(state[0].get<float>(vh * d * d), state[1].get<float>(vh * d * d), "carried recurrent state");
        equal(status[0].get<unsigned>(1), status[1].get<unsigned>(1), "finite status");
        if (status[0].get<unsigned>(1)[0]) throw std::runtime_error("finite fixture rejected");
    }
    if (!guarded) return;
    for (unsigned bad = 0; bad < 7; ++bad) {
        auto input = values(words, 71);
        if (bad == 1) input[0] = std::numeric_limits<float>::quiet_NaN();
        if (bad == 2) input[channels + width] = std::numeric_limits<float>::infinity();
        if (bad == 3) input[channels + width + vh] = std::numeric_limits<float>::infinity();
        if (bad == 4) input[channels] = std::numeric_limits<float>::quiet_NaN();
        if (bad == 5) for (unsigned i = 0; i < channels; ++i) input[i] = 1e21F;
        if (bad == 6) input[0] = std::numeric_limits<float>::infinity();
        for (unsigned arm = 0; arm < 2; ++arm) {
            work[arm].put(input); history[arm].put(zero_history); state[arm].put(zero_state);
            status[arm].put(std::vector<unsigned>{bad == 0 ? 4U : 0U});
            strix::HipGdnDecodeFuseFixture v{};
            v.qkv = work[arm].as<float>(); v.z = v.qkv + channels;
            v.beta = v.qkv + channels + width; v.gate = v.beta + vh; v.core = v.gate + vh;
            v.matrix = state[arm].as<float>(); v.conv_state = history[arm].as<std::uint16_t>();
            v.conv = cw.data.p; v.dt = dw.data.p; v.a = aw.data.p; v.norm = nw.data.p;
            v.small_type = type; v.key_heads = kh; v.value_heads = vh;
            strix::hip_gdn_decode_fuse_fixture(v, arm == 1, status[arm].as<unsigned>(), stream);
            check(hipStreamSynchronize(stream));
        }
        equal(history[0].get<std::uint16_t>(channels * kernel), history[1].get<std::uint16_t>(channels * kernel), "rejected convolution history");
        equal(state[0].get<float>(vh * d * d), state[1].get<float>(vh * d * d), "rejected recurrent state");
        equal(status[0].get<unsigned>(1), status[1].get<unsigned>(1), "rejection status");
        if (!status[0].get<unsigned>(1)[0]) throw std::runtime_error("nonfinite fixture accepted");
        // Rejected workspace is not committed. Original guarded normalization
        // may suppress other CTAs after one head publishes its failure.
    }
}
} // namespace

int main() {
    try {
        hipStream_t stream = nullptr;
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        for (unsigned type : {0U, 1U, 30U}) for (auto shape : {std::array{1U, 1U}, std::array{2U, 4U}, std::array{16U, 48U}})
            for (bool guarded : {false, true}) for (bool graph : {false, true})
                run(type, shape[0], shape[1], guarded, graph, stream);
        check(hipStreamDestroy(stream));
        std::cout << "EXACT: R1 fusion device/reference workspace, output, convolution and recurrent state; guards\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
