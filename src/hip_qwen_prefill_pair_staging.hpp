#pragma once
#include "strix/qwen_prefill_handoff.hpp"
#include <hip/hip_runtime.h>

namespace strix::qwen_pp {
inline void pair_hip_check(hipError_t status, const char* operation) {
    if (status != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}
class HipPipeStaging final : public PipeStaging {
public:
    ~HipPipeStaging() override {
        (void)hipSetDevice(device_);
        if (stream_) (void)hipStreamSynchronize(stream_);
        for (const auto event : pieces_) if (event) (void)hipEventDestroy(event);
        if (snapshot_) (void)hipEventDestroy(snapshot_);
        if (stream_) (void)hipStreamDestroy(stream_);
        if (device_bytes_) (void)hipFree(device_bytes_);
        if (host_bytes_) (void)hipHostFree(host_bytes_);
    }
    void allocate(std::size_t capacity) {
        capacity_ = capacity;
        pair_hip_check(hipGetDevice(&device_), "pair prefill staging device");
        pair_hip_check(hipHostMalloc(reinterpret_cast<void**>(&host_bytes_), capacity_, hipHostMallocDefault), "pair prefill pinned seam allocation");
        pair_hip_check(hipMalloc(reinterpret_cast<void**>(&device_bytes_), capacity_), "pair prefill device seam snapshot allocation");
        pair_hip_check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "pair prefill copy stream");
        pair_hip_check(hipEventCreateWithFlags(&snapshot_, hipEventDisableTiming), "pair prefill snapshot event");
        pieces_.resize((capacity_ + seam_piece_bytes - 1U) / seam_piece_bytes);
        // CRC and the wire read pinned bytes on the CPU. Device completion
        // alone does not publish coarse-grained host memory to that agent.
        for (auto& event : pieces_) pair_hip_check(hipEventCreateWithFlags(&event,
            hipEventDisableTiming | hipEventReleaseToSystem), "pair prefill piece event");
    }
    std::span<std::byte> resize(std::size_t bytes) override {
        if (bytes > capacity_) throw std::runtime_error("pair prefill seam exceeds reserved staging capacity");
        return {host_bytes_, bytes};
    }
    void copy(bool save, std::size_t offset, void* device, std::size_t bytes) override {
        auto* packed = device_bytes_ + offset;
        pair_hip_check(hipMemcpyAsync(save ? packed : device, save ? device : packed, bytes,
            hipMemcpyDeviceToDevice, hip_execution_stream()), "pair prefill snapshot/scatter");
    }
    void seal(std::size_t bytes) override {
        pair_hip_check(hipEventRecord(snapshot_, hip_execution_stream()), "pair prefill snapshot fence");
        // Detached compute streams may reuse the source arena on return. Only
        // D2D packing must finish here; pinned D2H remains overlapped with compute.
        pair_hip_check(hipEventSynchronize(snapshot_), "pair prefill immutable snapshot ready");
        pair_hip_check(hipStreamWaitEvent(stream_, snapshot_, 0), "pair prefill copy dependency");
        for (std::size_t offset = 0; offset < bytes; offset += seam_piece_bytes) {
            const auto length = std::min(seam_piece_bytes, bytes - offset);
            pair_hip_check(hipMemcpyAsync(host_bytes_ + offset, device_bytes_ + offset, length,
                hipMemcpyDeviceToHost, stream_), "pair prefill asynchronous D2H");
            pair_hip_check(hipEventRecord(pieces_[offset / seam_piece_bytes], stream_), "pair prefill D2H piece fence");
        }
    }
    void await_piece(std::size_t offset) override {
        select_device();
        pair_hip_check(hipEventSynchronize(pieces_[offset / seam_piece_bytes]), "pair prefill D2H piece ready");
    }
    void received_piece(std::size_t offset, std::size_t bytes) override {
        select_device();
        pair_hip_check(hipMemcpyAsync(device_bytes_ + offset, host_bytes_ + offset, bytes,
            hipMemcpyHostToDevice, stream_), "pair prefill asynchronous H2D");
    }
    void finish_receive() override {
        select_device();
        pair_hip_check(hipStreamSynchronize(stream_), "pair prefill authenticated H2D complete");
    }
    void* device_piece(std::size_t offset) override { return device_bytes_ + offset; }
    void read_device_piece(std::size_t offset, std::size_t bytes) override {
        select_device();
        pair_hip_check(hipMemcpyAsync(host_bytes_ + offset, device_bytes_ + offset, bytes,
            hipMemcpyDeviceToHost, stream_), "pair prefill routed CRC readback");
        pair_hip_check(hipEventRecord(pieces_[offset / seam_piece_bytes], stream_), "pair prefill routed publication");
        pair_hip_check(hipEventSynchronize(pieces_[offset / seam_piece_bytes]), "pair prefill routed bytes ready");
    }
private:
    int device_ = 0;
    std::size_t capacity_ = 0;
    std::byte* host_bytes_ = nullptr;
    std::byte* device_bytes_ = nullptr;
    hipStream_t stream_ = nullptr;
    hipEvent_t snapshot_ = nullptr;
    std::vector<hipEvent_t> pieces_;
    void select_device() const { pair_hip_check(hipSetDevice(device_), "pair prefill worker device"); }
};
std::shared_ptr<PipeStaging> make_hip_pipe_staging(std::size_t capacity) {
    auto staging = std::make_shared<HipPipeStaging>();
    staging->allocate(capacity);
    return staging;
}

class HipStateStaging final : public StateStaging {
public:
    ~HipStateStaging() override {
        (void)hipSetDevice(device_);
        if (stream_) { (void)hipStreamSynchronize(stream_); (void)hipStreamDestroy(stream_); }
        for (auto* const& slot : slots_) if (slot) (void)hipHostFree(slot);
    }
    void allocate() {
        pair_hip_check(hipGetDevice(&device_), "pair handoff staging device");
        for (auto*& slot : slots_)
            pair_hip_check(hipHostMalloc(reinterpret_cast<void**>(&slot), seam_piece_bytes, hipHostMallocDefault), "pair handoff pinned slot");
        pair_hip_check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "pair handoff copy stream");
    }
    std::span<std::byte> buffer(std::size_t slot) override { return {slots_.at(slot), seam_piece_bytes}; }
    void copy_device(bool save, void* device, std::span<std::byte> host) override {
        pair_hip_check(hipSetDevice(device_), "pair handoff copy device");
        pair_hip_check(hipMemcpyAsync(save ? host.data() : device, save ? device : host.data(), host.size(),
            save ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice, stream_), "pair handoff async state copy");
        pair_hip_check(hipStreamSynchronize(stream_), "pair handoff state slot ready");
    }
private:
    int device_ = 0;
    hipStream_t stream_ = nullptr;
    std::array<std::byte*, 2> slots_{};
};
std::unique_ptr<StateStaging> make_hip_state_staging() {
    auto staging = std::make_unique<HipStateStaging>();
    staging->allocate();
    return staging;
}
} // namespace strix::qwen_pp
