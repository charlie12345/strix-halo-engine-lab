#pragma once

#include "strix/device/gate.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

// CUDA side of the start gate. Enumeration reads device properties only: no context is created on any device, so an
// excluded card is never opened. bind_cuda() is the one call a CUDA binary makes before its first allocation.
namespace strix::device {

class CudaEnumerator final : public Enumerator {
public:
    [[nodiscard]] Vendor vendor() const noexcept override { return Vendor::nvidia; }
    [[nodiscard]] std::vector<Visible> enumerate() override {
        int count = 0;
        if (const auto status = cudaGetDeviceCount(&count); status != cudaSuccess)
            throw Refusal(std::string("device gate: refusing to start: cannot enumerate CUDA devices: ") + cudaGetErrorString(status));
        std::vector<Visible> out;
        for (int i = 0; i < count; ++i) {
            cudaDeviceProp p{};
            if (const auto status = cudaGetDeviceProperties(&p, i); status != cudaSuccess)
                throw Refusal(std::string("device gate: refusing to start: cannot read CUDA device properties: ") + cudaGetErrorString(status));
            Visible v;
            v.ordinal = i;
            v.vendor = Vendor::nvidia;
            v.name = p.name;
            v.uuid = uuid_from_bytes(reinterpret_cast<const unsigned char*>(p.uuid.bytes));
            v.pci = pci_from_parts(static_cast<unsigned>(p.pciDomainID), static_cast<unsigned>(p.pciBusID), static_cast<unsigned>(p.pciDeviceID));
            v.arch = "sm_" + std::to_string(p.major) + std::to_string(p.minor);
            v.memory_bytes = p.totalGlobalMem;
            v.integrated = p.integrated != 0;
            out.push_back(std::move(v));
        }
        return out;
    }
};

// Resolves the device (throws Refusal), makes it current and prints the receipt line on `receipt` (null = silent).
// `ordinal` is an operator-given index (-1 = none): it must point at an eligible device, it never selects by itself.
inline Bound bind_cuda(Request request = {}, const std::string& devices_path = {}, std::FILE* receipt = stderr) {
    CudaEnumerator enumerator;
    auto bound = start_gate(enumerator, std::move(request), devices_path);
    if (const auto status = cudaSetDevice(bound.visible.ordinal); status != cudaSuccess)
        throw Refusal(std::string("device gate: refusing to start: cudaSetDevice failed: ") + cudaGetErrorString(status));
    if (receipt) {
        std::fprintf(receipt, "%s\n", receipt_line(bound).c_str());
        std::fflush(receipt);
    }
    return bound;
}

} // namespace strix::device
