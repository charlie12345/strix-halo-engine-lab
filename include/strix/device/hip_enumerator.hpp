#pragma once

#include "strix/device/gate.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

// HIP side of the start gate. Enumeration reads device properties only. bind_hip() is the one call a HIP binary makes
// before its first allocation. Without a declaration, a lone integrated device keeps working (receipt: declared=implicit).
namespace strix::device {

class HipEnumerator final : public Enumerator {
public:
    [[nodiscard]] Vendor vendor() const noexcept override { return Vendor::amd; }
    [[nodiscard]] std::vector<Visible> enumerate() override {
        int count = 0;
        if (const auto status = hipGetDeviceCount(&count); status != hipSuccess)
            throw Refusal(std::string("device gate: refusing to start: cannot enumerate HIP devices: ") + hipGetErrorString(status));
        std::vector<Visible> out;
        for (int i = 0; i < count; ++i) {
            hipDeviceProp_t p{};
            if (const auto status = hipGetDeviceProperties(&p, i); status != hipSuccess)
                throw Refusal(std::string("device gate: refusing to start: cannot read HIP device properties: ") + hipGetErrorString(status));
            Visible v;
            v.ordinal = i;
            v.vendor = Vendor::amd;
            v.name = p.name;
            v.pci = pci_from_parts(static_cast<unsigned>(p.pciDomainID), static_cast<unsigned>(p.pciBusID), static_cast<unsigned>(p.pciDeviceID));
            v.arch = p.gcnArchName;
            if (const auto colon = v.arch.find(':'); colon != std::string::npos) v.arch.erase(colon);
            v.memory_bytes = p.totalGlobalMem;
            v.integrated = p.integrated != 0;
            out.push_back(std::move(v));
        }
        return out;
    }
};

// Resolves the device (throws Refusal), makes it current and prints the receipt line on `receipt` (null = silent).
inline Bound bind_hip(Request request = {}, const std::string& devices_path = {}, std::FILE* receipt = stderr) {
    HipEnumerator enumerator;
    auto bound = start_gate(enumerator, std::move(request), devices_path);
    if (const auto status = hipSetDevice(bound.visible.ordinal); status != hipSuccess)
        throw Refusal(std::string("device gate: refusing to start: hipSetDevice failed: ") + hipGetErrorString(status));
    if (receipt) {
        std::fprintf(receipt, "%s\n", receipt_line(bound).c_str());
        std::fflush(receipt);
    }
    return bound;
}

} // namespace strix::device
