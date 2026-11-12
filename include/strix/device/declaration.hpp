#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// strix-devices-v1: the device declaration every GPU binary reads before its first allocation.
//
//   schema = "strix-devices-v1"
//   machine = "workstation"                      # optional, display only
//
//   [devices.main]                               # the table key is the device id
//   vendor = "nvidia"                            # amd | nvidia | intel
//   uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"   # identity: uuid and/or pci, never an index
//   arch = "sm_120"                              # optional, checked when present
//   memory_bytes = 102641958912                  # optional, informational
//   roles = ["replica", "prefill"]               # empty or absent = no work
//   attach = { kind = "pcie", pcie_gen = 4, pcie_width = 16, h2d_gbytes_ps = 0.0 }
//
//   [devices.spare]
//   vendor = "nvidia"
//   uuid = "GPU-..."
//   excluded = true
//   reason = "never schedule on this card"
//
//   [device_policies.no-spare]                   # deny rules matched on the enumerated name
//   name_contains = "4090"
//   excluded = true
//   reason = "never schedule on this card"
//
// Unknown keys, a device without an identity, duplicate identities and unknown roles are errors.
namespace strix::device {

enum class Vendor { amd, nvidia, intel };
[[nodiscard]] std::string_view vendor_name(Vendor vendor) noexcept;
[[nodiscard]] Vendor parse_vendor(std::string_view text); // throws std::invalid_argument

struct Attach {
    std::string kind;            // "" (not stated) | integrated | pcie | oculink | remote
    std::uint32_t pcie_gen = 0;  // negotiated, not the capability
    std::uint32_t pcie_width = 0;
    double h2d_gbytes_ps = 0.0;  // 0 = not probed
    double gbps_measured = 0.0;  // remote only; 0 = not probed
    std::uint64_t rtt_us_measured = 0;
};

struct Descriptor {
    std::string id;
    Vendor vendor = Vendor::amd;
    std::string arch;
    std::string uuid; // normalised: lower-case, "gpu-" prefix removed; empty when not declared
    std::string pci;  // normalised: dddd:bb:dd.f lower-case; empty when not declared
    std::string name; // display only
    std::uint64_t memory_bytes = 0;
    Attach attach;
    std::vector<std::string> roles;
    bool excluded = false;
    std::string reason;
};

struct Policy {
    std::string name;
    std::string name_contains; // case-insensitive substring of the enumerated device name
    std::string vendor;        // optional: restricts the rule to one vendor
    bool excluded = true;
    std::string reason;
};

struct Declaration {
    std::string source; // path or "<inline>"
    std::string machine;
    std::vector<Descriptor> devices;
    std::vector<Policy> policies;
};

// Throws std::invalid_argument naming the offending key.
[[nodiscard]] Declaration parse_declaration(std::string_view toml, std::string source = "<inline>");
// Additionally throws std::runtime_error when the file cannot be read.
[[nodiscard]] Declaration load_declaration(const std::string& path);

// Identity normalisation shared by the parser and the enumerators. Both throw std::invalid_argument.
[[nodiscard]] std::string normalise_uuid(std::string_view text); // 32 hex digits -> 8-4-4-4-12
[[nodiscard]] std::string normalise_pci(std::string_view text);  // [dddd:]bb:dd.f -> dddd:bb:dd.f

[[nodiscard]] bool known_role(std::string_view role) noexcept;
// Empty when the descriptor may take the role; otherwise the reason it may not (role absent, or a role that needs a
// measured rate on a link that was never probed).
[[nodiscard]] std::string role_refusal(const Descriptor& device, std::string_view role);

} // namespace strix::device
