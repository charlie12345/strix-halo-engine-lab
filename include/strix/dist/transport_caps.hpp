#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::dist {
// Transport organization only. MiniMax keeps every bit OFF until qualified.
enum TransportCapability : std::uint16_t {
    usb4_stream_table = 1U << 0U,
    usb4_joint_calibration = 1U << 1U,
    staged_readiness = 1U << 2U,
    deferred_crc = 1U << 3U,
    host_chunk_default = 1U << 4U,
    host_crc_overlap_default = 1U << 5U,
    gpu_queue_lean = 1U << 6U,
};
inline constexpr std::uint16_t all_transport_capabilities = 0x7fU;
constexpr bool transport_has(std::uint16_t bits, TransportCapability capability) {
    return (bits & capability) != 0U;
}
// Preserve the existing all-OFF v2 and all-ON v3 bytes. Partial sets carry
// their exact mask in the existing CRC-covered hello/plan version field;
// their plan uses the v3 layout. No extra frame or payload byte is added.
inline std::uint16_t transport_pairing_version(std::uint16_t bits) {
    if ((bits & ~all_transport_capabilities) != 0U)
        throw std::invalid_argument("usb4 pairing: unknown transport capability bits");
    if (bits == 0U) return 2U;
    if (bits == all_transport_capabilities) return 3U;
    return static_cast<std::uint16_t>(0x8000U | bits);
}
inline void require_transport_agreement(std::uint16_t local, std::uint16_t peer) {
    if (local != peer)
        throw std::runtime_error("usb4 pairing: transport capability mismatch (local version/mask " +
            std::to_string(local) + ", peer " + std::to_string(peer) +
            "); both ranks must set the same STRIX_MM_USB4_CAPS");
}
inline std::uint16_t parse_transport_capabilities(std::string_view setting) {
    std::uint16_t bits = 0;
    while (!setting.empty()) {
        const auto comma = setting.find(',');
        const auto name = setting.substr(0, comma);
        if (name == "streams") bits |= usb4_stream_table;
        else if (name == "joint") bits |= usb4_joint_calibration;
        else if (name == "ready") bits |= staged_readiness;
        else if (name == "deferred-crc") bits |= deferred_crc;
        else if (name == "chunks") bits |= host_chunk_default;
        else if (name == "crc-overlap") bits |= host_crc_overlap_default;
        else if (name == "gpu-lean") bits |= gpu_queue_lean;
        else throw std::invalid_argument("STRIX_MM_USB4_CAPS: expected streams,joint,ready,deferred-crc,chunks,crc-overlap,gpu-lean");
        if (comma == std::string_view::npos) break;
        setting.remove_prefix(comma + 1U);
        if (setting.empty()) throw std::invalid_argument("STRIX_MM_USB4_CAPS: empty capability");
    }
    return bits;
}
} // namespace strix::dist
