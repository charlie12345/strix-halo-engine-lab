#pragma once
// Host-only parsing shared by the transport opener and its tests. Spellings are
// an app contract: callers select their existing dialect rather than add aliases.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::dist {
enum class TransportKind { tcp_listen, tcp_connect, usb4_host, usb4_auto, usb4_gpu };
struct TransportSpellings {
    std::string_view host = "usb4auto", gpu = "usb4gpu", host_alias = {};
    bool tcp = true, devices = false;
    bool usb_suffix = true; // the existing kind:rest apps ignored USB4 rest
};
inline constexpr TransportSpellings compact_transport{"usb4auto", "usb4gpu", {}, true, true};
inline constexpr TransportSpellings hyphen_transport{"usb4-auto", "usb4gpu"};
inline constexpr TransportSpellings prefill_transport{"usb4-auto", "usb4-gpu"};
struct TransportSpec {
    TransportKind kind;
    std::string host;
    std::uint16_t port = 0;
    std::vector<std::string> devices;
};
inline TransportSpec parse_transport(const std::string& spec, TransportSpellings spellings) {
    const auto colon = spec.find(':');
    const auto kind = spec.substr(0, colon);
    const auto rest = colon == std::string::npos ? "" : spec.substr(colon + 1);
    if (spellings.tcp && kind == "tcp-listen")
        return {TransportKind::tcp_listen, {}, static_cast<std::uint16_t>(std::stoul(rest)), {}};
    if (spellings.tcp && kind == "tcp-connect") {
        const auto c2 = rest.rfind(':');
        return {TransportKind::tcp_connect, rest.substr(0, c2), static_cast<std::uint16_t>(std::stoul(rest.substr(c2 + 1))), {}};
    }
    if (!spellings.usb_suffix && colon != std::string::npos)
        throw std::invalid_argument("unknown transport " + spec);
    if ((!spellings.host.empty() && kind == spellings.host) || (!spellings.host_alias.empty() && kind == spellings.host_alias))
        return {TransportKind::usb4_auto, {}, 0, {}};
    if (!spellings.gpu.empty() && kind == spellings.gpu) return {TransportKind::usb4_gpu, {}, 0, {}};
    if (spellings.devices && kind == "usb4") {
        TransportSpec result{TransportKind::usb4_host, {}, 0, {}};
        for (std::size_t begin = 0; begin < rest.size();) {
            const auto end = rest.find(',', begin);
            result.devices.push_back(rest.substr(begin, end == std::string::npos ? end : end - begin));
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        return result;
    }
    throw std::invalid_argument("unknown transport " + spec);
}
// Route identity after opening: a requested GPU transport is still the ordinary
// host route unless a live device exchange was negotiated. Never infer this from
// the request or the human-readable transport receipt.
inline std::string negotiated_transport(const std::string& requested, bool direct_live,
                                        TransportSpellings spellings) {
    if (parse_transport(requested, spellings).kind == TransportKind::usb4_gpu)
        return std::string(direct_live ? spellings.gpu : spellings.host);
    return requested;
}
} // namespace strix::dist
