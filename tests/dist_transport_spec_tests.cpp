#include "strix/dist/transport_spec.hpp"
#include <iostream>

namespace {
void require(bool ok) { if (!ok) throw std::runtime_error("transport parser contract"); }
void rejects(const std::string& text, strix::dist::TransportSpellings spellings) {
    bool caught = false;
    try { (void)strix::dist::parse_transport(text, spellings); }
    catch (const std::invalid_argument&) { caught = true; }
    require(caught);
}
}
int main() try {
    using namespace strix::dist;
    for (const auto dialect : {compact_transport, hyphen_transport, prefill_transport}) {
        require(negotiated_transport(std::string(dialect.gpu), false, dialect) == dialect.host);
        require(negotiated_transport(std::string(dialect.gpu), true, dialect) == dialect.gpu);
        require(negotiated_transport(std::string(dialect.host), false, dialect) == dialect.host);
        require(negotiated_transport("tcp-listen:29872", false, dialect) == "tcp-listen:29872");
        require(parse_transport(std::string(dialect.host), dialect).kind == TransportKind::usb4_auto);
        require(parse_transport(std::string(dialect.gpu), dialect).kind == TransportKind::usb4_gpu);
        const auto listen = parse_transport("tcp-listen:29872", dialect);
        require(listen.kind == TransportKind::tcp_listen && listen.port == 29872);
        const auto connect = parse_transport("tcp-connect:peer:29872", dialect);
        require(connect.kind == TransportKind::tcp_connect && connect.host == "peer" && connect.port == 29872);
        rejects("TCP-listen:29872", dialect);
        rejects("tcp-listen:", dialect);
        rejects("tcp-connect:peer:bad", dialect);
        rejects("usb4_gpu_auto", dialect); // manifest mapping remains an app concern
        // Preserve the former kind/rest split, including ignored USB4 suffixes
        // and the former stoul/narrowing TCP behavior.
        require(parse_transport(std::string(dialect.host) + ":ignored", dialect).kind == TransportKind::usb4_auto);
        require(parse_transport("tcp-listen:65537suffix", dialect).port == 1);
    }
    rejects("usb4-auto", compact_transport);
    rejects("usb4auto", hyphen_transport);
    rejects("usb4-gpu", hyphen_transport);
    rejects("usb4gpu", prefill_transport);
    rejects("usb4:/dev/a", hyphen_transport);
    const auto devices = parse_transport("usb4:a,,b,", compact_transport);
    require(devices.kind == TransportKind::usb4_host && devices.devices == std::vector<std::string>({"a", "", "b"}));
    const TransportSpellings qwen{"usb4auto", "usb4gpu", "usb4-auto", false, false, false};
    require(parse_transport("usb4-auto", qwen).kind == TransportKind::usb4_auto);
    require(parse_transport("usb4gpu", qwen).kind == TransportKind::usb4_gpu);
    rejects("tcp-listen:29872", qwen);
    rejects("usb4:a", qwen);
    rejects("usb4auto:ignored", qwen);
    rejects("usb4gpu:ignored", qwen);
    std::cout << "PASS: shared opener dialects, TCP endpoints, USB4 devices, aliases and rejection contracts\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
