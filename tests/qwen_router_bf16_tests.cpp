#include "../apps/qwen_effective_config.hpp"
#include "strix/q6p_registry.hpp"

#include <array>
#include <iostream>

using namespace strix;
using namespace strix::decode_cli;
namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
}
int main() try {
    for (const auto bits : {0U, 0x80000000U, 0x3f800000U, 0xbf800000U, 0x00010000U,
             0x7f800000U, 0xff800000U, 0x7fc10000U, 0xffc30000U, 0x7f810000U}) {
        const std::array values{bits};
        require(router_bf16_exact(std::as_bytes(std::span(values))), "representable coefficient rejected");
    }
    for (const auto bits : {1U, 0x80000001U, 0x3f800001U, 0x7f800001U, 0x7fc10001U, 0xffc30001U}) {
        // A single bad coefficient anywhere must decline the whole tensor.
        for (unsigned pos = 0; pos < 3U; ++pos) {
            std::array values{0U, 0x80000000U, 0x3f800000U}; values[pos] = bits;
            require(!router_bf16_exact(std::as_bytes(std::span(values))), "inexact tensor accepted");
        }
    }
    const std::array<std::byte, 3> truncated{};
    require(!router_bf16_exact(truncated), "partial F32 accepted");
    ModelManifest manifest;
    for (unsigned layer = 0; layer < 48U; ++layer) {
        ManifestTensor t;
        t.name = "blk." + std::to_string(layer) + ".ffn_gate_inp.weight";
        t.role = TensorRole::moe_router; t.ggml_type = 0U;
        t.dimensions = {2560U, 512U}; t.byte_size = 2560U * 512U * 4U;
        manifest.tensors.push_back(t);
    }
    constexpr std::uint64_t bytes = 125829120U; // 120 MiB; 2,621,440 bytes per layer
    require(qwen_router_bf16_reservation(manifest, true).device_bytes == bytes, "48-router census");
    auto wrong = manifest.tensors.front();
    wrong.ggml_type = 30U; require(!qwen_router_bf16_candidate(wrong), "non-F32 candidate");
    wrong = manifest.tensors.front(); wrong.byte_size--; require(!qwen_router_bf16_candidate(wrong), "malformed candidate");
    wrong = manifest.tensors.front(); wrong.dimensions = {UINT64_MAX, 512U};
    require(!qwen_router_bf16_candidate(wrong), "overflow candidate");
    // Shared charge is independent of CLI/HTTP, rows, context and rotary mode.
    for (const bool http : {false, true}) for (const auto context : {16384U, 1048576U})
        for (const bool named : {false, true}) {
            RequestedOptions request; request.options.http_serve = http; request.options.context_tokens = context;
            auto config = initialize_prefill_config(request);
            require(!config.options.qwen_router_bf16, "opt-in is on by default");
            request.environment["STRIX_QWEN_ROUTER_BF16"] = "1";
            config = initialize_prefill_config(request);
            require(config.options.qwen_router_bf16, "opt-in not on");
            const auto json = serve::parse_json(effective_config_json(config));
            require(json.find("qwen_router_bf16")->find("enabled")->as_bool(), "resolver report");
            PrefillRequirements r; r.runtime_payload = 100U; r.device_base = 200U;
            r.owners["runtime_payload"] = 100U;
            account_qwen_router_bf16(r, manifest, config.options.qwen_router_bf16, named);
            require(r.device_base == 200U + bytes && r.runtime_payload == 100U + bytes + 65536U, "UMA/GTT charge");
            if (named) {
                const auto peaks = prefill_owner_phase_peaks(r.owners, r.owner_phases);
                require(peaks[0] == 100U + bytes + 65536U && peaks[3] == 100U + bytes, "phase charge/double count");
            }
            for (const auto off : {"STRIX_QWEN_ROUTER_BF16", "STRIX_Q6P", "STRIX_QPF2K_DEFAULTS"}) {
                request.environment.clear(); request.environment["STRIX_QWEN_ROUTER_BF16"] = "1"; request.environment[off] = "0";
                config = initialize_prefill_config(request);
                require(!config.options.qwen_router_bf16, "off ignored");
                PrefillRequirements disabled;
                account_qwen_router_bf16(disabled, manifest, config.options.qwen_router_bf16, named);
                require(disabled.device_base == 0U && disabled.runtime_payload == 0U, "off charged");
            }
            request.environment.clear(); request.options.baseline = true;
            require(!initialize_prefill_config(request).options.qwen_router_bf16, "baseline ignored");
        }
    // Owner teardown must not erase another owner's registration or replacement.
    int original = 0, other = 0;
    const unsigned char a = 0, b = 0;
    q6p::register_tensor(&original, {&a, 1U, 1U, q6p::Kind::bf16});
    q6p::register_tensor(&other, {&a, 1U, 1U, q6p::Kind::bf16});
    q6p::register_tensor(&original, {&b, 1U, 1U, q6p::Kind::bf16});
    q6p::unregister_tensor(&original, &a);
    require(q6p::find(&original)->base == &b && q6p::registered(&other), "teardown erased replacement");
    q6p::unregister_tensor(&original, &b); q6p::unregister_tensor(&other, &a);
    require(q6p::registered_count() == 0U, "owned registrations leaked");
    std::cout << "Qwen router BF16 exactness, admission, resolver and registry checks passed\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
