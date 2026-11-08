#include "../src/hip_gdn_identity.hpp"
#include "strix/compiled_execution.hpp"

#include <array>
#include <iostream>
#include <set>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void no_launch(const strix::ExecutionDispatchContext&, void*) {
    throw std::logic_error("host identity test must never dispatch");
}

// Full host compiler/binder path, with the actual GDN id builder. Other operators
// are inert registry fixtures; this test neither links HIP nor submits device work.
strix::CompiledSerialExecution compile(const strix::HipGdnLauncherConfig& config,
    strix::ExecutionBindingRequest request, std::string_view reference_gdn_id = {}) {
    using namespace strix;
    kernels::KernelRegistry registry;
    std::set<std::pair<kernels::OperatorFamily, std::uint64_t>> seen;
    const auto execution = build_serial_decode_execution_plan(config.model);
    for (const auto& step : execution.steps) {
        const auto family = operator_family_for(step.op);
        if (!family) continue;
        const auto shape = execution_shape_for(config.model, step.op, request);
        if (!seen.emplace(*family, shape.auxiliary_size).second) continue;
        kernels::KernelVariant variant;
        variant.id = step.op == SerialOp::gated_delta_net ?
            (reference_gdn_id.empty() ? hip_gdn_identity::kernel_id(request, config) : std::string(reference_gdn_id)) :
            "fixture." + std::to_string(static_cast<unsigned>(*family)) + "." + std::to_string(shape.auxiliary_size);
        variant.family = *family;
        variant.phase = kernels::ExecutionPhase::decode;
        variant.shape = shape;
        variant.wave = kernels::WaveSize::wave32;
        variant.resources.workgroup_threads = 256;
        variant.numerics.input_precision = "bf16";
        variant.numerics.accumulator_precision = "fp32";
        variant.numerics.output_precision = "bf16";
        variant.numerics.row_invariant = true;
        variant.numerics.deterministic = true;
        variant.qualification = kernels::QualificationStatus::oracle_matched;
        variant.graph_safe = true;
        variant.input_layout = variant.output_layout = "row-major";
        variant.compiler_identity = "host-fixture";
        registry.add(std::move(variant));
    }
    ExecutionBufferRequest buffers;
    buffers.logical_rows = request.logical_rows;
    return compile_serial_execution(config.model, registry, request, buffers);
}

void projection_tests() {
    using namespace strix;
    ExecutionBindingRequest request;
    // Incumbent, virtual input, prefetched input, and their output overlays.
    for (unsigned mode = 0; mode < 3; ++mode) for (const bool output : {false, true}) {
        HipGdnLauncherConfig config;
        config.virtual_q8_input_projections = mode > 0;
        config.prefetch_q8_input_projections = mode > 1;
        config.virtual_q8_output_projection = output;
        const auto id = hip_gdn_identity::kernel_id(request, config);
        require(id.find("decode-fuse") == std::string::npos, "fusion must not retag GDN kernel ids");
        require(id.ends_with(".virtual256.q8-output") == output, "output selector remains terminal");
        hip_gdn_identity::validate_projection_id(id, config);
        for (unsigned flip = 0; flip < 3; ++flip) {
            auto wrong = config;
            if (flip == 0) wrong.virtual_q8_output_projection = !output;
            if (flip == 1) wrong.virtual_q8_input_projections = !config.virtual_q8_input_projections;
            if (flip == 2) wrong.prefetch_q8_input_projections = !config.prefetch_q8_input_projections;
            bool rejected = false;
            try { hip_gdn_identity::validate_projection_id(id, wrong); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "each projection binding mismatch must still reject");
        }
    }
    // Exact incumbent ids for R1/R2/R8, C1/C2 and scalar/device-position plans.
    for (const unsigned rows : {1U, 2U, 8U}) for (const unsigned concurrency : {1U, 2U})
        for (const bool device : {false, true}) {
            request.logical_rows = rows;
            request.request_concurrency = concurrency;
            request.device_graph_inputs = device;
            HipGdnLauncherConfig config;
            const auto id = hip_gdn_identity::kernel_id(request, config);
            require(id == "hip.gdn.bf16.h2560.kh16.vh48.d128.r" + std::to_string(rows) +
                ".c" + std::to_string(concurrency) + (device ? ".device" : ".scalar"),
                "non-fusion kernel id bytes must remain unchanged");
        }
}

void binding_and_launch_identity_tests() {
    using namespace strix;
    HipGdnLauncherConfig config;
    config.virtual_q8_input_projections = config.prefetch_q8_input_projections = true;
    config.virtual_q8_output_projection = true;
    config.wave8_q8_input_projections = config.wave8_q8_output_projection = true;
    ExecutionBindingRequest request;
    request.wave8_q8_gdn_input = request.wave8_q8_gdn_output = true;
    const auto compiled = compile(config, request);
    constexpr std::string_view incumbent_id =
        "hip.gdn.bf16.h2560.kh16.vh48.d128.r1.c1.scalar.wave8-input-v1.wave8-q8-output-v1"
        ".prefetch8.virtual256.q8-input.virtual256.q8-output";
    const auto incumbent = compile(config, request, incumbent_id);
    require(compiled.execution_identity == incumbent.execution_identity &&
        compiled.kernel_plan_identity == incumbent.kernel_plan_identity,
        "actual GDN ids retain the incumbent execution and kernel-plan identities");
    const auto id = hip_gdn_identity::kernel_id(request, config);
    require(id == incumbent_id, "known-window serial id remains byte-identical");
    std::array<std::uint64_t, 2> launch_ids{};
    for (const bool fuse : {false, true}) {
        ExecutionLauncherRegistry launchers;
        for (const auto& binding : compiled.bindings.bindings) {
            if (binding.host_or_runtime_step) {
                launchers.add_runtime_launcher(binding.op, {"fixture.runtime", 1, &no_launch});
            } else if (!launchers.find_kernel_launcher(*binding.kernel_id)) {
                const bool gdn = binding.op == SerialOp::gated_delta_net;
                if (gdn) {
                    require(*binding.kernel_id == id, "host compiler/binder preserves the actual R1 GDN id");
                    hip_gdn_identity::validate_projection_id(*binding.kernel_id, config);
                }
                launchers.add_kernel_launcher({*binding.kernel_id,
                    gdn ? hip_gdn_identity::implementation_identity(config, fuse) : 1, &no_launch});
            }
        }
        const auto plan = compile_launch_plan(compiled, launchers);
        require(plan.execution_identity == compiled.execution_identity, "launch organization preserves numerical identity");
        launch_ids[fuse] = plan.plan_identity;
    }
    require(launch_ids[0] != launch_ids[1], "fusion implementation hash must distinguish compiled launch plans");

    // Independently derived FNV goldens for the existing 128-wide device fixture.
    // All fields follow the incumbent contract; ON adds GDNDFUS1 after the launcher tag.
    config = HipGdnLauncherConfig{};
    config.model.hidden_size = 128;
    config.model.max_position_embeddings = 64;
    constexpr std::array<std::uint64_t, 3> off{0x4c2e7f55e611b0edULL, 0x6df298901ec0691aULL, 0x31229ee8b0186323ULL};
    constexpr std::array<std::uint64_t, 3> on{0x5d16c53e258db5cdULL, 0xadc2224b6cf451baULL, 0xbbac397b35745543ULL};
    for (unsigned mode = 0; mode < 3; ++mode) {
        config.virtual_q8_input_projections = mode > 0;
        config.prefetch_q8_input_projections = mode > 1;
        require(hip_gdn_identity::implementation_identity(config, false) == off[mode], "OFF restores incumbent implementation hash");
        require(hip_gdn_identity::implementation_identity(config, true) == on[mode], "ON retains the fusion implementation hash");
        require(hip_gdn_identity::implementation_identity(config) == (qpf2k::decode_fuse_enabled() ? on[mode] : off[mode]),
            "launcher identity follows the resolved environment");
    }
}
} // namespace

int main(int argc, char** argv) try {
    require(argc == 2 && strix::qpf2k::decode_fuse_enabled() == (std::string_view(argv[1]) == "1"),
        "CTest must exercise the requested fusion environment");
    projection_tests();
    binding_and_launch_identity_tests();
    std::cout << "GDN host identity tests passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
