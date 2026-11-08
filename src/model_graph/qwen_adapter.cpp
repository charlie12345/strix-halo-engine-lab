#include "qwen_adapter.hpp"

#include <array>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>

namespace strix::qwen_description {
namespace {
namespace mg = model_graph;
std::string scalar(float value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    return out.str();
}
std::string boolean(bool value) { return value ? "true" : "false"; }
mg::StateRequirement state(std::string role, std::string format, std::vector<std::string> shape) {
    return {{std::move(role), std::move(format), std::move(shape), "sequence-local"}, "sequence"};
}
mg::LayerDesc layer(const ModelContract& m, std::uint64_t id, bool qsa) {
    const auto dim = [](std::uint64_t n) { return std::to_string(n); };
    mg::LayerDesc out;
    out.id = id;
    out.block = qsa ? mg::BlockKind::transformer : mg::BlockKind::recurrent;
    out.attention = qsa ? mg::AttentionFamily::qsa : mg::AttentionFamily::gdn;
    out.hyper_connection = mg::HyperConnectionDesc{mg::HyperConnectionKind::hc, m.hc_count, 0};
    out.semantics = {{"hyper_connection_operations", "mix,inject"},
        {"router_normalize_topk", boolean(m.norm_topk_probability)}};
    out.dimensions = {{"hidden", m.hidden_size}, {"hc_lowrank", m.hc_lowrank}};
    out.norms = {{"rmsnorm", scalar(m.rms_norm_epsilon), mg::NormPlacement::pre_attention},
                 {"rmsnorm", scalar(m.rms_norm_epsilon), mg::NormPlacement::pre_mlp}};
    out.mlps = {{mg::MlpKind::moe, m.moe_intermediate_size, m.num_experts, m.num_experts_per_token},
                {mg::MlpKind::shared_expert, m.shared_expert_intermediate_size, 1, 1}};
    if (qsa) {
        out.dimensions.insert({{"attention_heads", m.attention_heads}, {"kv_heads", m.kv_heads},
            {"head_dim", m.head_dim}, {"index_heads", m.indexer_n_heads},
            {"index_kv_heads", m.indexer_kv_heads}, {"index_head_dim", m.indexer_head_dim},
            {"index_budget", m.indexer_budget}, {"index_compress_ratio", m.indexer_compress_ratio}});
        const std::string kv = m.attention_kv_storage == AttentionKvStorage::bf16 ? "bf16" : "int8-token";
        out.states = {state("attention_kv", kv, {"positions", "2", dim(m.kv_heads), dim(m.head_dim)}),
            state("sparse_index", "family-owned", {"compressed_positions", dim(m.indexer_kv_heads), dim(m.indexer_head_dim)})};
        out.semantics["attention_equations"] = "qwen-indexed-attention";
    } else {
        out.dimensions.insert({{"key_heads", m.gdn_key_heads}, {"key_dim", m.gdn_key_head_dim},
            {"value_heads", m.gdn_value_heads}, {"value_dim", m.gdn_value_head_dim},
            {"conv_kernel", m.gdn_conv_kernel}});
        out.semantics.insert({{"output_gate_sigmoid", boolean(m.gdn_output_gate_sigmoid)},
            {"qk_l2norm", boolean(m.gdn_qk_l2norm)}, {"qk_l2_epsilon", scalar(m.gdn_qk_l2_epsilon)},
            {"conv_silu", boolean(m.gdn_conv_silu)}});
        out.states = {state("recurrent_matrix", "fp32", {dim(m.gdn_value_heads), dim(m.gdn_key_head_dim), dim(m.gdn_value_head_dim)}),
            state("convolution_history", "family-owned", {dim(m.gdn_conv_kernel - 1), "gdn-conv-channels"})};
    }
    return out;
}
mg::ModelGraph adapt(const GraphTemplate& source) {
    const auto& m = source.contract;
    mg::ModelGraph graph;
    graph.model = "qwen3.8-flash-next";
    graph.semantic_version = "qwen-description/v1";
    graph.semantics = {{"source", "qwen_model_description"},
        {"arithmetic", "engine-owned; descriptive identity only"},
        {"hyper_connection_operations", "expand,mix,inject,collapse"},
        {"rotary_theta", scalar(m.rope_theta)}, {"partial_rotary_factor", scalar(m.partial_rotary_factor)},
        {"mtp_dedicated_embeddings", boolean(m.mtp_use_dedicated_embeddings)},
        {"mtp_explicit_hidden_source", boolean(m.mtp_hidden_state_source_layer_explicit)}};
    graph.dimensions = {{"hidden", m.hidden_size}, {"vocabulary", m.vocab_size},
        {"max_positions", m.max_position_embeddings}, {"full_attention_interval", m.full_attention_interval},
        {"streams", m.hc_count}, {"bytes_bf16", m.bytes_bf16}, {"bytes_fp32", m.bytes_fp32}};
    graph.inputs = {{"tokens", "u32", {"tokens"}, "replicated"}};
    graph.outputs = {{"logits", "fp32", {std::to_string(m.vocab_size)}, "vocabulary-partitionable"}};
    graph.tensors = {{"token_embedding", "family-owned", {std::to_string(m.vocab_size), std::to_string(m.hidden_size)}, "replicated"}};
    graph.states = {state("position_history", "family-owned", {"positions"}),
        state("ngram_history", "u32", {std::to_string(m.ngram_size - 1)})};
    for (const auto& entry : source.layers) {
        auto item = layer(m, entry.index, entry.attention == AttentionKind::qsa);
        if (entry.ple) {
            item.states.push_back(state("ple_convolution", "family-owned",
                {std::to_string(m.ple_conv_kernel_size - 1), std::to_string(m.ple_embed_dim)}));
            item.dimensions.insert({{"ple_embed_dim", m.ple_embed_dim}, {"ple_conv_kernel", m.ple_conv_kernel_size},
                {"ngram_size", m.ngram_size}, {"heads_per_ngram", m.heads_per_ngram},
                {"ngram_vocab_base", m.ngram_vocab_size_base}, {"ngram_vocab_divisor", m.ngram_vocab_divisor},
                {"ngram_hash_seed", m.ngram_hash_seed}, {"ple_eos_token", m.ple_eos_token_id}});
            mg::AttachmentDesc ple;
            ple.kind = mg::AttachmentKind::ple;
            ple.name = "ple";
            ple.target = "layer." + std::to_string(entry.index) + ".before-attention";
            graph.attachments.push_back(std::move(ple));
        }
        graph.layers.push_back(std::move(item));
        if (entry.index > 0) graph.edges.push_back({"layer." + std::to_string(entry.index - 1),
            "layer." + std::to_string(entry.index), mg::EdgeKind::hyper_connection});
    }
    if (m.mtp_full_attention_layers != 0) {
        mg::AttachmentDesc mtp;
        mtp.kind = mg::AttachmentKind::mtp;
        mtp.name = "mtp";
        mtp.target = "trunk.hidden";
        for (std::uint64_t i = 0; i < m.mtp_full_attention_layers; ++i) {
            mg::LayerDesc sidecar;
            sidecar.id = m.hidden_layers + i;
            sidecar.block = mg::BlockKind::auxiliary;
            sidecar.attention = mg::AttentionFamily::qsa;
            sidecar.dimensions = {{"hidden", m.hidden_size}, {"attention_heads", m.attention_heads},
                {"kv_heads", m.kv_heads}, {"head_dim", m.head_dim}};
            // The description configures full-attention sidecar assets. Its
            // internal predictor schedule is not inferred from a trunk block.
            sidecar.semantics["schedule"] = "sidecar; not executed by trunk serial plan";
            mtp.layers.push_back(std::move(sidecar));
        }
        graph.attachments.push_back(std::move(mtp));
    }
    mg::validate(graph);
    return graph;
}
mg::OpKind kind(SerialOp op) {
    switch (op) {
    case SerialOp::token_embedding: return mg::OpKind::embedding;
    case SerialOp::position_prepare: return mg::OpKind::state_update;
    case SerialOp::hyper_expand: case SerialOp::attn_hyper_mix: case SerialOp::attn_inject:
    case SerialOp::mlp_hyper_mix: case SerialOp::mlp_inject: case SerialOp::final_hyper_mix:
        return mg::OpKind::hyper_connection;
    case SerialOp::gated_delta_net: case SerialOp::indexed_attention: return mg::OpKind::attention;
    case SerialOp::moe_route: return mg::OpKind::router;
    case SerialOp::moe_experts: case SerialOp::shared_expert: return mg::OpKind::expert;
    case SerialOp::moe_combine: return mg::OpKind::combine;
    case SerialOp::lm_head: return mg::OpKind::head;
    case SerialOp::ple: return mg::OpKind::family_specific;
    }
    throw std::invalid_argument("Qwen projection: unknown operation");
}
constexpr std::array state_bits{PersistentState::attention_kv, PersistentState::recurrent_matrix,
    PersistentState::convolution_history, PersistentState::sparse_index,
    PersistentState::ngram_history, PersistentState::position_history};
constexpr std::array roles{"attention_kv", "recurrent_matrix", "convolution_history",
    "sparse_index", "ngram_history", "position_history"};
}  // namespace

model_graph::ModelGraph model_graph_from_description(const model_description::ModelDescription& file) {
    return adapt(graph_template(file));
}
model_graph::ModelGraph model_graph_from_contract(const ModelContract& model) {
    return adapt(graph_template(model));
}
model_graph::ExecutionPlan common_decode_plan(const SerialExecutionPlan& source, const model_graph::ModelGraph& graph) {
    mg::validate(graph);
    mg::ExecutionPlan plan;
    plan.graph_identity = mg::identity(graph);
    plan.source = "build_serial_decode_execution_plan";
    plan.ranks = 1;
    plan.end_layer = graph.layers.size();
    plan.owners = {"qwen-serial"};
    std::map<std::string, std::size_t> storage;
    for (const auto& step : source.steps) {
        if (step.id != plan.nodes.size()) throw std::invalid_argument("Qwen projection: nonordinal step id");
        constexpr std::uint32_t known_state_bits = (1U << 6U) - 1U;
        if (((static_cast<std::uint32_t>(step.reads_state) | static_cast<std::uint32_t>(step.writes_state)) &
             ~known_state_bits) != 0U) throw std::invalid_argument("Qwen projection: unknown state effect");
        mg::PlanNode node;
        node.kind = kind(step.op);
        node.operation = to_string(step.op);
        node.site = "serial." + std::to_string(step.id);
        node.owner = "qwen-serial";
        node.arithmetic_identity = "unchanged-engine-owned";
        node.layer = step.layer ? static_cast<std::int64_t>(*step.layer) : -1;
        node.dependencies.assign(step.dependencies.begin(), step.dependencies.end());
        node.graph_safe_candidate = step.graph_safe_candidate;
        node.dynamic_shape = step.dynamic_shape;
        for (std::size_t i = 0; i < state_bits.size(); ++i) {
            const bool read = has_state(step.reads_state, state_bits[i]);
            const bool write = has_state(step.writes_state, state_bits[i]);
            if (!read && !write) continue;
            std::string role = roles[i];
            if (step.op == SerialOp::ple && state_bits[i] == PersistentState::convolution_history)
                role = "ple_convolution";
            const bool global = state_bits[i] == PersistentState::ngram_history || state_bits[i] == PersistentState::position_history;
            const auto owner_layer = global ? -1 : node.layer;
            const std::string id = (global ? "sequence" : "layer." + std::to_string(owner_layer)) + "." + role;
            if (!storage.contains(id)) {
                mg::PlanBuffer buffer;
                buffer.id = id;
                buffer.lifetime = mg::BufferLifetime::sequence;
                buffer.layer = owner_layer;
                buffer.state_role = role;
                buffer.first_use = step.id;
                buffer.live_in = buffer.live_out = true;
                buffer.extent = "graph-state-shape; byte sizing remains engine-owned";
                storage.emplace(id, plan.buffers.size());
                plan.buffers.push_back(std::move(buffer));
            }
            plan.buffers[storage.at(id)].last_use = step.id;
            if (read) node.state_reads.push_back(id);
            if (write) node.state_writes.push_back(id);
        }
        plan.nodes.push_back(std::move(node));
    }
    mg::validate(plan, graph);
    return plan;
}
void validate_decode_projection(const ModelContract& model, const SerialExecutionPlan& source,
    const model_graph::ModelGraph& graph, const model_graph::ExecutionPlan& plan) {
    const auto expected_graph = model_graph_from_contract(model);
    if (mg::canonical(graph) != mg::canonical(expected_graph))
        throw std::invalid_argument("Qwen projection: graph differs from model contract");
    const auto expected_source = build_serial_decode_execution_plan(model);
    if (source.linear_attention_steps != expected_source.linear_attention_steps ||
        source.indexed_attention_steps != expected_source.indexed_attention_steps ||
        source.ple_steps != expected_source.ple_steps || source.moe_route_steps != expected_source.moe_route_steps ||
        source.moe_expert_steps != expected_source.moe_expert_steps ||
        mg::dump(common_decode_plan(source, graph)) != mg::dump(common_decode_plan(expected_source, expected_graph)))
        throw std::invalid_argument("Qwen projection: source differs from serial schedule");
    mg::validate(plan, graph);
    if (mg::dump(plan) != mg::dump(common_decode_plan(expected_source, expected_graph)))
        throw std::invalid_argument("Qwen projection: plan differs from serial schedule");
}
}  // namespace strix::qwen_description
