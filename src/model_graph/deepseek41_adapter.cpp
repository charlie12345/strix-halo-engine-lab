#include "deepseek41_adapter.hpp"
#include "strix/deepseek41_model_description.hpp"

#include <algorithm>
#include <map>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace strix::deepseek41 {
namespace {
namespace mg = model_graph;
std::string state_id(std::int64_t layer, const std::string& role) {
    return (layer < 0 ? "model/" : "layer/" + std::to_string(layer) + '/') + role;
}
std::string activation(Buffer buffer) { return "activation/" + std::string(to_string(buffer)); }
mg::StateRequirement state(std::string role, std::string format, std::vector<std::string> shape) {
    return {{std::move(role), std::move(format), std::move(shape), "sequence"}, "sequence"};
}
mg::OpKind kind(Op op) {
    switch (op) {
    case Op::embed: return mg::OpKind::embedding;
    case Op::hc_mix: case Op::hc_pre: case Op::hc_post: return mg::OpKind::hyper_connection;
    case Op::rms_norm: return mg::OpKind::normalization;
    case Op::q_a: case Op::q_b: case Op::kv: case Op::o_a: case Op::o_b: return mg::OpKind::projection;
    case Op::window_write: case Op::compress: return mg::OpKind::state_update;
    case Op::attention: return mg::OpKind::attention;
    case Op::router: return mg::OpKind::router;
    case Op::shared_expert: case Op::routed_experts: return mg::OpKind::expert;
    case Op::combine: return mg::OpKind::combine;
    case Op::head: return mg::OpKind::head;
    case Op::index: case Op::argmax: return mg::OpKind::selection;
    case Op::engram: return mg::OpKind::family_specific;
    }
    throw std::invalid_argument("DS41 common plan: invalid operation");
}
mg::LayerDesc block(const Config& c, std::uint64_t id, std::uint32_t ratio, bool draft) {
    mg::LayerDesc l;
    l.id = id;
    l.attention = ratio ? mg::AttentionFamily::compressed_indexed : mg::AttentionFamily::mla;
    l.semantics = {{"query_projection", "low_rank"}, {"kv_projection", "latent_multi_query"}};
    l.dimensions = {{"compression_ratio", ratio}, {"query_latent", c.q_lora_rank},
        {"output_latent", c.o_lora_rank}, {"head_dim", c.head_dim}, {"heads", c.num_attention_heads},
        {"window", c.sliding_window}, {"index_heads", c.index_n_heads}, {"index_head_dim", c.index_head_dim},
        {"index_topk", c.index_topk}, {"candidate_topk_blocks", c.candidate_topk_blocks}};
    l.hyper_connection = mg::HyperConnectionDesc{mg::HyperConnectionKind::hc, c.hc_mult, c.hc_sinkhorn_iters};
    l.mlps = {{mg::MlpKind::moe, c.moe_intermediate_size,
               draft ? c.dspark_n_routed_experts : c.n_routed_experts,
               draft ? c.dspark_num_experts_per_tok : c.num_experts_per_tok},
              {mg::MlpKind::shared_expert, c.moe_intermediate_size, c.n_shared_experts, c.n_shared_experts}};
    std::ostringstream epsilon;
    epsilon.imbue(std::locale::classic());
    epsilon << std::setprecision(std::numeric_limits<double>::max_digits10) << c.rms_norm_eps;
    l.norms = {{"rms", epsilon.str(), mg::NormPlacement::pre_attention},
               {"rms", epsilon.str(), mg::NormPlacement::pre_mlp}};
    l.states.push_back(state("window", "e4m3-ue8m0/32", {"window", "head_dim"}));
    return l;
}
}  // namespace

mg::ModelGraph model_graph_from_config(const Config& config) {
    // Reuse the engine's complete config admission; no description-supplied identity.
    const auto c = parse_config_json(model_description_config_json(config));
    const auto attrs = layer_attributes(c);
    mg::ModelGraph g;
    g.model = "deepseek-v4.1";
    g.semantic_version = "1";
    g.semantics = {{"adapter", "deepseek41-config-v1"}, {"config_contract", model_description_config_json(c)},
        {"scope", "description-only; MTP assets optional; no executor or cache identity"},
        {"token_history_update", "runner-owned; engram operations read history"}};
    g.dimensions = {{"hidden", c.hidden_size}, {"vocabulary", c.vocab_size}, {"layers", c.num_hidden_layers}};
    g.inputs = {{"token", "u32", {"1"}, "model"}};
    g.outputs = {{"logits", "f32", {"vocabulary"}, "vocabulary_partitionable"}};
    for (std::uint32_t i = 0; i < c.num_hidden_layers; ++i) {
        const auto& a = attrs[i];
        auto l = block(c, i, a.compress_ratio, false);
        l.semantics["kv_source_layer"] = std::to_string(a.kv_source_layer);
        l.semantics["index_source_layer"] = std::to_string(a.index_source_layer);
        if (a.kv_source) {
            l.states.push_back(state("compressed_kv", "e2m1-e4m3/16", {"compressed_positions", "head_dim"}));
            l.states.push_back(state("index_keys", "e2m1-e8m0/32", {"compressed_positions", "index_head_dim"}));
        }
        if (a.compressor_gate) l.states.push_back(state("compress_carry", "f32", {"2", "compression_ratio", "head_dim"}));
        if (a.index_source) l.states.push_back(state("selection", "u32", {"index_topk"}));
        if (a.candidate_source) l.states.push_back(state("candidates", "u32", {"candidate_topk_blocks"}));
        if (a.engram) l.semantics["token_history_reader"] = "true";
        g.layers.push_back(std::move(l));
        const auto target = "layer/" + std::to_string(i);
        if (i > 0) g.edges.push_back({"layer/" + std::to_string(i - 1U), target, mg::EdgeKind::data});
        for (const auto source : {a.kv_source_layer, a.index_source_layer})
            if (source >= 0 && source != static_cast<std::int32_t>(i))
                g.edges.push_back({"layer/" + std::to_string(source), target, mg::EdgeKind::cross_layer});
        if (a.uses_candidates && c.candidate_source_layer_id != i)
            g.edges.push_back({"layer/" + std::to_string(c.candidate_source_layer_id), target, mg::EdgeKind::cross_layer});
    }
    if (!c.engram_layer_ids.empty()) {
        g.dimensions["history_tokens"] = c.engram_max_ngram_size - 1U;
        g.states.push_back(state("token_history", "u32", {"history_tokens"}));
    }
    for (const auto& t : expected_target_tensors(c)) {
        mg::TensorRequirement requirement;
        requirement.role = t.name; requirement.format = "ggml-type/" + std::to_string(t.ggml_type);
        requirement.ownership = t.role == TensorRole::moe_expert ? "expert_partitionable" : "model";
        for (const auto extent : t.dimensions) requirement.shape.push_back(std::to_string(extent));
        if (t.layer < 0) g.tensors.push_back(std::move(requirement));
        else g.layers[static_cast<std::size_t>(t.layer)].tensors.push_back(std::move(requirement));
    }
    if (c.num_nextn_predict_layers > 0) {
        mg::AttachmentDesc a;
        a.kind = mg::AttachmentKind::mtp; a.name = "next_token_predictor"; a.target = "trunk";
        for (std::uint32_t i = 0; i < c.num_nextn_predict_layers; ++i)
            a.layers.push_back(block(c, std::uint64_t{c.num_hidden_layers} + i,
                                    c.compress_ratios[std::size_t{c.num_hidden_layers} + i], true));
        for (const auto& t : expected_draft_tensors(c)) {
            mg::TensorRequirement requirement;
            requirement.role = t.name; requirement.format = "ggml-type/" + std::to_string(t.ggml_type);
            requirement.ownership = "attachment";
            for (const auto extent : t.dimensions) requirement.shape.push_back(std::to_string(extent));
            a.layers[static_cast<std::size_t>(t.layer)].tensors.push_back(std::move(requirement));
        }
        g.attachments.push_back(std::move(a));
    }
    mg::validate(g);
    return g;
}

mg::ExecutionPlan common_decode_plan(const Plan& source, const mg::ModelGraph& graph, std::uint32_t rank) {
    if (source.ranks != 2 || rank >= source.ranks || source.expert_split != source.config.n_routed_experts / 2U ||
        source.first_layer >= source.end_layer ||
        source.end_layer > source.config.num_hidden_layers ||
        mg::identity(graph) != mg::identity(model_graph_from_config(source.config)))
        throw std::invalid_argument("DS41 common plan: config/graph/rank mismatch");
    const auto& c = source.config;
    const auto attrs = layer_attributes(c);
    std::map<std::uint32_t, const LayerState*> state_sizes;
    for (const auto& s : source.state)
        if (s.layer < source.first_layer || s.layer >= source.end_layer || !state_sizes.emplace(s.layer, &s).second)
            throw std::invalid_argument("DS41 common plan: invalid state owner");
    if (state_sizes.size() != source.end_layer - source.first_layer)
        throw std::invalid_argument("DS41 common plan: missing layer state");
    mg::ExecutionPlan out;
    out.graph_identity = mg::identity(graph); out.source = "deepseek41/build_decode_plan/logical-rank-view";
    out.rank = rank; out.ranks = source.ranks; out.first_layer = source.first_layer; out.end_layer = source.end_layer;
    out.owners = {std::string(to_string(Owner::formats_moe)), std::string(to_string(Owner::attention)),
                  std::string(to_string(Owner::layer)), std::string(to_string(Owner::integration))};
    std::map<std::string, std::size_t> buffers;
    const auto touch = [&](mg::PlanBuffer b, std::uint64_t step, bool write) {
        const auto [it, fresh] = buffers.emplace(b.id, out.buffers.size());
        if (fresh) {
            b.first_use = step; b.live_in = !write; b.rank = rank;
            out.buffers.push_back(std::move(b));
        }
        auto& stored = out.buffers[it->second];
        stored.last_use = step;
    };
    for (std::size_t i = 0; i < source.steps.size(); ++i) {
        const auto& s = source.steps[i];
        if (s.layer < -1 || (s.layer >= 0 && (static_cast<std::uint64_t>(s.layer) < source.first_layer ||
                                             static_cast<std::uint64_t>(s.layer) >= source.end_layer)))
            throw std::invalid_argument("DS41 common plan: invalid operation layer");
        mg::PlanNode n;
        n.kind = kind(s.op); n.operation = to_string(s.op); n.site = s.site; n.owner = to_string(s.owner);
        n.layer = s.layer; n.rank = rank; n.arithmetic_identity = s.identity; n.weights = s.weights;
        if (i > 0) n.dependencies.push_back(i - 1U);
        const auto activation_set = [&](const auto& values, auto& ids, bool write) {
            for (const auto value : values) {
                mg::PlanBuffer b;
                b.id = activation(value); b.bytes = std::uint64_t{source.buffer_elements(value)} * 4U;
                ids.push_back(b.id); touch(std::move(b), i, write);
            }
        };
        activation_set(s.inputs, n.inputs, false); activation_set(s.outputs, n.outputs, true);
        const auto effect = [&](std::int64_t layer, const std::string& role, bool read, bool write) {
            mg::PlanBuffer b;
            b.id = state_id(layer, role); b.layer = layer; b.state_role = role; b.lifetime = mg::BufferLifetime::sequence;
            b.external = layer >= 0 && (layer < source.first_layer || layer >= source.end_layer);
            b.live_out = true;
            if (role == "window") b.bytes = std::uint64_t{c.sliding_window} * (c.head_dim + c.head_dim / 32U);
            else if (role == "compressed_kv") b.bytes_per_position = c.head_dim / 2U + c.head_dim / 16U;
            else if (role == "index_keys") b.bytes_per_position = c.index_head_dim / 2U + c.index_head_dim / 32U;
            else if (role == "compress_carry") b.bytes = 2ULL * attrs[static_cast<std::size_t>(layer)].compress_ratio * c.head_dim * 4U;
            else if (role == "selection") b.bytes = std::uint64_t{c.index_topk} * 4U;
            else if (role == "candidates") b.bytes = std::uint64_t{c.candidate_topk_blocks} * 4U;
            else if (role == "token_history") b.bytes = std::uint64_t{c.engram_max_ngram_size - 1U} * 4U;
            // For selected layers copy the authoritative planner's sizes verbatim.
            // Only external source states require geometry from the admitted config.
            if (layer >= 0 && !b.external) {
                const auto& sizes = *state_sizes.at(static_cast<std::uint32_t>(layer));
                if (role == "window") b.bytes = sizes.window_ring_bytes;
                else if (role == "compress_carry") b.bytes = sizes.compress_carry_bytes;
                else if (role == "compressed_kv") b.bytes_per_position = sizes.kv_bytes_per_position;
                else if (role == "index_keys") b.bytes_per_position = sizes.index_key_bytes_per_position;
            }
            if (read) n.state_reads.push_back(b.id);
            if (write) n.state_writes.push_back(b.id);
            touch(std::move(b), i, write && !read);
        };
        if (s.layer >= 0) {
            const auto& a = attrs[static_cast<std::size_t>(s.layer)];
            if (s.op == Op::engram) effect(-1, "token_history", true, false);
            if (s.op == Op::window_write) effect(s.layer, "window", true, true);
            if (s.op == Op::compress) {
                effect(s.layer, "compressed_kv", true, true); effect(s.layer, "index_keys", true, true);
                if (a.compressor_gate) effect(s.layer, "compress_carry", true, true);
            }
            if (s.op == Op::index) {
                effect(a.kv_source_layer, "index_keys", true, false);
                effect(s.layer, "selection", false, true);
                if (a.uses_candidates) effect(c.candidate_source_layer_id, "candidates", true, false);
                if (a.candidate_source) effect(s.layer, "candidates", false, true);
            }
            if (s.op == Op::attention) {
                effect(s.layer, "window", true, false);
                if (a.compress_ratio > 0) {
                    effect(a.kv_source_layer, "compressed_kv", true, false);
                    effect(a.index_source_layer, "selection", true, false);
                }
            }
            if (s.op == Op::routed_experts) {
                n.expert_begin = rank * source.expert_split;
                n.expert_end = rank + 1U == source.ranks ? c.n_routed_experts : n.expert_begin + source.expert_split;
            }
        }
        out.nodes.push_back(std::move(n));
    }
    for (auto& b : out.buffers) if (b.id == activation(Buffer::stream) || b.id == activation(Buffer::pre_carry) ||
                                  b.id == activation(Buffer::token) || b.id == activation(Buffer::logits)) b.live_out = true;
    mg::validate(out, graph);
    return out;
}
}  // namespace strix::deepseek41
