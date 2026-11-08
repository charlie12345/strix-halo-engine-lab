#include "model_graph.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace strix::minimax::graph {
namespace {
namespace mg = model_graph;
using Description = model_description::ModelDescription;
std::uint64_t dim(const Description& d, const char* key) {
    return static_cast<std::uint64_t>(std::get<std::int64_t>(d.find(key)->data));
}
std::string number(const Description& d, const char* key) {
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<double>::max_digits10) << std::get<double>(d.find(key)->data);
    return out.str();
}
} // namespace

mg::ModelGraph build_model_graph(const Description& d, Attachments attachments) {
    description::validate(d);  // rejects unsupported families, dimensions and unknown fields
    if (!attachments.draft && (attachments.draft_linears || attachments.draft_heads != 0))
        throw std::invalid_argument("MiniMax graph: draft assets require a draft attachment");
    mg::ModelGraph g;
    g.model = std::get<std::string>(d.find("name")->data);
    g.semantic_version = "minimax-description/v1";
    g.semantics = {{"adapter", "minimax-description"}, {"attention", "msa"},
        {"normalization", "gemma_rmsnorm: (x*rsqrt(mean(x*x)+eps))*(1+w)"},
        {"math.eps", number(d, "math.eps")}, {"math.alpha", number(d, "math.alpha")},
        {"math.limit", number(d, "math.limit")}, {"mlp.activation", "limited_swiglu"},
        {"router", "sigmoid+bias-choice;top-k:score-desc,id-asc;normalized-weight*2"},
        {"expert.combine", "ascending-expert-id-fma;shared-unscaled"}};
    for (const char* key : {"dims.hidden", "dims.q", "dims.kv", "dims.index_q", "dims.head_dim",
                           "dims.n_heads", "dims.n_kv_heads", "dims.index_heads", "dims.layers",
                           "dims.intermediate", "dims.vocab", "moe.first_layer", "moe.experts",
                           "moe.top_k", "moe.intermediate"}) g.dimensions[key] = dim(d, key);
    // vocab_rank_pad is physical padding, not a model dimension.
    g.inputs = {{"tokens", "u32", {"rows"}, "replicable"}};
    g.outputs = {{"logits", "f32", {"rows", "dims.vocab"}, "vocabulary-partitionable"}};
    g.tensors = {{"embedding", "source-linear", {"dims.vocab", "dims.hidden"}, "replicable"},
                 {"final_norm", "f32", {"dims.hidden"}, "replicable"},
                 {"head", "source-linear", {"dims.vocab", "dims.hidden"}, "vocabulary-partitionable"}};
    for (std::uint64_t id = 0; id < dim(d, "dims.layers"); ++id) {
        mg::LayerDesc l;
        l.id = id;
        l.attention = mg::AttentionFamily::msa;
        l.norms = {{"gemma_rmsnorm", "math.eps", mg::NormPlacement::pre_attention},
                   {"gemma_rmsnorm", "math.eps", mg::NormPlacement::pre_mlp}};
        l.semantics = {{"attention.ownership", "query/kv/index-query-head-partitionable;index-key-replicable"},
            {"attention.head_norm", "per-head-gemma-rmsnorm:q,k,index_q,index_k"},
            {"attention.rotary", "neox;rotary-dim=64"},
            {"attention.indexer", "max-qk-per-128-position-block;top-16;newest-block-forced"},
            {"attention.grouping", "16-query-heads-per-kv-head"},
            {"mlp.ownership", "intermediate-partitionable;all-expert-ids-present"}};
        l.dimensions = {{"query_heads", dim(d, "dims.n_heads")}, {"kv_heads", dim(d, "dims.n_kv_heads")},
                        {"index_heads", dim(d, "dims.index_heads")}, {"head_dim", dim(d, "dims.head_dim")},
                        {"index_key_dim", 128}, {"index_block", 128}, {"index_top_k", 16}};
        l.tensors = {{"qkv", "source-linear", {"dims.hidden", "dims.q+2*dims.kv"}, "head-partitionable"},
            {"index_q", "source-linear", {"dims.hidden", "dims.index_q"}, "head-partitionable"},
            {"index_k", "source-linear", {"dims.hidden", "128"}, "replicable"},
            {"head_norms", "f32", {"4", "dims.head_dim"}, "replicable"},
            {"attention_output", "source-linear", {"dims.q", "dims.hidden"}, "input-head-partitionable"},
            {"input_norm", "f32", {"dims.hidden"}, "replicable"},
            {"post_attention_norm", "f32", {"dims.hidden"}, "replicable"}};
        const bool moe = id >= dim(d, "moe.first_layer");
        if (moe) {
            l.mlps = {{mg::MlpKind::moe, dim(d, "moe.intermediate"), dim(d, "moe.experts"), dim(d, "moe.top_k")},
                      {mg::MlpKind::shared_expert, dim(d, "moe.intermediate"), 1, 1}};
            l.tensors.push_back({"router", "f32", {"moe.experts", "dims.hidden"}, "replicable"});
            l.tensors.push_back({"router_bias", "f32", {"moe.experts"}, "replicable"});
        } else l.mlps = {{mg::MlpKind::dense, dim(d, "dims.intermediate"), 0, 0}};
        l.tensors.push_back({moe ? "experts" : "dense_mlp", "source-linear",
                            {moe ? "moe.experts" : "1", "dims.hidden", moe ? "moe.intermediate" : "dims.intermediate"},
                            "intermediate-partitionable"});
        if (moe) l.tensors.push_back({"shared_expert", "source-linear", {"dims.hidden", "moe.intermediate"}, "intermediate-partitionable"});
        l.states = {{{"key_cache", "bf16", {"positions", "dims.kv"}, "head-partitionable"}, "sequence"},
                    {{"value_cache", "bf16", {"positions", "dims.kv"}, "head-partitionable"}, "sequence"},
                    {{"index_key_cache", "bf16", {"positions", "128"}, "replicable"}, "sequence"}};
        const std::string here = "layer." + std::to_string(id);
        g.edges.push_back({id == 0 ? "input" : "layer." + std::to_string(id - 1), here, mg::EdgeKind::data});
        g.edges.push_back({here + ".input", here + ".attention_output", mg::EdgeKind::residual});
        g.edges.push_back({here + ".post_attention", here + ".mlp_output", mg::EdgeKind::residual});
        g.layers.push_back(std::move(l));
    }
    g.edges.push_back({"layer." + std::to_string(g.layers.size() - 1), "output", mg::EdgeKind::data});
    if (attachments.draft) {
        g.attachments.push_back({mg::AttachmentKind::draft, "configured-drafter", "trunk-taps+head", {}});
        g.semantics["draft.structure"] = "engine-local;attachment-inventory-only";
        g.semantics["draft.linears"] = attachments.draft_linears ? "configured" : "target";
        g.dimensions["draft.heads"] = attachments.draft_heads;
    }
    // This model description has no MTP block. Do not infer one from a draft pack.
    mg::validate(g);
    return g;
}

void assert_matches(const Description& d, const mg::ModelGraph& g, Attachments attachments) {
    mg::validate(g);
    if (mg::canonical(g) != mg::canonical(build_model_graph(d, attachments)))
        throw std::invalid_argument("MiniMax graph/description mismatch");
}

mg::ExecutionPlan common_decode_plan(const Description& d, const mg::ModelGraph& g,
                                    std::uint32_t rank, const defaults::Lookup& environment, Attachments attachments) {
    assert_matches(d, g, attachments);
    if (rank >= 2) throw std::invalid_argument("MiniMax plan: TP2 rank must be 0 or 1");
    mg::ExecutionPlan p;
    p.graph_identity = mg::identity(g);
    p.source = "MiniMax layers_range/stage_attn/stage_ffn: logical C1 decode and greedy head boundary";
    p.rank = rank; p.ranks = 2; p.end_layer = g.layers.size();
    p.owners = {"minimax.embedding", "minimax.norm", "minimax.linear", "minimax.msa", "minimax.moe", "minimax.exchange", "minimax.head"};
    p.policies = {{"scope", "logical-one-row;not-device-admission-or-dispatch"},
        {"attention.head_begin", std::to_string(rank * 32U)}, {"attention.head_end", std::to_string((rank + 1U) * 32U)},
        {"attention.kv_head_begin", std::to_string(rank * 2U)}, {"attention.kv_head_end", std::to_string((rank + 1U) * 2U)},
        {"attention.index_head_begin", std::to_string(rank * 2U)}, {"attention.index_head_end", std::to_string((rank + 1U) * 2U)},
        {"mlp.intermediate_partition", "equal-width-half;all-expert-ids-on-both-ranks"},
        {"head.vocab_begin", std::to_string(rank == 0 ? 0 : dim(d, "dims.vocab_rank_pad"))},
        {"head.vocab_end", std::to_string(rank == 0 ? dim(d, "dims.vocab_rank_pad") : dim(d, "dims.vocab"))}};
    // Pure resolver values are declarations, not measured active kernel choices.
    for (const auto& item : defaults::resolve(environment)) p.policies[std::string("resolved.") + item.item->env] = item.value;
    p.engine_local = {"kernel/fusion selection, arithmetic reduction trees and weight layout",
        "streams/events/index overlap, shared-first fallback ordering, host routing and dispatch tables",
        "transport protocol, exchange wait/publish, physical scratch/cache allocation, capacity and graph capture",
        "prefill/chunking/nested-serving scheduler and draft/verify/acceptance schedules"};
    const auto hidden = dim(d, "dims.hidden");
    const auto qkv_width = (dim(d, "dims.q") + 2 * dim(d, "dims.kv")) / 2;
    const auto index_width = dim(d, "dims.index_q") / 2 + 128;
    auto buffer = [&](const std::string& id, std::uint64_t width) {
        mg::PlanBuffer b;
        b.id = id; b.bytes = width * sizeof(float); b.rank = rank;
        b.first_use = std::numeric_limits<std::uint64_t>::max();
        p.buffers.push_back(std::move(b));
    };
    auto node = [&](mg::OpKind kind, const std::string& operation, std::int64_t layer,
                    const std::string& owner, std::vector<std::string> inputs, std::uint64_t width,
                    std::vector<std::string> weights) -> std::string {
        mg::PlanNode n;
        n.kind = kind; n.operation = operation; n.layer = layer; n.rank = rank; n.owner = owner;
        n.site = layer < 0 ? "global" : "layer." + std::to_string(layer);
        n.arithmetic_identity = "engine-owned:MiniMax-G4-decode;descriptive-only";
        n.graph_safe_candidate = false;
        n.inputs = std::move(inputs); n.weights = std::move(weights);
        if (!p.nodes.empty()) n.dependencies.push_back(p.nodes.size() - 1);
        const std::string output = "n" + std::to_string(p.nodes.size());
        if (width != 0) { buffer(output, width); n.outputs.push_back(output); }
        p.nodes.push_back(std::move(n));
        return output;
    };
    std::string residual = node(mg::OpKind::embedding, "embedding", -1, "minimax.embedding", {}, hidden, {"embedding"});
    for (const auto& layer : g.layers) {
        const auto id = static_cast<std::int64_t>(layer.id);
        const std::string norm = node(mg::OpKind::normalization, "input_gemma_rmsnorm", id, "minimax.norm", {residual}, hidden, {"input_norm"});
        const std::string qkv = node(mg::OpKind::projection, "head_split_qkv", id, "minimax.linear", {norm}, qkv_width, {"qkv"});
        const std::string index = node(mg::OpKind::projection, "index_query_and_replicated_key", id, "minimax.linear", {norm}, index_width, {"index_q", "index_k"});
        // A logical topological order; no claim that independent projections are serialized on device.
        const std::string hn = node(mg::OpKind::normalization, "per_head_gemma_rmsnorm", id, "minimax.norm", {qkv, index}, qkv_width + index_width, {"head_norms"});
        const std::string rope = node(mg::OpKind::family_specific, "neox_rope", id, "minimax.msa", {hn}, qkv_width + index_width, {});
        std::vector<std::string> states;
        for (const auto& state : layer.states) {
            mg::PlanBuffer b;
            b.id = "layer." + std::to_string(id) + '.' + state.tensor.role;
            b.layer = id; b.rank = rank; b.state_role = state.tensor.role;
            b.lifetime = mg::BufferLifetime::sequence; b.live_in = true; b.live_out = true;
            b.bytes_per_position = (state.tensor.role == "index_key_cache" ? 128 : dim(d, "dims.kv") / 2) * 2;
            b.first_use = std::numeric_limits<std::uint64_t>::max();
            states.push_back(b.id); p.buffers.push_back(std::move(b));
        }
        (void)node(mg::OpKind::state_update, "append_kv_index", id, "minimax.msa", {rope}, 0, {});
        p.nodes.back().state_writes = states;
        const std::string scores = node(mg::OpKind::attention, "index_block_scores", id, "minimax.msa", {rope}, 1, {});
        p.nodes.back().state_reads = {states[2]};
        const std::string selected = node(mg::OpKind::selection, "top16_blocks_newest_forced", id, "minimax.msa", {scores}, 1, {});
        p.nodes.back().dynamic_shape = true;
        const std::string attended = node(mg::OpKind::attention, "selected_block_grouped_query_attention", id, "minimax.msa", {rope, selected}, dim(d, "dims.q") / 2, {});
        p.nodes.back().state_reads = {states[0], states[1]};
        const std::string partial = node(mg::OpKind::projection, "output_input_head_slice", id, "minimax.linear", {attended}, hidden, {"attention_output"});
        residual = node(mg::OpKind::family_specific, "X1.fixed_rank_order_residual", id, "minimax.exchange", {residual, partial}, hidden, {});
        const std::string post = node(mg::OpKind::normalization, "post_attention_gemma_rmsnorm", id, "minimax.norm", {residual}, hidden, {"post_attention_norm"});
        std::string mlp;
        if (layer.id < dim(d, "moe.first_layer")) {
            mlp = node(mg::OpKind::expert, "dense_limited_swiglu_intermediate_half", id, "minimax.moe", {post}, hidden, {"dense_mlp"});
        } else {
            const std::string routes = node(mg::OpKind::router, "sigmoid_bias_top4", id, "minimax.moe", {post}, dim(d, "moe.top_k") * 2, {"router", "router_bias"});
            const std::string experts = node(mg::OpKind::expert, "routed_limited_swiglu_intermediate_half", id, "minimax.moe", {post, routes}, hidden, {"experts"});
            p.nodes.back().expert_end = static_cast<std::uint32_t>(dim(d, "moe.experts"));
            p.nodes.back().dynamic_shape = true;
            const std::string shared = node(mg::OpKind::expert, "shared_limited_swiglu_intermediate_half", id, "minimax.moe", {post}, hidden, {"shared_expert"});
            mlp = node(mg::OpKind::combine, "ascending_expert_fma_then_shared", id, "minimax.moe", {experts, routes, shared}, hidden, {});
        }
        residual = node(mg::OpKind::family_specific, "X2.fixed_rank_order_residual", id, "minimax.exchange", {residual, mlp}, hidden, {});
    }
    const std::string normalized = node(mg::OpKind::normalization, "final_gemma_rmsnorm", -1, "minimax.norm", {residual}, hidden, {"final_norm"});
    const auto vocab_begin = rank == 0 ? 0 : dim(d, "dims.vocab_rank_pad");
    const auto vocab_end = rank == 0 ? dim(d, "dims.vocab_rank_pad") : dim(d, "dims.vocab");
    const std::string logits = node(mg::OpKind::head, "vocabulary_shard_logits", -1, "minimax.head", {normalized}, vocab_end - vocab_begin, {"head"});
    (void)node(mg::OpKind::selection, "greedy_argmax_pair_exchange", -1, "minimax.head", {logits}, 1, {});
    p.buffers.back().live_out = true;
    // Symbolic dynamic scores/indices: no physical scratch size is projected.
    for (auto& b : p.buffers) {
        for (std::size_t i = 0; i < p.nodes.size(); ++i) {
            const auto& n = p.nodes[i];
            const auto uses = [&](const auto& ids) { return std::find(ids.begin(), ids.end(), b.id) != ids.end(); };
            if (uses(n.inputs) || uses(n.outputs) || uses(n.state_reads) || uses(n.state_writes)) {
                b.first_use = std::min(b.first_use, static_cast<std::uint64_t>(i)); b.last_use = i;
                if (n.operation == "index_block_scores" || n.operation == "top16_blocks_newest_forced")
                    if (uses(n.outputs)) {
                        b.extent = "symbolic-index-blocks;physical-scratch-engine-local";
                        b.bytes = 0;
                    }
            }
        }
    }
    mg::validate(p, g);
    return p;
}

bool requested(const char* environment) {
    const char* value = std::getenv(environment);
    return value != nullptr && std::string_view(value) == "1";
}
bool inspect_or_check(const std::string& directory, std::uint32_t rank, Attachments attachments,
                      bool print_graph, bool print_plan) {
    if (!print_graph && !print_plan && !requested("STRIX_MINIMAX_MODEL_GRAPH_CHECK")) return false;
    const auto d = description::load((std::filesystem::path(directory) / "minimax-m3.1-flash.toml").string());
    const auto g = build_model_graph(d, attachments);
    assert_matches(d, g, attachments);
    const auto p = common_decode_plan(d, g, rank, [](const char* key) { return std::getenv(key); }, attachments);
    if (print_graph) std::fputs(mg::dump(g).c_str(), stdout);
    if (print_plan) std::fputs(mg::dump(p).c_str(), stdout);
    return print_graph || print_plan;
}
} // namespace strix::minimax::graph
