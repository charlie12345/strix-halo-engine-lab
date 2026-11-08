#include "plan.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>

namespace strix::model_graph {
namespace {
[[noreturn]] void fail(const char* message) { throw std::invalid_argument(std::string("execution plan: ") + message); }
std::string quote(std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    std::string out = "\"";
    for (const char ch : value) {
        const auto c = static_cast<unsigned char>(ch);
        if (c == '"' || c == '\\') { out += '\\'; out += ch; }
        else if (c < 32U) { out += "\\u00"; out += hex[c >> 4U]; out += hex[c & 15U]; }
        else out += ch;
    }
    return out + '"';
}
template<class T, class F> std::string array(const std::vector<T>& values, F emit) {
    std::string out = "[";
    bool comma = false;
    for (const auto& value : values) { if (comma) out += ','; comma = true; out += emit(value); }
    return out + ']';
}
std::string strings(const std::vector<std::string>& values) {
    return array(values, [](const auto& value) { return quote(value); });
}
std::string kind_name(OpKind kind) {
    switch (kind) {
    case OpKind::embedding: return "embedding";
    case OpKind::hyper_connection: return "hyper_connection";
    case OpKind::normalization: return "normalization";
    case OpKind::projection: return "projection";
    case OpKind::state_update: return "state_update";
    case OpKind::attention: return "attention";
    case OpKind::router: return "router";
    case OpKind::expert: return "expert";
    case OpKind::combine: return "combine";
    case OpKind::head: return "head";
    case OpKind::selection: return "selection";
    case OpKind::family_specific: return "family_specific";
    }
    fail("invalid op kind");
}
}  // namespace

void validate(const ExecutionPlan& plan, const ModelGraph& graph) {
    validate(graph);
    if (plan.graph_identity != identity(graph)) fail("graph identity mismatch");
    if (plan.ranks == 0 || plan.rank >= plan.ranks || plan.source.empty()) fail("invalid rank/source");
    const std::set<std::string> owners(plan.owners.begin(), plan.owners.end());
    if (owners.empty() || owners.contains("") || owners.size() != plan.owners.size()) fail("invalid owner vocabulary");
    if (plan.first_layer >= plan.end_layer || plan.end_layer > graph.layers.size() || plan.nodes.empty())
        fail("invalid layer range or empty schedule");
    std::map<std::string, const PlanBuffer*> buffers;
    for (const auto& b : plan.buffers) {
        if (b.id.empty() || !buffers.emplace(b.id, &b).second || b.rank != plan.rank ||
            b.first_use > b.last_use || b.last_use >= plan.nodes.size()) fail("invalid buffer/owner/lifetime");
        if (b.lifetime == BufferLifetime::sequence) {
            const auto* states = &graph.states;
            if (b.layer >= 0) {
                const auto it = std::find_if(graph.layers.begin(), graph.layers.end(),
                    [&](const auto& l) { return l.id == static_cast<std::uint64_t>(b.layer); });
                if (it == graph.layers.end()) fail("state layer missing from graph");
                states = &it->states;
            }
            if (std::none_of(states->begin(), states->end(), [&](const auto& s) { return s.tensor.role == b.state_role; }))
                fail("state absent from graph");
            const bool outside = b.layer >= 0 && (static_cast<std::uint64_t>(b.layer) < plan.first_layer ||
                                                  static_cast<std::uint64_t>(b.layer) >= plan.end_layer);
            if (outside != b.external) fail("invalid external state owner");
        } else if (b.lifetime != BufferLifetime::decode_step || (b.bytes == 0 && b.extent.empty()) || !b.state_role.empty() || b.external)
            fail("invalid activation buffer");
    }
    std::set<std::uint64_t> layers;
    std::set<std::string> used, written;
    for (std::size_t i = 0; i < plan.nodes.size(); ++i) {
        const auto& n = plan.nodes[i];
        (void)kind_name(n.kind);
        if (n.rank != plan.rank || !owners.contains(n.owner) || n.operation.empty() || n.arithmetic_identity.empty() ||
            n.expert_begin > n.expert_end) fail("invalid node owner/identity");
        if (n.layer >= 0) {
            const auto l = static_cast<std::uint64_t>(n.layer);
            if (l < plan.first_layer || l >= plan.end_layer || graph.layers[static_cast<std::size_t>(l)].id != l)
                fail("node layer absent from graph/range");
            layers.insert(l);
            if (n.expert_end > 0 && std::none_of(graph.layers[static_cast<std::size_t>(l)].mlps.begin(),
                graph.layers[static_cast<std::size_t>(l)].mlps.end(), [&](const auto& m) {
                    return m.kind == MlpKind::moe && n.expert_end <= m.experts;
                })) fail("expert owner exceeds graph inventory");
        } else if (n.layer != -1) fail("invalid global layer");
        for (const auto dependency : n.dependencies) if (dependency >= i) fail("non-causal dependency");
        const auto check = [&](const auto& ids, bool state, bool write) {
            for (const auto& id : ids) {
                const auto it = buffers.find(id);
                if (it == buffers.end()) fail("unknown buffer");
                const auto& b = *it->second;
                if ((b.lifetime == BufferLifetime::sequence) != state || i < b.first_use || i > b.last_use)
                    fail("buffer kind/lifetime mismatch");
                if (write && b.external) fail("write to external state");
                if (!state && !write && !b.live_in && !written.contains(id)) fail("read before producer");
                if (write) written.insert(id);
                used.insert(id);
            }
        };
        check(n.inputs, false, false); check(n.outputs, false, true);
        check(n.state_reads, true, false); check(n.state_writes, true, true);
    }
    if (layers.size() != plan.end_layer - plan.first_layer || used.size() != buffers.size())
        fail("missing layers or unused buffers");
}

std::string dump(const ExecutionPlan& plan) {
    std::string out = "{\"schema\":\"strix-execution-plan/v1\",\"graph_identity\":" + quote(plan.graph_identity) +
        ",\"source\":" + quote(plan.source) + ",\"rank\":" + std::to_string(plan.rank) + ",\"ranks\":" + std::to_string(plan.ranks) +
        ",\"first_layer\":" + std::to_string(plan.first_layer) + ",\"end_layer\":" + std::to_string(plan.end_layer) +
        ",\"owners\":" + strings(plan.owners);
    out += ",\"buffers\":" + array(plan.buffers, [](const auto& b) {
        return "{\"id\":" + quote(b.id) + ",\"bytes\":" + std::to_string(b.bytes) + ",\"lifetime\":" +
            quote(b.lifetime == BufferLifetime::sequence ? "sequence" : "decode_step") +
            ",\"first_use\":" + std::to_string(b.first_use) + ",\"last_use\":" + std::to_string(b.last_use) +
            ",\"bytes_per_position\":" + std::to_string(b.bytes_per_position) + ",\"layer\":" + std::to_string(b.layer) +
            ",\"rank\":" + std::to_string(b.rank) + ",\"state_role\":" + quote(b.state_role) +
            ",\"external\":" + (b.external ? "true" : "false") + ",\"live_in\":" + (b.live_in ? "true" : "false") +
            ",\"live_out\":" + (b.live_out ? "true" : "false") +
            (b.extent.empty() ? "" : ",\"extent\":" + quote(b.extent)) + '}';
    });
    out += ",\"nodes\":" + array(plan.nodes, [](const auto& n) {
        return "{\"kind\":" + quote(kind_name(n.kind)) + ",\"operation\":" + quote(n.operation) +
            ",\"layer\":" + std::to_string(n.layer) + ",\"rank\":" + std::to_string(n.rank) + ",\"owner\":" + quote(n.owner) +
            ",\"site\":" + quote(n.site) + ",\"arithmetic_identity\":" + quote(n.arithmetic_identity) +
            ",\"inputs\":" + strings(n.inputs) + ",\"outputs\":" + strings(n.outputs) +
            ",\"state_reads\":" + strings(n.state_reads) + ",\"state_writes\":" + strings(n.state_writes) +
            ",\"weights\":" + strings(n.weights) + ",\"dependencies\":" +
            array(n.dependencies, [](auto id) { return std::to_string(id); }) +
            ",\"expert_begin\":" + std::to_string(n.expert_begin) + ",\"expert_end\":" + std::to_string(n.expert_end) +
            (n.graph_safe_candidate ? "" : ",\"graph_safe_candidate\":false") +
            (n.dynamic_shape ? ",\"dynamic_shape\":true" : "") + '}';
    });
    if (!plan.policies.empty()) {
        out += ",\"policies\":{";
        bool comma = false;
        for (const auto& [key, value] : plan.policies) {
            if (comma) out += ',';
            comma = true;
            out += quote(key) + ':' + quote(value);
        }
        out += '}';
    }
    if (!plan.engine_local.empty()) out += ",\"engine_local\":" + strings(plan.engine_local);
    return out + "}\n";
}
}  // namespace strix::model_graph
