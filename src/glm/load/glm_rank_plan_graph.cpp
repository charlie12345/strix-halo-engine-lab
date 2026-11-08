#include "glm_rank_plan_graph.hpp"

#include <charconv>
#include <map>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace strix::glm::load {
namespace {
struct Inventory { bool kda = false, indexed = false, dense = false, moe = false, shared = false, mtp = false; };
[[noreturn]] void mismatch(std::uint64_t id, const char* field) {
    throw std::runtime_error("GLM model graph / rank plan mismatch at layer " + std::to_string(id) + ": " + field);
}
}  // namespace

void check_rank_plan_graph(const model_graph::ModelGraph& graph, const std::vector<std::string>& names) {
    namespace mg = model_graph;
    mg::validate(graph);
    std::map<std::uint64_t, Inventory> inventory;
    for (const auto& name : names) {
        constexpr std::string_view prefix = "model.language_model.layers.";
        if (!std::string_view(name).starts_with(prefix)) continue;
        const std::string_view tail = std::string_view(name).substr(prefix.size());
        std::uint64_t id = 0;
        const auto parsed = std::from_chars(tail.data(), tail.data() + tail.size(), id);
        if (parsed.ec != std::errc{} || parsed.ptr == tail.data() + tail.size() || *parsed.ptr != '.')
            throw std::runtime_error("GLM model graph: malformed rank-plan layer name");
        const std::string_view role(parsed.ptr + 1, static_cast<std::size_t>(tail.data() + tail.size() - parsed.ptr - 1));
        auto& record = inventory[id];
        const auto matches = [role](std::string_view stem) {
            return role == stem || (role.size() > stem.size() && role.substr(0, stem.size()) == stem && role[stem.size()] == '.');
        };
        record.kda |= matches("self_attn.qkv_proj");
        record.indexed |= matches("self_attn.kv_a_proj_with_mqa");
        record.dense |= matches("mlp.gate_proj");
        record.moe |= role.starts_with("mlp.experts.");
        record.shared |= role.starts_with("mlp.shared_experts.");
        record.mtp |= matches("eh_proj");
    }
    std::vector<const mg::LayerDesc*> expected;
    for (const auto& layer : graph.layers) expected.push_back(&layer);
    const std::size_t trunk = expected.size();
    for (const auto& attachment : graph.attachments) {
        if (attachment.kind != mg::AttachmentKind::mtp) throw std::runtime_error("GLM model graph: unexpected attachment");
        for (const auto& layer : attachment.layers) expected.push_back(&layer);
    }
    if (inventory.size() != expected.size()) throw std::runtime_error("GLM model graph / rank plan mismatch: layer count");
    auto actual = inventory.begin();
    for (std::size_t ordinal = 0; ordinal < expected.size(); ++ordinal, ++actual) {
        const auto& layer = *expected[ordinal];
        const auto& record = actual->second;
        if (layer.id != actual->first) mismatch(layer.id, "layer order/id");
        if (layer.block != mg::BlockKind::transformer) mismatch(layer.id, "block kind");
        const bool kda = layer.attention == mg::AttentionFamily::kda;
        if ((!kda && layer.attention != mg::AttentionFamily::compressed_indexed) ||
            record.kda != kda || record.indexed == kda) mismatch(layer.id, "attention family");
        const bool dense = layer.mlps.size() == 1 && layer.mlps[0].kind == mg::MlpKind::dense;
        const bool moe = layer.mlps.size() == 2 && layer.mlps[0].kind == mg::MlpKind::moe &&
                         layer.mlps[1].kind == mg::MlpKind::shared_expert;
        if ((!dense && !moe) || record.dense != dense || record.moe != moe || record.shared != moe)
            mismatch(layer.id, "MLP kind/shared expert");
        if (record.mtp != (ordinal >= trunk)) mismatch(layer.id, "MTP attachment");
    }
}
}  // namespace strix::glm::load
