#pragma once

#include "plan.hpp"
#include "strix/execution_plan.hpp"
#include "strix/qwen_model_description.hpp"

namespace strix::qwen_description {
// Description/projection only. MTP describes configured model assets, not a
// claim that the trunk serial schedule executes speculative decoding.
[[nodiscard]] model_graph::ModelGraph model_graph_from_description(
    const model_description::ModelDescription& file);
[[nodiscard]] model_graph::ModelGraph model_graph_from_contract(const ModelContract& model);
[[nodiscard]] model_graph::ExecutionPlan common_decode_plan(
    const SerialExecutionPlan& source, const model_graph::ModelGraph& graph);
// Diagnostic gate: checks the source schedule and the complete projection
// against the resolved model, including state effects and capture flags.
void validate_decode_projection(const ModelContract& model, const SerialExecutionPlan& source,
    const model_graph::ModelGraph& graph, const model_graph::ExecutionPlan& plan);
}  // namespace strix::qwen_description
