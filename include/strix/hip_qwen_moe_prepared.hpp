#pragma once
#include "strix/qwen_moe_prepared.hpp"
#include <vector>

namespace strix::qwen_moe_prepared {
struct Entry { std::uintptr_t original = 0; PreparedTensor tensor; };
struct Model {
    ModelContentIdentity model_hash;
    std::vector<Entry> entries;
    mutable std::uint64_t engaged = 0, declined = 0;
    mutable std::string_view reason = "not-executed";
    [[nodiscard]] const PreparedTensor* find(std::uintptr_t original, Role role, Spec expected) const;
};
// Setup-only synchronous uploads directly to one final HIP allocation.
// Retain the PreparedTensor owner until all consumers/graphs have quiesced.
[[nodiscard]] std::unique_ptr<Allocation> allocate_hip(std::uint64_t bytes);
} // namespace strix::qwen_moe_prepared
