#pragma once

#include <array>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::glm::pf {

enum class ScratchPolicy : unsigned { full, routed, automatic };
struct CapacityPolicy {
    ScratchPolicy scratch = ScratchPolicy::automatic;
    int query_window = -1;  // -1 = auto; 0 = full chunk
};
struct CapacityLayout {
    bool routed = false;
    unsigned query_window = 0;
    bool sharded = false;
};

inline ScratchPolicy parse_scratch_policy(const char* value) {
    const std::string s = value ? value : "auto";
    if (s == "auto") return ScratchPolicy::automatic;
    if (s == "full") return ScratchPolicy::full;
    if (s == "routed") return ScratchPolicy::routed;
    throw std::invalid_argument("STRIX_GLM_PF_SCRATCH must be auto, full or routed");
}
inline unsigned parse_attention_query_window(const char* value) {
    if (!value) return 0;
    unsigned rows = 0;
    if (*value == '\0') throw std::invalid_argument("STRIX_GLM_PF_ATTN_QWIN must be auto or 0..65536");
    for (const char* p = value; *p; ++p) {
        if (*p < '0' || *p > '9' || rows > (65536U - unsigned(*p - '0')) / 10)
            throw std::invalid_argument("STRIX_GLM_PF_ATTN_QWIN must be auto or 0..65536");
        rows = rows * 10 + unsigned(*p - '0');
    }
    return rows;
}
inline int parse_query_window_policy(const char* value) {
    return !value || std::string(value) == "auto" ? -1 : static_cast<int>(parse_attention_query_window(value));
}
inline const CapacityPolicy& capacity_policy() {
    static const CapacityPolicy policy{parse_scratch_policy(std::getenv("STRIX_GLM_PF_SCRATCH")),
                                       parse_query_window_policy(std::getenv("STRIX_GLM_PF_ATTN_QWIN"))};
    return policy;
}

// Explicit axes are pinned. Within each KV layout, retain the original byte
// layout first, then reclaim the unused MoE arena, then bound attention queries.
inline std::vector<CapacityLayout> capacity_candidates(CapacityPolicy policy, bool sharded) {
    const bool routed = policy.scratch == ScratchPolicy::routed;
    const unsigned window = policy.query_window < 0 ? 0U : static_cast<unsigned>(policy.query_window);
    std::vector<CapacityLayout> candidates{{routed, window, sharded}};
    if (policy.scratch == ScratchPolicy::automatic)
        candidates.push_back({true, window, sharded});
    if (policy.query_window < 0)
        candidates.push_back({policy.scratch != ScratchPolicy::full, 2048, sharded});
    return candidates;
}

// Both ranks must admit the SAME candidate. In particular, independent rank-local
// first-fit selections must never be combined into a false admission.
template<class Vote>
inline bool select_capacity_layout(CapacityPolicy policy, bool initial_sharded, bool shard_auto,
                                   Vote&& vote, CapacityLayout& selected, unsigned skip = 0) {
    unsigned candidate = 0;
    for (unsigned pass = 0; pass < (shard_auto && !initial_sharded ? 2U : 1U); ++pass) {
        for (const auto layout : capacity_candidates(policy, initial_sharded || pass != 0)) {
            // A jointly refused installation resumes strictly after that layout.
            if (candidate++ < skip) continue;
            const auto fits = vote(layout);
            if (fits[0] && fits[1]) {
                selected = layout;
                return true;
            }
        }
    }
    return false;
}

// Resolved once during startup, before any path allocation. Budgeting and path
// construction read the same value; no policy checks are added to kernel calls.
inline CapacityLayout& capacity_layout() {
    static CapacityLayout layout{capacity_policy().scratch == ScratchPolicy::routed,
        capacity_policy().query_window < 0 ? 0U : static_cast<unsigned>(capacity_policy().query_window), false};
    return layout;
}
inline const char* scratch_policy_name(ScratchPolicy policy) {
    return policy == ScratchPolicy::automatic ? "auto" : policy == ScratchPolicy::full ? "full" : "routed";
}
inline const char* capacity_layout_reason(CapacityLayout layout) {
    if (layout.sharded) return "replicated-candidates-refused-sharded-admitted";
    if (layout.query_window != 0) return "first-admitted-query-window";
    return layout.routed ? "first-admitted-routed-scratch" : "original-layout-admitted";
}

}  // namespace strix::glm::pf
