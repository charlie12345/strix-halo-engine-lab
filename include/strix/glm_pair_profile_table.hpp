#pragma once

#include "strix/engine_profile.hpp"

#include <array>

namespace strix::glm::rank_defaults {

// Default/quality export nothing. The FAST column selects no lever today: the FAST prefill set (pf_wmma) fails the
// 8K quality gate, so it is UNQUALIFIED for fast and runs only with an explicit STRIX_GLM_PF_WMMA=1 (the existing
// WMMA startup vote checks both ranks). The turbo cell keeps its previous value and status (turbo has its own gate).
inline constexpr std::array<strix::profile::Lever, 4> product_profile_table{{
    {"pf_wmma", "STRIX_GLM_PF_WMMA", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, true}, {true, true, true, true}, "glm-pf-wmma-x", "0",
     "docs/optimizations.md#glm-pf-wmma-x: UNQUALIFIED for fast: the FAST prefill set fails the 8K quality gate (KL 0.0587 and "
     "0.492 on two windows, no single member carries it); explicit STRIX_GLM_PF_WMMA=1 only until fixed; OFF equals exact anchors"},
    // PENDING turbo cells: not covered by pf_wmma (no code yet; the written analysis precedes any kernel).
    {"router_indexer_wmma", "STRIX_GLM_PF_ROUTER_INDEXER_WMMA", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {false, false, false, false}, "", "0",
     "pending: router and DSA indexer projections on WMMA (not covered by STRIX_GLM_PF_WMMA)"},
    {"kda_state16", "STRIX_GLM_PF_KDA_STATE16", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {false, false, false, false}, "", "0",
     "pending: KDA recurrent state in 16-bit (not covered by STRIX_GLM_PF_WMMA)"},
    // Server: sampled requests decode through MTP (explicit opt-in only; no profile selects it). Coupled acceptance keeps
    // the tokens of plain sampling; rejection acceptance changes per-seed tokens and is admitted by fast / turbo only
    // (rank_defaults::admit_sampled_mtp). Additive row: the builtin fingerprint stays the legacy one.
    {"serve_sampled_mtp", "STRIX_GLM_SERVE_SAMPLED_MTP", "", {"", "", "", ""}, "0", strix::profile::Kind::exact,
     {false, false, false, false}, {true, true, true, true}, "1", "0",
     "pending: pair gates of sampled MTP (coupled: seeded tokens and committed rows == plain sampling, forced rejection, "
     "batch handover, prefix restore, stops; rejection: acceptance/state oracle + distribution gate)",
     false, {}, true},
}};
inline const strix::profile::Engine product_profile_engine{"glm-pair", "", product_profile_table, false, "glm-turbo-v1"};

}  // namespace strix::glm::rank_defaults
