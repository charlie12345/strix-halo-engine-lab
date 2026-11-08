#include "qsa_index_score_fixture.hpp"
#include "strix/prefill_identity.hpp"
#include <iostream>
#include <limits>

int main() try {
    using namespace index_fixture;
    using namespace strix;
    require(median({100.0F, 1.0F, 3.0F, 2.0F}) == 2.5, "even median uses both middle values");
    require(median({20.0F, 1.0F, 100.0F, 3.0F, 10.0F, 2.0F}) == 6.5, "six-sample ABBA median");
    require(median({8.0F, 1.0F, 7.0F, 2.0F, 6.0F, 3.0F, 5.0F, 4.0F}) == 4.5, "eight-sample ABBA median");
    require(median({9.0F, 1.0F, 2.0F}) == 2.0 && median({7.0F}) == 7.0, "odd median");
    const auto largest = std::numeric_limits<float>::max();
    require(median({largest, largest}) == static_cast<double>(largest), "median intermediate overflow");
    bool empty_rejected = false;
    try { (void)median({}); } catch (const std::runtime_error&) { empty_rejected = true; }
    require(empty_rejected, "empty median accepted");
    const auto score = prefill_option_bit(PrefillOption::index_score_wmma);
    const auto empty = [](const char*) -> const char* { return nullptr; };
    const auto optin = [](const char* key) -> const char* {
        return std::string_view(key) == "STRIX_PREFILL_INDEX_SCORE_WMMA" ? "1" : nullptr;
    };
    require(resolve_prefill_options(empty) == 1U && !(kPrefillOptionsReady & score), "default mask changed");
    require(resolve_prefill_options(optin) == (1U | score), "opt-in missing");
    require(resolve_prefill_options(optin, false, score) == 1U && resolve_prefill_options(optin, true) == 0U, "OFF precedence");
    require(parse_prefill_option_list("index_score_wmma") == score, "named OFF switch");
    require(prefill_identity_name(1U | score) == "fp16-v1+hcout16+idxscore16", "score tag");
    require(prefill_identity_hash(0U) == UINT64_C(0xa0094634843e16ee) &&
        prefill_identity_hash(1U) == UINT64_C(0xee2cd64684149069) &&
        prefill_identity_hash(3U) == UINT64_C(0xde0052de9ed6494d), "main identity bytes changed");
    require(index_wmma_unqualified_route(score, kFastPrefillQsa) &&
        !index_wmma_unqualified_route(score, kFastPrefillHc), "MTP/nested exclusion");
    std::vector<float> q(heads * dim, 0.0F), k(dim, 0.0F);
    k[0] = 1.0F; q[0] = 2.0F; q[dim] = -4.0F; q[2U * dim] = 1.0F;
    require(panel_score(q.data(), k.data()) == 3.0F / std::sqrt(128.0F), "headwise ReLU lost");
    for (unsigned i = 0U; i < q.size(); ++i) q[i] = sample(i);
    for (unsigned i = 0U; i < k.size(); ++i) k[i] = sample(i + 7881U);
    const auto error = std::fabs(panel_score(q.data(), k.data()) - fp32_score(q.data(), k.data()));
    require(error < 1e-4F, "host panel sanity");
    std::vector<float> ties(8192U, 1.0F);
    auto a = selected_blocks(ties.data(), static_cast<unsigned>(ties.size()));
    require(a.size() == 384U && a.front() == 0U && a.back() == 383U, "1536 budget or ascending tie order changed");
    ties[500] = 2.0F;
    auto b = selected_blocks(ties.data(), static_cast<unsigned>(ties.size()));
    SelectionChanges m; m.add(a, a); m.add(a, b);
    require(m.rows == 2U && m.changed == 1U && m.maximum == 2U && m.mean() == 1.0, "selection statistic");
    std::cout << "PASS host controls/default identity, headwise ReLU, FP32/panel oracle, 1536 budget/ties/selection metrics\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
