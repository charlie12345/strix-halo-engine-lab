// Load-time weight decisions are scoped to the consumers' versioned arithmetic (weight::implementation_identity)
// and to the checkpoint content digest, never to the executable that holds the decision table.
#include <strix/profiles/hy4.hpp>
#include <strix/weight/levers.hpp>

#include <array>
#include <cstdio>
#include <limits>
#include <span>
#include <string>
#include <string_view>

namespace {
namespace weight = strix::weight;

bool check(bool ok, const char* what) {
    if (!ok) std::fprintf(stderr, "FAIL: %s\n", what);
    return ok;
}
}  // namespace

int main() {
    using weight::ArithmeticId;
    using weight::Digest256;
    bool ok = true;
    const std::array<ArithmeticId, 2> consumers{ArithmeticId{"rows-sweep", 1}, ArithmeticId{"f16-tile", 2}};
    const std::array<ArithmeticId, 2> reordered{consumers[1], consumers[0]};
    const std::array<ArithmeticId, 2> bumped{consumers[0], ArithmeticId{"f16-tile", 3}};
    const auto impl = weight::implementation_identity(consumers);
    ok &= check(!(impl == Digest256{}), "identity of a valid consumer set is non-zero");
    ok &= check(weight::implementation_identity(reordered) == impl, "identity is order-insensitive");
    ok &= check(!(weight::implementation_identity(bumped) == impl), "a bumped revision changes the identity");
    ok &= check(!(weight::implementation_identity(std::span(consumers).first(1)) == impl), "a dropped consumer changes the identity");
    const std::array<ArithmeticId, 2> duplicate{consumers[0], ArithmeticId{"rows-sweep", 2}};
    const std::array<ArithmeticId, 1> unnamed{ArithmeticId{"", 1}}, zero{ArithmeticId{"rows-sweep", 0}};
    ok &= check(weight::implementation_identity({}) == Digest256{}, "empty set refused");
    ok &= check(weight::implementation_identity(duplicate) == Digest256{}, "duplicate consumer refused");
    ok &= check(weight::implementation_identity(unnamed) == Digest256{}, "unnamed consumer refused");
    ok &= check(weight::implementation_identity(zero) == Digest256{}, "zero revision refused");

    constexpr auto parsed = weight::digest_from_hex("07000000000000000000000000000000000000000000000000000000000000ff");
    static_assert(parsed.bytes[0] == 7 && parsed.bytes[1] == 0 && parsed.bytes[31] == 0xff);
    ok &= check(weight::digest_from_hex("07") == Digest256{}, "short digest literal refused");
    ok &= check(weight::digest_from_hex("g7000000000000000000000000000000000000000000000000000000000000ff") == Digest256{},
                "non-hex digest literal refused");

    const auto checkpoint = weight::digest_from_hex("0700000000000000000000000000000000000000000000000000000000000001");
    const auto gate = weight::digest_from_hex("0800000000000000000000000000000000000000000000000000000000000001");
    const auto gain = weight::digest_from_hex("0900000000000000000000000000000000000000000000000000000000000001");
    const weight::Id policy{"policy", 1};
    weight::WeightLoadScope scope{weight::Backend::hip, "gfx1151", "hip-7.16", "resident", impl, checkpoint};
    const std::array rows{weight::WeightLoadDecision{policy, scope, gate, gain}};
    ok &= check(weight::find_weight_load_decision(policy, scope, rows) == rows.data(), "matching scope selects the row");
    auto other = scope;
    other.build = weight::implementation_identity(bumped);
    ok &= check(!weight::find_weight_load_decision(policy, other, rows), "row retired by a consumer revision bump");
    other.build = weight::implementation_identity({});
    ok &= check(!weight::find_weight_load_decision(policy, other, rows), "zero implementation identity never matches");
    other = scope;
    other.checkpoint = gate;
    ok &= check(!weight::find_weight_load_decision(policy, other, rows), "another checkpoint never matches");

    // Generic weight-load lever ids: one shared policy per role set and format; a model binds it and keeps the
    // id its earlier records used only as an alias.
    const auto& head = weight::kHeadQ8Policy;
    constexpr weight::Id legacy{"family-attention-head-bf16-q8_0", 1};
    constexpr auto attention_head = weight::with_legacy(weight::kAttentionHeadQ8Policy, legacy);
    constexpr std::array bound{attention_head};
    ok &= check(weight::serialize_id(attention_head.id) == "weights.attention_head.q8_0@1", "attention+head id");
    ok &= check(weight::serialize_id(head.id) == "weights.head.q8_0@1", "head id");
    static_assert(weight::find_weight_load_policy("weights.attention_head.q8_0@1") == &weight::kAttentionHeadQ8Policy);
    static_assert(weight::find_weight_load_policy("family-attention-head-bf16-q8_0@1") == nullptr);
    static_assert(weight::find_weight_load_policy("family-attention-head-bf16-q8_0@1", bound) ==
                  &weight::kAttentionHeadQ8Policy);
    static_assert(weight::find_weight_load_policy("weights.attention_head.q8_0@1", bound) ==
                  &weight::kAttentionHeadQ8Policy);
    static_assert(weight::find_weight_load_policy("family-attention-head-bf16-q8_0@2", bound) == nullptr);
    static_assert(weight::find_weight_load_policy("weights.head.q8_0@1", bound) == &weight::kHeadQ8Policy);
    static_assert(weight::find_weight_load_policy("weights.head.q8_0@2") == nullptr);
    // The attention-only row is its own policy: no alias resolves to it and it never resolves to the head rows.
    ok &= check(weight::serialize_id(weight::kAttentionQ8Policy.id) == "weights.attention.q8_0@1", "attention id");
    static_assert(weight::find_weight_load_policy("weights.attention.q8_0@1") == &weight::kAttentionQ8Policy);
    static_assert(weight::find_weight_load_policy("weights.attention.q8_0@1", bound) == &weight::kAttentionQ8Policy);
    static_assert(weight::find_weight_load_policy("weights.attention.q8_0@2") == nullptr);
    static_assert(weight::same_id(weight::kAttentionQ8Policy.target, weight::kQ8WeightStorage));
    static_assert(!weight::same_id(weight::kAttentionQ8Policy.id, weight::kAttentionHeadQ8Policy.id));
    static_assert(weight::find_weight_load_policy("weights.head.q8_0@01") == nullptr);
    static_assert(weight::find_weight_load_policy("weights.head.q8_0@") == nullptr);
    static_assert(weight::find_weight_load_policy("weights.head.q8_0") == nullptr);
    static_assert(weight::find_weight_load_policy("weights.head.q8_0@1x") == nullptr);
    static_assert(weight::find_weight_load_policy("") == nullptr);
    static_assert(weight::id_has_key({"a", 12}, "a@12") && !weight::id_has_key({"a", 12}, "a@1") &&
                  !weight::id_has_key({"a", 1}, "a@12") && !weight::id_has_key({"a", 0}, "a@0") &&
                  !weight::id_has_key({"a", 1}, "ab@1") && !weight::id_has_key({"", 1}, "@1"));
    bool hy4_head = false;
    for (const auto& row : strix::hy4::profile::table)
        hy4_head |= row.name == "head_q8" && row.target.find(weight::serialize_id(head.id)) != std::string_view::npos;
    ok &= check(hy4_head, "Hy4 head row names the generic head policy");
    // A decision row recorded under the former model-named key still selects the generic policy; another
    // policy, another revision or another key does not.
    const weight::WeightLoadEvidence timed{"receipts/timing", "2026-11-24", 16, 0, 1.10, 0.02};
    const std::array renamed{weight::WeightLoadDecision{legacy, scope, gate, gain, timed},
                             weight::WeightLoadDecision{attention_head.id, scope, gate, gain, timed}};
    ok &= check(weight::find_weight_load_decision(attention_head, scope, renamed) == renamed.data(),
                "legacy decision row matches the generic policy");
    ok &= check(weight::find_weight_load_decision(attention_head, scope, std::span(renamed).last(1)) == &renamed[1],
                "generic decision row matches");
    ok &= check(!weight::find_weight_load_decision(head, scope, renamed), "another policy never matches");
    ok &= check(!weight::find_weight_load_decision(attention_head.id, scope, std::span(renamed).first(1)),
                "the plain id lookup accepts no alias");
    const std::array bumped_legacy{weight::WeightLoadDecision{{legacy.name, 2}, scope, gate, gain}};
    ok &= check(!weight::find_weight_load_decision(attention_head, scope, bumped_legacy),
                "another legacy revision never matches");

    // The resolver and its receipt line use the generic id, whatever key the decision row was stored under.
    constexpr std::array formats{weight::kBf16WeightStorage, weight::kQ8WeightStorage};
    weight::WeightLoadRequest request{strix::profile::Name::fast, scope, formats, true, false, {}, 4096, 1};
    const auto plan = weight::resolve_weight_load(weight::builtin_registry(), attention_head, request,
                                                  std::span(renamed).first(1));
    ok &= check(plan.status == strix::profile::Status::set && plan.storage == weight::kQ8WeightStorage,
                "FAST selects the qualified storage through a legacy row");
    ok &= check(weight::weight_load_receipt(plan) == "strix-weight-load/v1 weights.attention_head.q8_0@1 gguf.q8_0@1 profile",
                "receipt prints the generic id");
    request.tier = strix::profile::Name::standard;
    ok &= check(weight::weight_load_receipt(weight::resolve_weight_load(weight::builtin_registry(), attention_head, request,
                                                                        renamed)) ==
                    "strix-weight-load/v1 weights.attention_head.q8_0@1 plain.bf16@1 builtin",
                "EXACT receipt keeps BF16 storage");

    // Per-regime rows: one resident image serves prefill and decode. A gate-only row (gate v2 PASS, no measured
    // gain) never selects the image nor admits its regime (ladder rule); the image needs a full row with a measured
    // speedup beyond noise in every regime it serves.
    {
        const std::array<std::string_view, 2> regimes{weight::kPrefillRegime, weight::kDecodeRegime};
        auto prefill = scope;
        prefill.regime = weight::kPrefillRegime;
        auto decode = scope;
        decode.regime = weight::kDecodeRegime;
        const weight::WeightLoadEvidence pf_receipt{"receipts/prefill", "2026-11-23", 2457, 0};
        const weight::WeightLoadEvidence banded{"receipts/decode", "2026-11-23", 2457, 65536, 1.158, 0.02};
        const weight::WeightLoadEvidence pf_timed{"receipts/prefill", "2026-11-23", 2457, 0, 1.05, 0.02};
        const weight::WeightLoadEvidence pf_slower{"receipts/prefill", "2026-11-23", 2457, 0, 0.997, 0.02};
        const auto& q8 = weight::kAttentionHeadQ8Policy;
        weight::WeightLoadRequest r{strix::profile::Name::fast, scope, formats, true, false, regimes, 32768, 1};
        const auto resolve = [&](std::span<const weight::WeightLoadDecision> table) {
            return weight::resolve_weight_load(weight::builtin_registry(), q8, r, table);
        };
        const std::array gate_only{weight::WeightLoadDecision{q8.id, prefill, gate, {}, pf_receipt}};
        auto p = resolve(gate_only);
        ok &= check(p.storage == weight::kBf16WeightStorage && p.status == strix::profile::Status::pending &&
                        p.regime_count == 2 && p.regimes[0].qualification == weight::RegimeQualification::gate_only &&
                        p.regimes[1].qualification == weight::RegimeQualification::none,
                    "a gate-only prefill row keeps the image BF16 and is reported");
        ok &= check(weight::weight_load_regime_receipt(p) == "strix-weight-load-regimes/v1 weights.attention_head.q8_0@1 "
                        "plain.bf16@1 pending prefill=gate-only:receipts/prefill decode=none", "per-regime receipt line");
        ok &= check(weight::find_weight_load_decision(q8, prefill, gate_only) == nullptr,
                    "a gate-only row is not a full decision row");
        const std::array full_decode_only{weight::WeightLoadDecision{q8.id, decode, gate, gain, banded}};
        p = resolve(full_decode_only);
        ok &= check(p.storage == weight::kBf16WeightStorage && p.status == strix::profile::Status::pending,
                    "a full decode row cannot select the image while prefill has no gate row (no second image)");
        const std::array both{weight::WeightLoadDecision{q8.id, prefill, gate, {}, pf_receipt},
                              weight::WeightLoadDecision{q8.id, decode, gate, gain, banded}};
        p = resolve(both);
        ok &= check(p.storage == weight::kBf16WeightStorage && p.status == strix::profile::Status::pending &&
                        p.regimes[0].qualification == weight::RegimeQualification::gate_only &&
                        p.regimes[1].qualification == weight::RegimeQualification::full,
                    "ladder: a gate-only prefill row no longer admits prefill onto the decode image");
        const std::array slower{weight::WeightLoadDecision{q8.id, prefill, gate, gain, pf_slower},
                                weight::WeightLoadDecision{q8.id, decode, gate, gain, banded}};
        p = resolve(slower);
        ok &= check(p.storage == weight::kBf16WeightStorage && p.status == strix::profile::Status::pending &&
                        p.regimes[0].qualification == weight::RegimeQualification::gate_only,
                    "ladder: a timed prefill row inside the noise margin (x0.997) keeps the image BF16");
        const std::array untimed{weight::WeightLoadDecision{q8.id, decode, gate, gain, pf_receipt}};
        r.regimes = {};
        r.scope = decode;
        ok &= check(resolve(untimed).storage == weight::kBf16WeightStorage,
                    "ladder: a gain digest without a measured speedup never selects");
        r.regimes = regimes;
        r.scope = scope;
        const std::array full_both{weight::WeightLoadDecision{q8.id, prefill, gate, gain, pf_timed},
                                   weight::WeightLoadDecision{q8.id, decode, gate, gain, banded}};
        p = resolve(full_both);
        ok &= check(p.storage == weight::kQ8WeightStorage && p.status == strix::profile::Status::set &&
                        p.measured_gain == gain && p.regimes[0].qualification == weight::RegimeQualification::full &&
                        p.regimes[1].qualification == weight::RegimeQualification::full,
                    "a measured ladder gain in every regime selects the image");
        r.context = 131072;
        p = resolve(full_both);
        ok &= check(p.storage == weight::kBf16WeightStorage && p.regimes[1].row == nullptr &&
                        p.regimes[1].outside_band == &full_both[1], "a context outside the gate band is refused");
        ok &= check(weight::weight_load_regime_receipt(p).ends_with("decode=none:outside-band context=1..65536 lanes=1..*"),
                    "outside-band receipt");
        r.context = 0;
        ok &= check(resolve(full_both).storage == weight::kBf16WeightStorage, "unknown context never meets a banded row");
        r.context = 32768;
        r.tier = strix::profile::Name::standard;
        ok &= check(resolve(full_both).storage == weight::kBf16WeightStorage, "EXACT never takes a regime row");
        r.tier = strix::profile::Name::fast;
        r.enabled = false;
        ok &= check(resolve(full_both).status == strix::profile::Status::off, "off switch");
        r.enabled = true;
        r.regimes = {};
        r.scope = decode;
        ok &= check(resolve(full_both).storage == weight::kQ8WeightStorage, "single-regime request keeps its old meaning");
        r.scope = prefill;
        ok &= check(resolve(both).storage == weight::kBf16WeightStorage, "single gate-only regime never selects");
    }
    request.tier = strix::profile::Name::fast;
    auto narrow = renamed.front();
    narrow.evidence.min_context = 4096;
    narrow.evidence.max_context = 8192;
    narrow.evidence.min_lanes = 2;
    narrow.evidence.max_lanes = 4;
    narrow.evidence.speedup = 1.008;
    narrow.evidence.noise = 0.002;
    const auto selected = [&](const weight::WeightLoadRequest& run, const weight::WeightLoadDecision& row) {
        return weight::resolve_weight_load(weight::builtin_registry(), attention_head, run, std::span(&row, 1)).storage;
    };
    request.lanes = 2;
    ok &= check(selected(request, narrow) == weight::kQ8WeightStorage, "lower band edge and small measured gain");
    request.lanes = 4;
    request.context = 8192;
    ok &= check(selected(request, narrow) == weight::kQ8WeightStorage, "upper band edge");
    request.lanes = 5;
    ok &= check(selected(request, narrow) == weight::kBf16WeightStorage, "lane above band declines");
    request.lanes = 1;
    ok &= check(selected(request, narrow) == weight::kBf16WeightStorage, "lane below band declines");
    request.lanes = 2;
    request.context = 4095;
    ok &= check(selected(request, narrow) == weight::kBf16WeightStorage, "context below band declines");
    request.context = 4096;
    auto other_device = request;
    other_device.scope.device_class = "other";
    ok &= check(selected(other_device, narrow) == weight::kBf16WeightStorage, "other device declines");
    auto malformed = narrow;
    malformed.evidence.receipt = {};
    ok &= check(selected(request, malformed) == weight::kBf16WeightStorage, "missing receipt declines");
    malformed = narrow;
    malformed.evidence.noise = std::numeric_limits<double>::quiet_NaN();
    ok &= check(selected(request, malformed) == weight::kBf16WeightStorage, "NaN noise declines");
    malformed = narrow;
    malformed.evidence.speedup = std::numeric_limits<double>::infinity();
    ok &= check(selected(request, malformed) == weight::kBf16WeightStorage, "infinite speedup declines");
    auto envelope = request;
    envelope.lanes = 4;
    envelope.first_lanes = 1;
    ok &= check(selected(envelope, narrow) == weight::kBf16WeightStorage, "an image needs the entire lane envelope");
    envelope.first_lanes = 2;
    ok &= check(selected(envelope, narrow) == weight::kQ8WeightStorage, "covering image receipt applies");
    auto slower_subset = narrow;
    slower_subset.evidence.min_lanes = slower_subset.evidence.max_lanes = 3;
    slower_subset.evidence.speedup = 0.95;
    const std::array overlapping{narrow, slower_subset};
    ok &= check(weight::resolve_weight_load(weight::builtin_registry(), attention_head, envelope, overlapping).storage ==
                    weight::kBf16WeightStorage, "a measured losing sub-band vetoes the image");
    request.tier = strix::profile::Name::turbo;
    ok &= check(selected(request, narrow) == weight::kQ8WeightStorage, "TURBO inherits the effective FAST storage");
    auto upper = narrow;
    upper.evidence.tier = strix::profile::Name::turbo;
    upper.evidence.reference = weight::kQ8WeightStorage;
    ok &= check(selected(request, upper) == weight::kBf16WeightStorage, "wrong FAST reference declines");
    upper.evidence.reference = weight::kBf16WeightStorage;
    ok &= check(selected(request, upper) == weight::kQ8WeightStorage, "correct effective FAST reference applies");
    if (ok) std::puts("weight decision scope tests: PASS");
    return ok ? 0 : 1;
}
