#pragma once
#include "registry.hpp"
#include "resolve.hpp"
#include <strix/device/cost_identity.hpp>
#include <strix/engine_profile.hpp>
#include <strix/lever.hpp>

#include <array>
#include <optional>
#include <string>
#include <string_view>
namespace strix::weight {
inline constexpr lever::Decl measured_decisions{
    "weight-measured-decisions", "STRIX_WEIGHT_MEASURED_DECISIONS", lever::Exactness::bitwise,
    true, {}, {}, "per-device/compiler/regime byte-equal faster decision receipts"};
static_assert(lever::valid(std::span(&measured_decisions, 1)));
// The family passes its existing environment reader; this never introduces a second getenv path.
inline void apply_decision_override(ResolveRequest& r, const char* raw) {
    r.use_qualified_decisions = lever::select(measured_decisions, {}, raw).on;
}

// These decisions cover a whole family image. Changing the set of tagged roles,
// transcode rule or numerical bundle requires a new policy revision.
// `id` is backend- and model-neutral: 'weights.<roles>.<format>@<revision>'. A model binds one of the shared
// policies below (with_legacy when an earlier record used another key) and never mints a model-named id.
// `legacy` is the id an earlier receipt or decision row stored for the same policy (empty name: none). It only
// resolves old records; receipts print `id`.
struct WeightLoadPolicy {
    Id id;
    FormatId target;
    Id legacy{};
};

// Attention projections and the output head stored as Q8_0, converted from BF16 at load.
inline constexpr WeightLoadPolicy kAttentionHeadQ8Policy{{"weights.attention_head.q8_0", 1}, kQ8WeightStorage};
// The output head alone stored as Q8_0, converted from BF16 at load.
inline constexpr WeightLoadPolicy kHeadQ8Policy{{"weights.head.q8_0", 1}, kQ8WeightStorage};
// Every attention projection matrix (query, key, value, output and any auxiliary projection of the attention input,
// such as an output gate, relative or sink projection) stored as Q8_0, converted from BF16 at load; the output head
// stays BF16, so an exact head (full or screened) keeps its bits. Router and normalisation weights are not covered.
inline constexpr WeightLoadPolicy kAttentionQ8Policy{{"weights.attention.q8_0", 1}, kQ8WeightStorage};
inline constexpr std::array<const WeightLoadPolicy*, 3> kWeightLoadPolicies{
    &kAttentionHeadQ8Policy, &kHeadQ8Policy, &kAttentionQ8Policy};

// The binding of a shared policy by a model whose earlier records stored it under `legacy`.
[[nodiscard]] constexpr WeightLoadPolicy with_legacy(const WeightLoadPolicy& shared, Id legacy) noexcept {
    return {shared.id, shared.target, legacy};
}

// True when `key` is the 'name@revision' text of `id` (decimal revision, no sign, no leading zero).
[[nodiscard]] constexpr bool id_has_key(Id id, std::string_view key) noexcept {
    if (id.name.empty() || !id.revision || key.size() <= id.name.size() + 1 ||
        key.substr(0, id.name.size()) != id.name || key[id.name.size()] != '@') return false;
    const auto digits = key.substr(id.name.size() + 1);
    if (digits.front() == '0') return false;
    std::uint64_t value = 0;
    for (const char c : digits) {
        if (c < '0' || c > '9') return false;
        value = value * 10U + static_cast<std::uint64_t>(c - '0');
        if (value > id.revision) return false;
    }
    return value == id.revision;
}

[[nodiscard]] constexpr bool same_id(Id a, Id b) noexcept { return a.name == b.name && a.revision == b.revision; }

// True when a record stored under `stored` belongs to `policy`: its id, or its legacy id.
[[nodiscard]] constexpr bool policy_has_id(const WeightLoadPolicy& policy, Id stored) noexcept {
    return same_id(policy.id, stored) || (!policy.legacy.name.empty() && same_id(policy.legacy, stored));
}
// The same for a stored 'name@revision' key (a receipt line).
[[nodiscard]] constexpr bool policy_has_key(const WeightLoadPolicy& policy, std::string_view key) noexcept {
    return id_has_key(policy.id, key) || (!policy.legacy.name.empty() && id_has_key(policy.legacy, key));
}

// The shared policy a stored key names, or one of `bound` (a model's bindings, legacy ids included). The
// result is the shared catalogue row of that id; nullptr: unknown key.
[[nodiscard]] constexpr const WeightLoadPolicy* find_weight_load_policy(
    std::string_view key, std::span<const WeightLoadPolicy> bound = {}) noexcept {
    for (const auto& binding : bound)
        if (policy_has_key(binding, key))
            for (const auto* policy : kWeightLoadPolicies)
                if (same_id(policy->id, binding.id)) return policy;
    for (const auto* policy : kWeightLoadPolicies)
        if (id_has_key(policy->id, key)) return policy;
    return nullptr;
}

// Shared ids are unique, versioned and generic ('weights.' prefix); shared rows carry no legacy id.
[[nodiscard]] constexpr bool valid_weight_load_policies() noexcept {
    for (std::size_t i = 0; i < kWeightLoadPolicies.size(); ++i) {
        const auto& p = *kWeightLoadPolicies[i];
        if (p.id.name.substr(0, 8) != "weights." || !p.id.revision || p.target.name.empty() || !p.legacy.name.empty())
            return false;
        for (std::size_t j = 0; j < i; ++j)
            if (kWeightLoadPolicies[j]->id.name == p.id.name) return false;
    }
    return true;
}
static_assert(valid_weight_load_policies());

// `build` is the implementation identity of the admitted consumers (implementation_identity below), not the
// executable digest: a table compiled into an executable cannot name the digest of the executable that holds it.
// `checkpoint` is the raw SHA-256 of the checkpoint content the gate and timing receipts ran on.
struct WeightLoadScope {
    Backend backend;
    std::string_view device_class, compiler_id, regime;  // device_class: device_class_key below
    Digest256 build, checkpoint;
};

// Device class of a weight-load scope and of the decision rows measured on it: the shared cost-device identity
// (strix/device/cost_identity.hpp: architecture, product name, compute units, memory, clock, runtime and driver
// versions), the same key the persisted cost tables use. Two products that report the same architecture and resource
// counts but a different name or clock limit never share a receipt. The HIP and CUDA queries
// (device/hip_cost_identity.hpp, device/cuda_cost_identity.hpp) fill the facts of the current device.
[[nodiscard]] inline std::string device_class_key(const device::CostDeviceFacts& facts) {
    return device::cost_device_identity(facts);
}

// Regimes a decision row is qualified for (WeightLoadScope::regime): the gate v2 regime of its quality receipt.
// Prompt rows of a prefill chunk are the prefill regime; one-row steps (and their verify rows) the decode regime.
inline constexpr std::string_view kPrefillRegime = "prefill";
inline constexpr std::string_view kDecodeRegime = "decode";

// Where a row's evidence lives. `receipt` is a stable repository-relative id of the gate report (or receipt set),
// `date` its ISO date, `rows` the scored rows of the regime. `max_context` is the largest context the gate covered
// when it was banded by context (0: not banded); a banded row applies only to an engine whose resident context is
// known and inside the band. `speedup` is the measured throughput of the regime with the policy storage over the tier
// below (same build, device, regime; the timing receipt behind measured_gain) and `noise` its stated margin: the ladder
// rule admits the row only when the speedup clears 1 + noise for the matched band and reference.
struct WeightLoadEvidence {
    std::string_view receipt, date;
    std::uint32_t rows = 0, max_context = 0;
    double speedup = 0.0, noise = 0.0;
    std::uint32_t min_context = 1, min_lanes = 1, max_lanes = profile::kNoLaneBound;
    profile::Name tier = profile::Name::fast;
    FormatId reference = kBf16WeightStorage;  // effective storage of the measured tier below
};

// A row with a zero measured_gain records a gate v2 PASS without a speed gain in its regime: it never selects the
// storage, and under the ladder rule it does not admit its regime onto the policy image either (the regime would run
// lossy weights with no measured gain); it is reported as gate-only.
struct WeightLoadDecision {
    Id policy;
    WeightLoadScope scope;
    Digest256 gate_v2_pass, measured_gain;
    WeightLoadEvidence evidence{};
};

struct WeightLoadRequest {
    profile::Name tier;
    WeightLoadScope scope;
    // Intersection of the formats consumed by every selected projection and
    // head site, including any exact fallbacks that can run under this regime.
    std::span<const FormatId> consumer_formats;
    bool enabled = true;
    bool allow_pending = false;  // Explicit, stamped qualification-arm opt-in.
    // Every regime that runs on the one resident image (empty: scope.regime alone). The image takes the policy storage
    // only when every listed regime has a full row (gate v2 PASS and a measured ladder gain): a regime without one would
    // run lossy weights slower than (or no faster than) the tier below. An engine that keeps a second image asks each
    // regime alone and routes per regime.
    std::span<const std::string_view> regimes{};
    std::uint32_t context = 0;  // actual context (0: unknown)
    std::uint32_t lanes = 0;    // actual lane count (0: unknown)
    // With a fixed image, the upper and lower endpoints must fit the same measured band.
    std::uint32_t first_context = 0, first_lanes = 0;  // 0: single point at context/lanes
};

// What one regime contributed to a plan: a full row (gate v2 PASS + measured gain), a gate-only row, or nothing.
enum class RegimeQualification : std::uint8_t { none, gate_only, full };

[[nodiscard]] constexpr std::string_view regime_qualification_name(RegimeQualification q) noexcept {
    return q == RegimeQualification::full ? "full" : q == RegimeQualification::gate_only ? "gate-only" : "none";
}

struct RegimeWeightLoad {
    std::string_view regime;
    RegimeQualification qualification = RegimeQualification::none;
    const WeightLoadDecision* row = nullptr;          // matched row (nullptr: none)
    const WeightLoadDecision* outside_band = nullptr;
    bool faster = false;  // every matching timing row must establish a gain
};

inline constexpr std::size_t kMaxWeightLoadRegimes = 4;

struct WeightLoadPlan {
    Id policy;
    FormatId storage;
    profile::Status status;
    std::string_view reason;
    Digest256 gate_v2_pass{}, measured_gain{};
    std::array<RegimeWeightLoad, kMaxWeightLoadRegimes> regimes{};  // per regime, in request order
    std::size_t regime_count = 0;

    [[nodiscard]] FormatId storage_for(std::optional<Role> role) const {
        return role ? storage : kBf16WeightStorage;
    }
    [[nodiscard]] std::span<const RegimeWeightLoad> regime_rows() const { return std::span(regimes).first(regime_count); }
};

// A 64-hex content digest literal for receipt rows compiled into a decision table. A malformed literal yields the
// zero digest, which never matches a decision row.
[[nodiscard]] constexpr Digest256 digest_from_hex(std::string_view hex) noexcept {
    const auto nibble = [](char c) -> int {
        return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
    };
    Digest256 out{};
    if (hex.size() != 2 * out.bytes.size()) return {};
    for (std::size_t i = 0; i < out.bytes.size(); ++i) {
        const int high = nibble(hex[2 * i]), low = nibble(hex[2 * i + 1]);
        if (high < 0 || low < 0) return {};
        out.bytes[i] = static_cast<std::uint8_t>(high * 16 + low);
    }
    return out;
}

// Digest of the versioned arithmetic of every consumer route the policy admits (projections, head, prefill tile,
// fallbacks). Order-insensitive; a change of any consumer's arithmetic revision changes it. An empty list, an
// unnamed entry, a zero revision or a duplicate yields the zero digest, which never matches a decision row.
[[nodiscard]] Digest256 implementation_identity(std::span<const ArithmeticId> consumers);

// Only passing v2 gate and speed evidence enters this table. Its receipts must come from the same consumer
// implementation identity, checkpoint, backend, device, compiler and regime.
const WeightLoadDecision* find_weight_load_decision(Id, const WeightLoadScope&,
                                                  std::span<const WeightLoadDecision>);
// As above, also matching rows recorded under the policy's legacy id.
const WeightLoadDecision* find_weight_load_decision(const WeightLoadPolicy&, const WeightLoadScope&,
                                                  std::span<const WeightLoadDecision>);
// Gate and timing evidence for this exact execution scope and lower-tier storage.
void match_weight_load_evidence(const WeightLoadPolicy&, const WeightLoadScope&,
                                std::span<const WeightLoadDecision>, const WeightLoadRequest&,
                                FormatId reference, RegimeWeightLoad&);
WeightLoadPlan resolve_weight_load(const Registry&, const WeightLoadPolicy&,
                                   const WeightLoadRequest&, std::span<const WeightLoadDecision>);

// Per-regime receipt line of a plan: 'strix-weight-load-regimes/v1 <policy id> <storage id> <status> <regime>=<q>'
// per regime, <q> = full | gate-only | none, followed by the matching receipt or the missed context/lane band.
[[nodiscard]] std::string weight_load_regime_receipt(const WeightLoadPlan& plan);

// Before a device is known (a route description): the committed rows of `policy` for `backend`, per regime,
// '<regime>=<full|gate-only|none>[@<device>]' joined by ','. Says what a device could match, not what it matched.
[[nodiscard]] inline std::string registered_regime_rows(const WeightLoadPolicy& policy, Backend backend,
                                                        std::span<const std::string_view> regimes,
                                                        std::span<const WeightLoadDecision> decisions) {
    std::string out;
    for (const auto regime : regimes) {
        if (!out.empty()) out += ',';
        out += regime;
        out += '=';
        const WeightLoadDecision* found = nullptr;
        for (const auto& row : decisions)
            if (policy_has_id(policy, row.policy) && row.scope.backend == backend && row.scope.regime == regime &&
                !(row.gate_v2_pass == Digest256{}) && (!found || !(row.measured_gain == Digest256{}))) found = &row;
        out += regime_qualification_name(!found ? RegimeQualification::none
            : found->measured_gain == Digest256{} ? RegimeQualification::gate_only : RegimeQualification::full);
        if (found) {
            out += '@';
            out += found->scope.device_class;
        }
    }
    return out;
}

// Receipt line of a resolved plan: 'strix-weight-load/v1 <policy id> <storage id> <status>'. The policy is
// always printed under its generic id; a reader resolves older lines through find_weight_load_policy.
[[nodiscard]] inline std::string weight_load_receipt(const WeightLoadPlan& plan) {
    const auto put = [](std::string& out, Id id) {
        out += id.name;
        out += '@';
        out += std::to_string(id.revision);
    };
    std::string out = "strix-weight-load/v1 ";
    put(out, plan.policy);
    out += ' ';
    put(out, plan.storage);
    out += ' ';
    out += profile::status_name(plan.status);
    return out;
}
}
