#pragma once

#include "strix/engine_profile.hpp"
#include "strix/model_description.hpp"
#include "strix/serve/json.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

// Multi-device placement as model DATA: the [multi_device] section of a model TOML, resolved once by the shared
// profile resolver for the CLI, the server and the replica router (one function, one effective config).
//
//   [multi_device]
//   version = 1
//   default = "replica"          # form used when no regime matches; "single" when absent
//   fallback = "single"          # form used when the default cannot run (too few devices, tier); "single" when absent
//   order = ["short", "long"]    # regime evaluation order; required when there is more than one regime
//
//   [multi_device.forms.replica]    ranks (minimum devices), objective
//   [multi_device.forms.layers]     ranks, axis, cut, components, window_rows, state, wire
//   [multi_device.forms.tensor]     ranks, stages, components, expert_partition, expert_shares, wire,
//                                   rows_per_message, draft_placement, draft_rank
//   [multi_device.forms.speculation] ranks, draft_placement, draft_rank, wire, rows_per_message
//
//   [multi_device.regimes.NAME]     phase, min/max_prompt_rows, min/max_ready_requests, min_tier, prefer,
//                                   wire, lever, enabled
//
// A model without the section resolves `single`: nothing a launcher passes to an engine changes. Forms that are not
// declared are unavailable. The wire `reference` keeps every site's native exact contract (BF16 joins stay BF16, FP32
// partials stay FP32); narrower wires (fp16, fp8-*) are FAST / TURBO data and never resolve under EXACT. A regime
// whose `enabled = false` is kept as data (awaiting its receipt) and never selected.
namespace strix::profile::multi_device {

enum class Form { single, replica, layers, tensor, speculation };
enum class Phase { request, prefill, decode };

[[nodiscard]] std::string_view form_name(Form) noexcept;
[[nodiscard]] std::optional<Form> parse_form(std::string_view) noexcept;
[[nodiscard]] std::string_view phase_name(Phase) noexcept;
// True for a wire that keeps the reference bits (reference, bf16-exact).
[[nodiscard]] bool exact_wire(std::string_view wire) noexcept;

struct FormSpec {
    Form form = Form::single;
    std::uint32_t ranks = 1;              // single: 1; replica: minimum devices; layers/tensor/speculation: ranks used
    std::string wire = "reference";
    std::uint32_t draft_rank = 0;
    std::string draft_placement = "none";
    std::map<std::string, serve::Json, std::less<>> fields;   // every validated key, for the effective config
};

struct Regime {
    std::string name;
    Phase phase = Phase::request;
    std::optional<std::uint64_t> min_prompt_rows, max_prompt_rows, min_ready_requests, max_ready_requests;
    Name min_tier = Name::standard;
    std::vector<Form> prefer;
    std::string wire;                     // "" = the form's wire
    std::string lever;
    bool enabled = true;
};

struct Policy {
    std::int64_t version = 0;             // 0 = the model declares no [multi_device] section
    Form default_form = Form::single;
    Form fallback = Form::single;
    std::map<Form, FormSpec> forms;       // always holds single
    std::vector<Regime> regimes;          // evaluation order
};

// Throws std::invalid_argument naming the offending key.
[[nodiscard]] Policy parse_policy(const model_description::ModelDescription&);

struct Query {
    Name tier = Name::standard;
    Phase phase = Phase::request;
    std::optional<std::uint64_t> prompt_rows;   // unknown: a regime with a row bound does not match
    std::uint64_t ready_requests = 1;
    std::uint32_t devices = 1;                  // whole-model capable devices available to this launch
    std::optional<Form> placement;              // explicit --placement: wins or refuses, never silently replaced
};

struct LegacyOverride {
    std::string env, value, key;
};

struct Decision {
    Form form = Form::single;
    std::uint32_t ranks = 1;
    std::string wire = "reference";
    std::string rule = "absent";                // absent | explicit | regime:NAME | default | fallback
    std::string reason;
    std::vector<LegacyOverride> legacy;
    [[nodiscard]] serve::Json json(const Policy&, const Query&) const;
};

// Explicit placement > first matching enabled regime (in `order`) whose preferred form can run > default > fallback
// > single. An explicit placement that cannot run throws std::invalid_argument with the reason.
[[nodiscard]] Decision select(const Policy&, const Query&);

// Legacy switches the resolver replaces. They still act inside the engines that read them (no kit breaks); the
// resolver reports each one that is set, with the [multi_device] key it maps to, and logs one deprecation line.
struct LegacySwitch {
    const char* env;
    const char* key;
};
[[nodiscard]] const std::vector<LegacySwitch>& legacy_switches();
[[nodiscard]] std::vector<LegacyOverride> legacy_overrides(const Lookup& env = {});
[[nodiscard]] std::string deprecation_line(const LegacyOverride&);

// One call for every launcher: parse + select + legacy report.
[[nodiscard]] Decision resolve(const model_description::ModelDescription&, const Query&, const Lookup& env = {});

} // namespace strix::profile::multi_device
