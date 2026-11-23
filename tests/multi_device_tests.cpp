// Host tests of the [multi_device] model data and its one resolver (strix/profile/multi_device.hpp):
// defaults equal today's single-device behaviour, regime selection, tier rules for the wire, explicit placement,
// legacy environment reports, and the registered model descriptions.
#include "strix/model_registry.hpp"
#include "strix/profile/model_preset.hpp"
#include "strix/profile/multi_device.hpp"

#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <string>

namespace {
namespace md = strix::profile::multi_device;
using strix::profile::Name;

int failures = 0;
void require(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

void refuses(const std::function<void()>& f, const std::string& needle) {
    try {
        f();
        require(false, "expected refusal containing '" + needle + "'");
    } catch (const std::invalid_argument& e) {
        require(std::string(e.what()).find(needle) != std::string::npos,
                "refusal '" + std::string(e.what()) + "' should contain '" + needle + "'");
    }
}

md::Policy policy(const std::string& text) { return md::parse_policy(strix::model_description::parse(text)); }

md::Query query(Name tier, std::uint32_t devices, md::Phase phase = md::Phase::request,
                std::optional<std::uint64_t> rows = {}, std::uint64_t ready = 1) {
    md::Query q;
    q.tier = tier;
    q.devices = devices;
    q.phase = phase;
    q.prompt_rows = rows;
    q.ready_requests = ready;
    return q;
}

const char* kFull = R"(
[multi_device]
version = 1
default = "replica"
fallback = "single"
order = ["short_concurrent", "long_prefill", "long_prefill_fp8", "parked"]

[multi_device.forms.replica]
ranks = 2
objective = "throughput"

[multi_device.forms.layers]
ranks = 2
axis = "visit"
cut = "auto"
components = ["trunk"]
window_rows = 2048
state = "relay-if-needed"

[multi_device.forms.tensor]
ranks = 2
components = ["attention", "ffn", "moe", "head"]
expert_partition = "intermediate"
expert_shares = [0.5, 0.5]
wire = "bf16-exact"
rows_per_message = "auto"
draft_placement = "leader"
draft_rank = 0

[multi_device.regimes.short_concurrent]
max_prompt_rows = 4095
min_ready_requests = 2
prefer = ["replica", "single"]

[multi_device.regimes.long_prefill]
phase = "prefill"
min_prompt_rows = 4096
prefer = ["layers", "single"]

[multi_device.regimes.long_prefill_fp8]
phase = "prefill"
min_prompt_rows = 16384
min_tier = "fast"
prefer = ["tensor"]
wire = "fp8-e4m3-g64-v1"
lever = "pair-wire-fp8@1"

[multi_device.regimes.parked]
phase = "decode"
prefer = ["tensor"]
enabled = false
)";

void defaults_are_single() {
    const auto absent = policy("name = 'x'\n");
    require(absent.version == 0 && absent.forms.size() == 1, "absent section declares only single");
    for (Name tier : {Name::standard, Name::fast, Name::turbo})
        for (std::uint32_t devices : {1U, 2U, 8U}) {
            const auto d = md::select(absent, query(tier, devices, md::Phase::prefill, 100000, 16));
            require(d.form == md::Form::single && d.ranks == 1 && d.wire == "reference" && d.rule == "absent",
                    "a model without [multi_device] always resolves single");
        }
    refuses([&] { (void)md::select(absent, [] { md::Query q; q.placement = md::Form::replica; q.devices = 2; return q; }()); },
            "replica is not declared");
}

void regimes() {
    const auto p = policy(kFull);
    require(p.regimes.size() == 4 && p.regimes[0].name == "short_concurrent", "regimes follow `order`");
    // One device: the replica default cannot run -> fallback single (today's behaviour on one node).
    auto d = md::select(p, query(Name::standard, 1));
    require(d.form == md::Form::single && d.rule == "fallback" && d.reason.find("needs 2 devices") != std::string::npos,
            "one device falls back to single with the reason");
    // Two devices, one request of unknown length: no regime matches -> default replica over every device.
    d = md::select(p, query(Name::standard, 2));
    require(d.form == md::Form::replica && d.ranks == 2 && d.rule == "default", "two devices default to replicas");
    d = md::select(p, query(Name::standard, 3));
    require(d.form == md::Form::replica && d.ranks == 3, "replicas use every device");
    // Short prompts with concurrency -> replicas through the regime.
    d = md::select(p, query(Name::standard, 2, md::Phase::request, 512, 4));
    require(d.form == md::Form::replica && d.rule == "regime:short_concurrent", "short concurrent -> replica");
    // Long prefill -> staged split, EXACT keeps the reference wire.
    d = md::select(p, query(Name::standard, 2, md::Phase::prefill, 8192));
    require(d.form == md::Form::layers && d.rule == "regime:long_prefill" && d.wire == "reference",
            "long prefill -> layers with the reference wire");
    // A request-phase query never matches a prefill regime.
    d = md::select(p, query(Name::standard, 2, md::Phase::request, 8192));
    require(d.form == md::Form::replica && d.rule == "default", "phase gates a regime");
    // Very long prefill: EXACT skips the FP8 regime (min_tier) and takes long_prefill; FAST takes the FP8 wire.
    d = md::select(p, query(Name::standard, 2, md::Phase::prefill, 32768));
    require(d.form == md::Form::layers && d.rule == "regime:long_prefill", "EXACT never takes the FP8 regime");
    // Order: long_prefill precedes long_prefill_fp8, so FAST also matches long_prefill first.
    d = md::select(p, query(Name::fast, 2, md::Phase::prefill, 32768));
    require(d.rule == "regime:long_prefill", "regimes are evaluated in declared order");
    // Disabled regimes are data only.
    d = md::select(p, query(Name::turbo, 2, md::Phase::decode));
    require(d.form == md::Form::replica && d.rule == "default", "a disabled regime is never selected");
    // The JSON states form, wire, rule and the declared forms.
    const auto json = d.json(p, query(Name::turbo, 2, md::Phase::decode)).dump();
    require(json.find("\"form\":\"replica\"") != std::string::npos && json.find("\"declared\":[") != std::string::npos,
            "effective JSON states the form and the declared forms: " + json);
}

void wire_tiers() {
    const auto p = policy(R"(
[multi_device]
version = 1
[multi_device.forms.tensor]
wire = "fp16"
[multi_device.regimes.long]
min_prompt_rows = 1024
min_tier = "fast"
prefer = ["tensor"]
)");
    auto d = md::select(p, query(Name::fast, 2, md::Phase::request, 2048));
    require(d.form == md::Form::tensor && d.wire == "fp16", "FAST takes a declared narrow wire");
    d = md::select(p, query(Name::standard, 2, md::Phase::request, 2048));
    require(d.form == md::Form::single, "EXACT never resolves a narrow wire");
    md::Query explicit_exact = query(Name::standard, 2);
    explicit_exact.placement = md::Form::tensor;
    refuses([&] { (void)md::select(p, explicit_exact); }, "changes bits and the tier is exact");
    md::Query explicit_fast = query(Name::fast, 2);
    explicit_fast.placement = md::Form::tensor;
    d = md::select(p, explicit_fast);
    require(d.form == md::Form::tensor && d.rule == "explicit", "explicit placement wins when it can run");
    explicit_fast.devices = 1;
    refuses([&] { (void)md::select(p, explicit_fast); }, "needs 2 devices, 1 available");
}

void validation() {
    refuses([] { (void)policy("[multi_device]\ndefault = 'single'\n"); }, "version = 1");
    refuses([] { (void)policy("[multi_device]\nversion = 1\nplace = 1\n"); }, "unknown key multi_device.place");
    refuses([] { (void)policy("[multi_device]\nversion = 1\ndefault = 'replica'\n"); }, "undeclared form");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.forms.replica]\nwire = 'fp16'\n"); },
            "unknown key multi_device.forms.replica.wire");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.forms.replica]\n"
                              "[multi_device]\n"); }, "");
    refuses([] { (void)policy("[multi_device]\nversion = 1\nfallback = 'replica'\n[multi_device.forms.replica]\n"); },
            "fallback must be");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.forms.tensor]\ndraft_rank = 2\n"); },
            "draft_rank");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.forms.speculation]\n"); },
            "leader or split");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.forms.tensor]\nwire = 'int4'\n"); },
            "must be one of");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.regimes.a]\nprefer = ['replica']\n"
                              "wire = 'fp8-e4m3-g64-v1'\n"); }, "needs min_tier");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.regimes.a]\nprefer = ['x']\n"); },
            "unknown form");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.regimes.a]\nprefer = ['single']\n"
                              "[multi_device.regimes.b]\nprefer = ['single']\n"); }, "order must list every regime");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.regimes.a]\nprefer = ['single']\n"
                              "min_prompt_rows = 9\nmax_prompt_rows = 8\n"); }, "min_prompt_rows > max_prompt_rows");
    refuses([] { (void)policy("[multi_device]\nversion = 1\n[multi_device.forms.tensor]\nexpert_shares = [0.7, 0.7]\n"); },
            "sum to 1");
}

void legacy_environment() {
    const std::map<std::string, std::string> env{{"STRIX_QWEN_PF_PIPE_CUT", "20"}, {"STRIX_HY4_TP_DRAFT", "both"},
                                                 {"STRIX_UNRELATED", "1"}, {"STRIX_G4_PAIR", ""}};
    const strix::profile::Lookup lookup = [&](const char* key) -> const char* {
        const auto it = env.find(key);
        return it == env.end() ? nullptr : it->second.c_str();
    };
    const auto found = md::legacy_overrides(lookup);
    require(found.size() == 2, "set legacy switches are reported, empty and unrelated ones are not");
    bool cut = false, draft = false;
    for (const auto& o : found) {
        cut |= o.env == "STRIX_QWEN_PF_PIPE_CUT" && o.value == "20" && o.key == "multi_device.forms.layers.cut";
        draft |= o.env == "STRIX_HY4_TP_DRAFT" && o.key == "multi_device.forms.tensor.draft_placement";
        require(md::deprecation_line(o).find("deprecated: " + o.env) != std::string::npos, "deprecation line names the env");
    }
    require(cut && draft, "legacy switches map to their [multi_device] keys");
    // Reports never change the decision: engines still read their switches.
    const auto d = md::resolve(strix::model_description::parse("name = 'x'\n"), query(Name::standard, 1), lookup);
    require(d.form == md::Form::single && d.legacy.size() == 2, "legacy reports ride on the decision");
    const auto json = d.json(policy("name = 'x'\n"), query(Name::standard, 1)).dump();
    require(json.find("\"legacy_env\":[{") != std::string::npos, "effective JSON lists legacy switches");
}

void registered_models(const std::filesystem::path& root) {
    const auto entries = strix::model_registry::load((root / "models/registry.toml").string());
    require(!entries.empty(), "registry has models");
    for (const auto& entry : entries) {
        const auto description = strix::model_description::load((root / entry.description).string());
        const auto p = md::parse_policy(description);
        // Today's behaviour: one device always resolves single, whatever the model declares.
        for (Name tier : {Name::standard, Name::fast, Name::turbo}) {
            const auto d = md::select(p, query(tier, 1, md::Phase::prefill, 65536, 16));
            require(d.form == md::Form::single, entry.name + ": one device resolves single");
        }
        // The preset resolver states the placement in every effective config.
        const auto preset = strix::profile::resolve_model_preset(description, {}, "hip");
        const auto* member = preset.config.find("multi_device");
        require(member && member->find("form") && member->find("form")->as_string() == "single",
                entry.name + ": effective config states multi_device.form = single on one device");
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: strix-multi-device-tests SOURCE_ROOT\n";
        return 2;
    }
    defaults_are_single();
    regimes();
    wire_tiers();
    validation();
    legacy_environment();
    registered_models(argv[1]);
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "multi_device: ok\n";
    return 0;
}
