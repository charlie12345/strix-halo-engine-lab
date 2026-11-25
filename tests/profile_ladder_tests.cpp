// Ladder rule (strix/engine_profile.hpp): TURBO is never slower than FAST, FAST never slower than EXACT, in any
// regime. Fixtures check the verdicts and the route line; the walk resolves every lossy lever of every registered engine
// at every probe regime of its ladder rows and fails when a regime would run a cell that is not measured faster than
// the cell below it (or a ladder row is malformed).
#include "strix/engine_profile.hpp"
#include "strix/cuda/qwen_options.hpp"
#include "strix/glm_pair_profile_table.hpp"
#include "strix/deepseek41_pair_profile_table.hpp"
#include "strix/model_registry.hpp"
#include "strix/profiles/ds41_rtx.hpp"
#include "strix/profiles/gemma4.hpp"
#include "strix/profiles/qwen_cuda.hpp"
#include "strix/profiles/qwen_dense_cuda.hpp"
#include "strix/profiles/qwen_dense_hip.hpp"
#include "strix/profiles/hy4.hpp"
#include "strix/profiles/inkling2_mini.hpp"
#include "minimax/engine/mm_defaults.hpp"
#include "cuda/models/glm/producer/profile.hpp"

#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <span>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace p = strix::profile;

namespace {

int failures = 0;
void require(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

constexpr std::string_view kLever = "fixture-w4a4@1";
// FAST cell "a8", TURBO cell "a4": both lossy, both ready.
constexpr std::array<p::Lever, 1> kRows{{
    {kLever, "STRIX_FIXTURE_W4A4", "", {"", "a8", "", "a4"}, "off", p::Kind::bit_changing,
     {true, true, false, true}, {true, true, false, true}, "", "off", "fixture receipts"},
}};

p::Resolution resolve(const p::Engine& engine, p::Name tier, p::Regime regime = {}) {
    p::Inputs in;
    in.profile = tier;
    in.regime = regime;
    in.env = [](const char*) -> const char* { return nullptr; };
    return p::resolve(engine, in);
}

// The lower arm a timing row was measured over (LadderRow::over).
constexpr p::LadderRow measured_over(p::LadderRow row, std::string_view arm) {
    row.over = arm;
    return row;
}
// The references of the one-lever fixture (LadderArms::over spelling): a FAST row is measured over EXACT (the builtin
// off), a TURBO row over the FAST arm where its FAST cell runs.
constexpr std::string_view kExactArm = "ladder-arm/v2;fixture-w4a4@1=off";
constexpr std::string_view kFastArm = "ladder-arm/v2;fixture-w4a4@1=a8";

void verdicts() {
    constexpr p::LadderRow fast_pf = measured_over({kLever, "a8", p::Name::fast, p::Phase::prefill, "dev", 1, 8192, 1, 1, 1.20,
                                                    0.02, "r/fast-pf"}, kExactArm);
    constexpr p::LadderRow turbo_pf = measured_over({kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 2049, 4096, 1,
                                                     1, 1.30, 0.02, "r/turbo-pf-4k"}, kFastArm);
    constexpr p::LadderRow turbo_pf_noise = measured_over({kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 4097,
                                                           8192, 1, 1, 1.003, 0.02, "r/turbo-pf-8k"}, kFastArm);
    // Measured where FAST is not measured (decode at 8-16 lanes): over the builtin FAST arm, which runs there.
    constexpr p::LadderRow turbo_dec_slow = measured_over({kLever, "a4", p::Name::turbo, p::Phase::decode, "dev", 1, 65536, 8,
                                                           16, 0.95, 0.02, "r/turbo-dec"}, kExactArm);
    constexpr p::LadderRow fast_dec_slow = measured_over({kLever, "a8", p::Name::fast, p::Phase::decode, "dev", 1, 65536, 1,
                                                          1, 0.97, 0.01, "r/fast-dec"}, kExactArm);
    constexpr p::LadderRow turbo_dec_b1 = measured_over({kLever, "a4", p::Name::turbo, p::Phase::decode, "dev", 1, 65536,
                                                         1, 1, 1.25, 0.02, "r/turbo-dec-b1"}, kFastArm);
    constexpr std::array rows{fast_pf, turbo_pf, turbo_pf_noise, turbo_dec_slow, fast_dec_slow, turbo_dec_b1};
    const p::Engine engine{"fixture", "", kRows, false, "fixture-turbo-v1", {}, {}, rows};
    require(p::ladder_defect(engine).empty(), "fixture ladder is well formed: " + p::ladder_defect(engine));

    static_assert(p::ladder_gain(1.008, 0.002), "a measured sub-percent gain is enough");
    static_assert(!p::ladder_gain(1.015, 0.02) && p::ladder_gain(1.021, 0.02), "stated margin");
    static_assert(!p::ladder_gain(std::numeric_limits<double>::infinity(), 0.02));
    static_assert(!p::ladder_gain(1.2, std::numeric_limits<double>::quiet_NaN()));
    const auto check = [&](p::Name tier, p::Regime regime, std::string_view value, std::string_view route) {
        const auto actual = resolve(engine, tier, regime);
        const auto* entry = actual.find(kLever);
        require(entry && entry->value == value, "effective production value " + std::string(value));
        if (route.empty()) return;
        require(entry && entry->ladder.find(route) != std::string::npos &&
                    actual.describe().find(entry->ladder) != std::string::npos &&
                    actual.json().find(entry->ladder) != std::string::npos, "route in CLI and server receipts");
        const auto exported = p::exports(actual, std::set<std::string>{});
        const auto setting = std::ranges::find_if(exported, [](const auto& item) {
            return item.first == "STRIX_FIXTURE_W4A4";
        });
        require(value == "off" ? setting == exported.end() : setting != exported.end() && setting->second == value,
                "only the effective value is exported");
    };
    check(p::Name::turbo, {p::Phase::prefill, 4096, 1, "dev"}, "a4", "applies x1.300");
    check(p::Name::turbo, {p::Phase::prefill, 2049, 1, "dev"}, "a4", "-> turbo=a4");
    check(p::Name::turbo, {p::Phase::prefill, 2048, 1, "dev"}, "a8", "-> fast=a8");
    check(p::Name::turbo, {p::Phase::prefill, 4097, 1, "dev"}, "a8", "within noise");
    check(p::Name::turbo, {p::Phase::prefill, 8193, 1, "dev"}, "off", "unmeasured regime");
    check(p::Name::turbo, {p::Phase::prefill, 4096, 1, "other"}, "off", "unmeasured regime");
    check(p::Name::turbo, {p::Phase::prefill, 4096, 2, "dev"}, "off", "unmeasured regime");
    check(p::Name::turbo, {p::Phase::prefill, 0, 1, "dev"}, "off", "unmeasured regime");
    check(p::Name::turbo, {p::Phase::decode, 4096, 8, "dev"}, "off", "slower");
    // FAST is measured slower in decode at one lane, so the FAST arm that runs there is not the arm the TURBO row was
    // measured over: the row is stale and the cell does not run.
    check(p::Name::turbo, {p::Phase::decode, 4096, 1, "dev"}, "off", "timing rows measured over another fast arm");
    check(p::Name::fast, {p::Phase::prefill, 4096, 1, "dev"}, "a8", "-> fast=a8");
    check(p::Name::standard, {p::Phase::prefill, 4096, 1, "dev"}, "off", "");

    // A refused nonempty TURBO cell leaves the measured FAST cell and its exported status intact.
    auto refused_rows = kRows;
    refused_rows[0].ready[3] = false;
    const p::Engine refused{"refused", "", refused_rows, false, "refused-v1", {}, {}, rows};
    const p::Regime measured{p::Phase::prefill, 4096, 1, "dev"};
    const auto fast_run = resolve(refused, p::Name::fast, measured);
    const auto exported = [](const p::Resolution& run) {
        const auto env = p::exports(run, std::set<std::string>{});
        return std::ranges::find(env, std::pair<std::string, std::string>{"STRIX_FIXTURE_W4A4", "a8"}) != env.end();
    };
    require(exported(fast_run), "measured FAST fixture exports a8");
    for (const bool runnable : {true, false}) {
        refused_rows[0].runnable[3] = runnable;
        const auto turbo_run = resolve(refused, p::Name::turbo, measured);
        require(turbo_run.find(kLever)->value == "a8" &&
                    turbo_run.find(kLever)->status == fast_run.find(kLever)->status &&
                    turbo_run.find(kLever)->ladder.find("turbo=a4 refused") != std::string::npos &&
                    exported(turbo_run),
                "pending/unavailable TURBO replacement keeps the FAST export and records the refusal");
    }

    refused_rows[0].value[3] = "a8";
    require(resolve(refused, p::Name::turbo, measured).find(kLever)->status == p::Status::pending,
            "a same-value TURBO binding explicitly unavailable in this composition remains pending");

    // An explicit setting is the user's own: it runs as set in every regime.
    p::Inputs in;
    in.profile = p::Name::turbo;
    in.env = [](const char* key) -> const char* {
        return std::string_view(key) == "STRIX_FIXTURE_W4A4" ? "a4" : nullptr;
    };
    const auto opted = p::resolve(engine, in);
    require(opted.find(kLever)->value == "a4", "explicit environment wins in an unmeasured regime");
    in.cli.emplace_back(kLever, "a8");
    require(p::resolve(engine, in).find(kLever)->value == "a8", "explicit CLI beats explicit environment");

    // An engine without ladder rows: every lossy cell resolves to the builtin; an exact lever keeps its cell.
    const p::Engine bare{"bare", "", kRows, false, "bare-turbo-v1"};
    require(resolve(bare, p::Name::turbo).find(kLever)->value == "off",
            "no ladder rows: lossy cells do not apply");
    constexpr std::array<p::Lever, 1> exact_rows{{
        {"sched", "STRIX_FIXTURE_SCHED", "", {"", "2", "", ""}, "1", p::Kind::exact, {true, true, false, true},
         {true, true, false, true}, "", "", "fixture scheduling receipt"},
    }};
    const p::Engine scheduling{"sched", "", exact_rows, false, "sched-turbo-v1"};
    require(resolve(scheduling, p::Name::turbo).find("sched")->value == "2",
            "exact (scheduling) levers are outside the ladder");
}

void defects() {
    const auto defect = [](p::LadderRow row) {
        const std::array rows{row};
        const p::Engine engine{"fixture", "", kRows, false, "fixture-turbo-v1", {}, {}, rows};
        return p::ladder_defect(engine);
    };
    const p::LadderRow good = measured_over({kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 1, 4096, 1, 1, 1.3, 0.02,
                                             "r"}, kFastArm);
    require(defect(good).empty(), "good row");
    auto row = good;
    row.value = {};
    require(!defect(row).empty(), "timing row naming no value (it would match every value)");
    row = good;
    for (const std::string_view unbound : {std::string_view{}, std::string_view{"fixture-w4a4@1=a8"},
                                           std::string_view{"ladder-arm/v1;fixture-w4a4@1=a8"},
                                           std::string_view{"ladder-arm/v2"}}) {
        row.over = unbound;
        require(!defect(row).empty(), "timing row without a versioned reference: " + std::string(unbound));
    }
    row = good;
    row.lever = "missing";
    require(!defect(row).empty(), "row naming no lever");
    row = good;
    row.value = "a8";
    require(!defect(row).empty(), "row value that is not the tier's cell");
    row = good;
    row.tier = p::Name::standard;
    require(!defect(row).empty(), "exact column row");
    row = good;
    row.receipt = "";
    require(!defect(row).empty(), "row without receipt");
    row = good;
    row.device = "";
    require(!defect(row).empty(), "row without device");
    row = good;
    row.speedup = 0.0;
    require(!defect(row).empty(), "unmeasured speedup");
    row = good;
    row.noise = std::numeric_limits<double>::quiet_NaN();
    require(!defect(row).empty(), "NaN noise");
    row = good;
    row.min_context = 0;
    require(!defect(row).empty(), "band covering the unknown context");
    row = good;
    row.min_lanes = 4;
    row.max_lanes = 2;
    require(!defect(row).empty(), "unordered lane band");
}

// Walk all declared rows, including currently pending candidates. Marking the candidate ready here makes this
// test fail if a first binding can export the value before a timing receipt is registered.
void walk(const p::Engine& registered) {
    require(p::ladder_defect(registered).empty(), p::ladder_defect(registered));
    // Every shipped timing row names the reference that runs at its own band: a row left behind by a changed builtin,
    // schedule or lower cell would silently stop deciding.
    for (const auto& row : registered.ladder) {
        if (row.incumbent || !row.joint.empty()) continue;
        const auto actual = resolve(registered, row.tier, {row.phase, row.min_context, row.min_lanes, row.device});
        const auto* entry = actual.find(row.lever);
        require(entry != nullptr && entry->ladder.find("measured over another") == std::string::npos,
                std::string(registered.tag) + "/" + std::string(row.lever) + " shipped timing row is stale (" +
                    std::string(row.receipt) + "): reference " + (entry ? entry->ladder_reference : std::string()));
    }
    std::vector<p::Lever> rows(registered.table.begin(), registered.table.end());
    auto engine = registered;
    engine.table = rows;
    p::Inputs implicit;
    implicit.env = [](const char*) -> const char* { return nullptr; };
    const auto exact = resolve(engine, p::Name::standard);
    require(p::resolve(engine, implicit).fingerprint() == exact.fingerprint(), "implicit native default equals EXACT");
    for (const auto& entry : exact.entries) {
        const auto wanted = entry.lever->value[0];
        const auto baseline = !wanted.empty() && entry.lever->ready[0] ? wanted : entry.lever->builtin;
        require(entry.value == baseline && entry.ladder.empty(), "EXACT retains the registered value");
    }
    for (auto& row : rows) {
        if (row.kind == p::Kind::exact) continue;
        row.ready.fill(true);
        row.provisional.fill(false);
    }
    auto probes = p::ladder_probes(engine);
    probes.push_back({p::Phase::prefill, 0, 1, {}});
    for (const auto& regime : probes) {
        for (const auto tier : {p::Name::fast, p::Name::turbo}) {
            if (tier == p::Name::turbo && engine.turbo_version.empty()) continue;
            const auto actual = resolve(engine, tier, regime);
            for (const auto& entry : actual.entries) {
                if (entry.lever->kind == p::Kind::exact || (entry.value == entry.lever->builtin && !p::applied(entry.status))) continue;
                require(entry.ladder.find("applies") != std::string::npos ||
                            entry.ladder.find("incumbent") != std::string::npos,
                        std::string(engine.tag) + "/" + std::string(entry.lever->name) + " applied without the ladder");
            }
        }
    }
}

void cuda_cli_server() {
    namespace qo = strix::cuda::qwen_options;
    p::LadderRow timing{"qsa_mma", "1", p::Name::fast, p::Phase::prefill, "fixture", 4096, 4096,
                        1, 1, 1.02, 0.005, "fixture-qsa"};
    qo::Request request;
    request.engine.model_gguf = "model.gguf";
    request.engine.tables_dir = "tables";
    request.engine.rope_table = "rope";
    request.inputs.env = [](const char*) -> const char* { return nullptr; };
    request.profile = p::Name::fast;
    // The reference the adapter runs under the cell (LadderArms::over): the row is measured over it.
    const auto reference_of = [&](std::string_view lever) {
        const auto* entry = qo::resolve(request, qo::Entry::cli).resolution.find(lever);
        return entry ? entry->ladder_reference : std::string();
    };
    request.inputs.regime = {p::Phase::prefill, 4096, 1, "fixture"};
    const std::string qsa_reference = reference_of("qsa_mma");
    require(qsa_reference.starts_with("ladder-arm/v2;"), "Qwen CUDA reports the reference: " + qsa_reference);
    timing.over = qsa_reference;
    request.inputs.ladder = std::span(&timing, 1);
    for (const auto context : {4096U, 4097U}) {
        request.inputs.regime = {p::Phase::prefill, context, 1, "fixture"};
        const auto cli = qo::resolve(request, qo::Entry::cli);
        const auto server = qo::resolve(request, qo::Entry::serve);
        // Inside the timing band the row decides; outside it the incumbent grant keeps the cell main selected.
        require(cli.choice.qsa_mma && server.choice.qsa_mma == cli.choice.qsa_mma &&
                    cli.resolution.json().find(context == 4096U ? "applies x1.020" : "incumbent") != std::string::npos,
                "Qwen CUDA CLI and server select the same measured or incumbent kernel form");
        require(cli.resolution.json() == server.resolution.json() &&
                    server.resolution.json().find("strix-ladder/v1 qsa_mma") != std::string::npos,
                "Qwen CUDA CLI and server emit the same decision");
    }
    // Provide evidence for a non-incumbent row through the native options adapter.
    const p::RegimeGate gate{"moe_route_bf16", "1", 4096, 4097, "fixture gate"};
    p::LadderRow candidate{"moe_route_bf16", "1", p::Name::fast, p::Phase::prefill, "fixture", 4096, 4096,
                           1, 1, 1.02, 0.005, "fixture route storage"};
    request.inputs.cli = {{"expert_fast", "off"}};
    request.inputs.regimes = std::span(&gate, 1);
    request.inputs.ladder = {};
    request.inputs.max_prompt_tokens = 4096;
    request.inputs.regime = {p::Phase::prefill, 4096, 1, "fixture"};
    const std::string route_reference = reference_of("moe_route_bf16");
    require(route_reference.find("expert_fast=off") != std::string::npos,
            "the explicit expert_fast=off is part of the reference: " + route_reference);
    candidate.over = route_reference;
    request.inputs.ladder = std::span(&candidate, 1);
    const auto profile_selects_storage = [](const qo::Resolved& r) {
        const auto* entry = r.resolution.find("moe_route_bf16");
        return entry && entry->value == "1" && entry->status == p::Status::set;
    };
    require(profile_selects_storage(qo::resolve(request, qo::Entry::cli)) &&
            profile_selects_storage(qo::resolve(request, qo::Entry::serve)),
            "a measured non-incumbent candidate runs with the same known prompt on both entries");
    ++request.inputs.regime.context;
    require(!profile_selects_storage(qo::resolve(request, qo::Entry::cli)),
            "the neighboring context declines the candidate");
    request.inputs.regime = p::startup_prefill_regime("fixture");
    const auto startup = qo::resolve(request, qo::Entry::serve);
    require(!profile_selects_storage(startup) && startup.choice.qsa_mma,
            "startup declines the non-incumbent timing row and retains the incumbent form");
}

void custom_adapters() {
    constexpr p::Regime regime{p::Phase::prefill, 4096, 1, "dev"};
    constexpr p::LadderRow sample = measured_over({kLever, "a8", p::Name::fast, p::Phase::prefill, "dev", 4096, 4096, 1, 1,
                                                   1.008, 0.002, "r/fast"}, kExactArm);
    for (unsigned probe = 0; probe < 4; ++probe) {
        auto row = sample;
        switch (probe) {
        case 1: row.receipt = {}; break;
        case 2: row.speedup = std::numeric_limits<double>::infinity(); break;
        case 3: row.noise = std::numeric_limits<double>::quiet_NaN(); break;
        default: break;
        }
        strix::qwen::profile::Inputs in;
        in.env = [](const char*) -> const char* { return nullptr; };
        in.regime = regime;
        in.ladder = std::span(&row, 1);
        const auto qwen = strix::qwen::profile::resolve(p::Name::fast, in, kRows);
        const auto* cell = qwen.find(kLever);
        require(cell && cell->value == (probe == 0 ? "a8" : "off") &&
                    qwen.json().find("strix-ladder/v1") != std::string::npos, "Qwen native selection checks timing");
    }

    namespace g = strix::gemma4::profile;
    g::Lever gemma{.name = kLever, .value = {"", "a8", ""}, .builtin = "off", .ready = {true, true, true},
                   .kind = p::Kind::bit_changing, .apps = unsigned(g::App::engine)};
    const auto g4 = g::resolve(p::Name::fast, g::App::engine, {}, std::span(&gemma, 1));
    require(g4.find(kLever)->value == "off", "Gemma cannot select a lossy unmeasured row");

    namespace r = strix::ds41::rtx::profile;
    r::Lever rtx{.name = kLever, .value = {"", "a8", ""}, .builtin = "off", .ready = {true, true, true},
                 .kind = p::Kind::bit_changing, .apps = unsigned(r::App::replica), .source = "fixture"};
    const auto ds = r::resolve(p::Name::fast, false, r::App::replica, {}, std::span(&rtx, 1));
    require(ds.find(kLever)->value == "off", "DeepSeek RTX cannot select a lossy unmeasured row");

    namespace glm = strix::glm::rtxpf::profile;
    glm::Lever candidate{.name = kLever, .env = "", .value = {"", "a8", ""}, .builtin = "off",
                         .ready = {true, true, true}, .kind = glm::Kind::bit_changing,
                         .apps = unsigned(glm::App::decode_service), .source = "fixture"};
    const auto gpu = glm::resolve(glm::Name::fast, false, glm::App::decode_service, {}, std::span(&candidate, 1));
    require(gpu.find(kLever)->value == "off", "GLM RTX cannot select a lossy unmeasured row");
}

// Incumbent preservation: every production entry point resolves FAST / TURBO (and refuses) exactly as before the
// ladder rule. One line per resolved entry: "<entry point> <tier> <regime> <lever>=<value> s<status>". The golden
// (tests/golden/profile_selections.txt) pins incumbent execution values and their receipt statuses.
// Additions and intended receipt changes are reviewed when the golden is updated.
// The individual adapter suites exercise --allow-pending separately.
template <class Res>
void lines(std::vector<std::string>& out, const std::string& where, const Res& res) {
    for (const auto& e : res.entries)
        if (e.lever != nullptr)
            out.push_back(where + " " + std::string(e.lever->name) + "=" + std::string(e.value) + " s" +
                          std::to_string(static_cast<int>(e.status)));
}
template <class F>
void guarded(std::vector<std::string>& out, const std::string& where, F&& f) {
    try {
        f();
    } catch (const std::exception& error) {
        out.push_back(where + " throws " + error.what());
    }
}

std::vector<std::string> selections() {
    namespace q = strix::qwen;
    namespace qd = strix::qwen35;
    namespace mm = strix::minimax::defaults;
    const auto none = [](const char*) -> const char* { return nullptr; };
    const auto tier = [](int t) { return std::string(t == 1 ? "fast" : "turbo"); };
    std::vector<std::string> out;
    const p::Engine engines[] = {
        {"deepseek41-pair", "", strix::deepseek41::pair_defaults::profile_table, false, "ds41-turbo-v1", {}, {},
         strix::deepseek41::pair_defaults::ladder_rows},
        {"deepseek41-rtx", "", strix::ds41::rtx::profile::shared_rows()},
        strix::glm::rank_defaults::product_profile_engine,
        mm::profile_engine(),
        {"qwen35", "", qd::profile::table, false, qd::profile::kTurboVersion},
        qd::hip_profile::engine(qd::hip_profile::App::decode),
        qd::hip_profile::engine(qd::hip_profile::App::spec),
        strix::hy4::profile::engine(),
        strix::inkling2::kMiniProfile,
    };
    for (const auto& engine : engines)
        for (int t : {1, 3}) {
            const auto where = "A " + std::string(engine.tag) + " " + tier(t);
            guarded(out, where, [&] {
                p::Inputs in;
                in.profile = static_cast<p::Name>(t);
                in.env = none;
                lines(out, where, p::resolve(engine, in));
            });
        }
    for (int t : {1, 3})
        for (std::uint64_t tokens : {0ULL, 1ULL, 4096ULL, 8192ULL, 8193ULL, 32768ULL}) {
            const auto where = "B qwen-hip " + tier(t) + " ctx=" + std::to_string(tokens);
            guarded(out, where, [&] {
                q::profile::Inputs in;
                in.env = none;
                in.max_prompt_tokens = tokens;
                in.regimes = q::profile::fast_regimes;
                in.regime = {p::Phase::prefill, tokens, 1, "gfx1151"};
                lines(out, where, q::profile::resolve(static_cast<p::Name>(t), in));
            });
            const auto cuda = "B qwen-cuda " + tier(t);
            guarded(out, cuda, [&] {
                q::profile::Inputs in;
                in.env = none;
                lines(out, cuda, q::cuda_profile::resolve(static_cast<p::Name>(t), in));
            });
        }
    for (int t : {1, 3})
        for (int w = 0; w < 3; ++w)
            for (int r = 0; r < 2; ++r)
                for (std::uint64_t tokens : {0ULL, 4096ULL, 8191ULL, 8192ULL, 32768ULL}) {
                    const auto where = "C qwen35-cuda-dense " + tier(t) + " w" + std::to_string(w) + " r" +
                                       std::to_string(r) + " ctx=" + std::to_string(tokens);
                    guarded(out, where, [&] {
                        qd::profile::Regime regime;
                        regime.weights = static_cast<qd::profile::Weights>(w);
                        regime.route = static_cast<qd::profile::Route>(r);
                        regime.prompt_tokens = tokens;
                        regime.device = "sm_120";
                        regime.arch = "sm_120";
                        lines(out, where, qd::profile::resolve(static_cast<p::Name>(t), regime, {})->resolution);
                    });
                }
    for (int t : {1, 3}) {
        namespace g = strix::gemma4::profile;
        for (auto app : {g::App::engine, g::App::decode_bench, g::App::prefill_bench}) {
            const auto where = "D gemma4 " + tier(t) + " app" + std::to_string(static_cast<unsigned>(app));
            guarded(out, where, [&] { lines(out, where, g::resolve(static_cast<p::Name>(t), app, {})); });
        }
        namespace r = strix::ds41::rtx::profile;
        for (auto app : {r::App::replica, r::App::prefill_service}) {
            const auto where = "E ds41-rtx " + tier(t) + " app" + std::to_string(static_cast<unsigned>(app));
            guarded(out, where, [&] { lines(out, where, r::resolve(static_cast<p::Name>(t), false, app, {})); });
        }
    }
    {
        namespace gl = strix::glm::rtxpf::profile;
        guarded(out, "F glm-rtx fast", [&] {
            lines(out, "F glm-rtx fast", gl::resolve(gl::Name::fast, false, gl::App::decode_service, {}));
        });
    }
    for (int t : {1, 3})
        for (int d = 0; d < 2; ++d)
            for (int topo = 0; topo < 2; ++topo)
                for (std::uint64_t tokens : {0ULL, 4096ULL, 8192ULL, 32768ULL}) {
                    const auto where = "G hy4 " + tier(t) + " d" + std::to_string(d) + " t" + std::to_string(topo) +
                                       " ctx=" + std::to_string(tokens);
                    guarded(out, where, [&] {
                        const strix::hy4::profile::Regime regime{static_cast<strix::hy4::profile::Device>(d),
                                                                 static_cast<strix::hy4::profile::Topology>(topo), tokens};
                        p::Inputs in;
                        in.profile = static_cast<p::Name>(t);
                        in.env = none;
                        lines(out, where, strix::hy4::profile::resolve(regime, in)->resolution);
                    });
                }
    for (const char* word : {"fast", "turbo"}) {
        const auto where = std::string("H minimax ") + word;
        guarded(out, where, [&] {
            std::map<std::string, std::string> env{{mm::kProductProfileEnv, word}};
            const auto lookup = [&env](const char* k) -> const char* {
                const auto it = env.find(k);
                return it == env.end() ? nullptr : it->second.c_str();
            };
            const auto policy = mm::prefill_policy(lookup);
            out.push_back(where + " msa_flash=" + std::to_string(policy.msa_flash) + " router_wmma=" +
                          std::to_string(int(policy.router_wmma)) + " bundle=" +
                          std::to_string(int(mm::fast_profile_selects_bundle(lookup))) + " id=" +
                          mm::prefill_identity(lookup));
            lines(out, where, mm::profile_resolution(lookup));
        });
    }
    return out;
}

void incumbents(const char* golden, bool write) {
    const auto now = selections();
    if (write) {
        std::ofstream file(golden, std::ios::binary);
        for (const auto& line : now) file << line << '\n';
        return;
    }
    std::ifstream file(golden, std::ios::binary);
    require(static_cast<bool>(file), std::string("golden selections not readable: ") + golden);
    std::vector<std::string> before;
    for (std::string line; std::getline(file, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) before.push_back(line);
    }
    require(before.size() == now.size(),
            "selection count " + std::to_string(now.size()) + " != golden " + std::to_string(before.size()));
    for (std::size_t i = 0; i < before.size() && i < now.size(); ++i)
        require(before[i] == now[i], "selection moved: '" + before[i] + "' -> '" + now[i] + "'");
}
// TURBO columns that leave a FAST member off are compatibility opt-outs kept from before the ladder rule, not measured
// claims: the nested rule (TURBO over the admitted FAST set) does not hold for them. The exact list is pinned here; a new
// opt-out must be a deliberate change of this list, and an off cell without the opt-out receipt is a defect.
void compatibility_opt_outs(std::span<const p::Engine> engines) {
    std::set<std::string> found;
    for (const auto& engine : engines)
        for (const auto& row : engine.ladder) {
            if (!row.incumbent || row.tier != p::Name::turbo) continue;
            const p::Lever* lever = nullptr;
            for (const auto& candidate : engine.table)
                if (candidate.name == row.lever) lever = &candidate;
            if (lever == nullptr) continue;
            const auto fast = lever->value[static_cast<unsigned>(p::Name::fast)];
            const auto turbo = lever->value[static_cast<unsigned>(p::Name::turbo)];
            const bool leaves_fast_off = !fast.empty() && !turbo.empty() && turbo != fast && turbo == lever->builtin;
            const bool opt_out_receipt = row.receipt == p::kTurboOffBeforeLadder;
            require(leaves_fast_off == opt_out_receipt, std::string(engine.tag) + "/" + std::string(row.lever) +
                    ": a TURBO cell that leaves a FAST member off must carry the compatibility opt-out receipt");
            if (opt_out_receipt) found.insert(std::string(engine.tag) + "/" + std::string(row.lever));
        }
    const std::set<std::string> expected{"deepseek41-pair/fast_prefill", "deepseek41-pair/engram_f16wmma",
                                         "deepseek41-pair/shared_f16", "deepseek41-pair/moe_f16wmma",
                                         "minimax-pair/router_wmma"};
    if (found != expected) {
        std::string got;
        for (const auto& item : found) got += " " + item;
        require(false, "compatibility opt-out list changed:" + got);
    }
}

// A paired decision is admitted only by a row measured on the pair (LadderRow::joint): single-cell rows of either half
// never read it, and the joint row is invisible to the plain judge.
void paired_rows() {
    const p::Regime regime{p::Phase::prefill, 8192, 1, "dev"};
    p::LadderRow joint{"shadow", "fp8", p::Name::turbo, p::Phase::prefill, "dev", 8192, 8192, 1, 1, 1.10, 0.01, "pair receipt"};
    joint.joint = "reader";
    const p::JointLadderScope scope{"device-id", "fast@1", "candidate-base@1", "shadow@1", "reader@1",
                                    {8192, 32768, 1, 2048, 16, 1, 0, 128}, "on"};
    joint.joint_scope = scope;
    p::LadderRow plain = joint;
    plain.joint = {};
    plain.receipt = "shadow alone";
    const p::LadderRow both[] = {joint, plain};
    require(p::ladder_judge(std::span(&plain, 1), "shadow", "fp8", p::Name::turbo, regime, "reader", scope).rung !=
                p::Rung::applies, "a single-cell row admitted the pair");
    require(p::ladder_judge(std::span(&joint, 1), "shadow", "fp8", p::Name::turbo, regime).rung != p::Rung::applies,
            "a joint row decided the plain cell");
    const auto verdict = p::ladder_judge(both, "shadow", "fp8", p::Name::turbo, regime, "reader", scope);
    require(verdict.rung == p::Rung::applies && verdict.row == &both[0], "the joint row decides the pair");
    require(p::ladder_judge(both, "shadow", "fp8", p::Name::turbo, regime, "other", scope).rung != p::Rung::applies,
            "a joint row decided another partner");
    p::LadderRow slow = joint;
    slow.speedup = 0.97;
    require(p::ladder_judge(std::span(&slow, 1), "shadow", "fp8", p::Name::turbo, regime, "reader", scope).rung == p::Rung::slower,
            "a losing joint row must not admit the pair");
}
// A new gate row (a wider prompt band for a qualified cell) is not an incumbent grant: the cell keeps the tier below
// beyond the frozen band until a matching timing row is registered.
void hy4_new_quality_rows() {
    namespace h = strix::hy4::profile;
    const auto none = [](const char*) -> const char* { return nullptr; };
    p::Inputs in;
    in.profile = p::Name::fast;
    in.env = none;
    const h::Regime beyond{h::Device::gfx1151, h::Topology::one_node, 40000};
    const h::Regime inside{h::Device::gfx1151, h::Topology::one_node, 32568};
    require(h::resolve(inside, in)->find("prompt_rows")->value == "4096", "frozen grant: the qualified band still runs");
    std::vector<h::Qualified> rows(std::begin(h::qualified), std::end(h::qualified));
    rows.push_back({h::Device::gfx1151, h::Topology::one_node, {"prompt_rows", "4096", 0, 65536, "fixture wider gate row"}});
    rows.push_back({h::Device::gfx1151, h::Topology::one_node, {"pf_mha", "1", 0, 65536, "fixture wider gate row"}});
    const auto wide = h::resolve(beyond, in, rows);
    require(wide->find("prompt_rows")->value == "0" && wide->find("pf_mha")->value == "0",
            "a new quality row beyond the frozen band must leave the cell at the lower tier without a timing row");
    require(h::resolve(inside, in, rows)->find("prompt_rows")->value == "4096", "the frozen band keeps its cell");
    // Exercise the same adapter as CLI and serve, including its input shape and topology checks.
    h::Measured timings[]{{h::Topology::one_node,
        {"prompt_rows", "4096", p::Name::fast, p::Phase::prefill, "gfx1151", 40000, 65536, 1, 1, 1.05, 0.01,
         "fixture timing"}}};
    // Measured over the reference the adapter runs there (LadderArms::over).
    const std::string hy4_reference = h::resolve(beyond, in, rows)->find("prompt_rows")->ladder_reference;
    require(hy4_reference.starts_with("ladder-arm/v2;"), "Hy4 reports the reference: " + hy4_reference);
    timings[0].timing.over = hy4_reference;
    const auto resolved = h::resolve(beyond, in, rows, h::incumbent_grants, timings);
    require(resolved->find("prompt_rows")->value == "4096", "a matching timing row admits the new band");
    auto other = beyond;
    --other.prompt_tokens;
    require(h::resolve(other, in, rows, h::incumbent_grants, timings)->find("prompt_rows")->value == "0",
            "neighboring prompt declines the same measured row");
    other = beyond;
    other.topology = h::Topology::pair;
    rows.push_back({h::Device::gfx1151, h::Topology::pair, {"prompt_rows", "4096", 0, 65536, "fixture wider pair gate"}});
    require(h::resolve(other, in, rows, h::incumbent_grants, timings)->find("prompt_rows")->value == "0",
            "pair topology declines the one-node measured row");
}

// Ordinary timing rows decide only the comparison they measured (LadderRow::over), --allow-pending never runs a measured
// loser, and a startup resolution (context 0) runs grants only.
void bound_evidence() {
    const auto run = [](const p::Engine& engine, p::Name tier, p::Regime regime, p::Pairs cli = {},
                        bool allow_pending = false) {
        p::Inputs in;
        in.profile = tier;
        in.regime = regime;
        in.cli = std::move(cli);
        in.allow_pending = allow_pending;
        in.env = [](const char*) -> const char* { return nullptr; };
        return p::resolve(engine, in);
    };
    const auto value_of = [](const p::Resolution& res, std::string_view lever) {
        const auto* entry = res.find(lever);
        return entry ? entry->value : std::string("<none>");
    };
    const auto route_of = [](const p::Resolution& res, std::string_view lever) {
        const auto* entry = res.find(lever);
        return entry ? entry->ladder : std::string();
    };
    const p::Regime at4k{p::Phase::prefill, 4096, 1, "dev"};
    constexpr p::LadderRow fast_a8 = measured_over({kLever, "a8", p::Name::fast, p::Phase::prefill, "dev", 1, 8192, 1, 1, 1.20,
                                                    0.02, "r/a8"}, kExactArm);
    constexpr p::LadderRow turbo_a4 = measured_over({kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 1, 8192, 1, 1,
                                                     1.30, 0.02, "r/a4-over-a8"}, kFastArm);
    {
        // The review case: a new FAST value qualifies and gets its own FAST record; the TURBO record (denominator a8)
        // is left untouched. It must not decide over the new FAST arm.
        constexpr std::array rows{fast_a8, turbo_a4};
        const p::Engine before{"bound", "", kRows, false, "bound-v1", {}, {}, rows};
        require(value_of(run(before, p::Name::turbo, at4k), kLever) == "a4", "TURBO row over the running FAST arm applies");
        auto requalified = kRows;
        requalified[0].value[1] = "a6";
        constexpr p::LadderRow fast_a6 = measured_over({kLever, "a6", p::Name::fast, p::Phase::prefill, "dev", 1, 8192, 1, 1,
                                                        1.15, 0.02, "r/a6"}, kExactArm);
        constexpr std::array after_rows{fast_a6, turbo_a4};
        const p::Engine after{"bound", "", requalified, false, "bound-v1", {}, {}, after_rows};
        const auto res = run(after, p::Name::turbo, at4k);
        require(value_of(res, kLever) == "a6", "stale TURBO row (measured over a8) keeps the new FAST cell a6");
        require(route_of(res, kLever).find("timing rows measured over another fast arm") != std::string::npos,
                "the route names the stale evidence: " + route_of(res, kLever));
        // Re-timed over the new FAST arm, the TURBO cell runs again.
        constexpr std::array retimed_rows{fast_a6, measured_over({kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 1,
                                                                  8192, 1, 1, 1.25, 0.02, "r/a4-over-a6"},
                                                                 "ladder-arm/v2;fixture-w4a4@1=a6")};
        const p::Engine retimed{"bound", "", requalified, false, "bound-v1", {}, {}, retimed_rows};
        require(value_of(run(retimed, p::Name::turbo, at4k), kLever) == "a4", "TURBO row re-timed over a6 applies");
    }
    {
        // An explicit setting of another lossy lever changes the lower arm: the TURBO row measured over the profile's
        // FAST arm no longer decides.
        constexpr std::string_view kOther = "fixture-other@1";
        std::array<p::Lever, 2> table{kRows[0], kRows[0]};
        table[1].name = kOther;
        table[1].env = "STRIX_FIXTURE_OTHER";
        table[1].value = {"", "on", "", ""};
        constexpr std::string_view kBoth = "ladder-arm/v2;fixture-w4a4@1=a8,fixture-other@1=on";
        const std::array rows{measured_over(fast_a8, "ladder-arm/v2;fixture-w4a4@1=off,fixture-other@1=off"),
                              p::incumbent_cell(kOther, "on", p::Name::fast, "r/other-grant"),
                              measured_over({kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 1, 8192, 1, 1, 1.30,
                                             0.02, "r/a4-over-both"}, kBoth)};
        const p::Engine engine{"override", "", table, false, "override-v1", {}, {}, rows};
        const auto plain = run(engine, p::Name::turbo, at4k);
        require(value_of(plain, kLever) == "a4" && value_of(plain, kOther) == "on", "TURBO over the full FAST arm applies");
        const auto overridden = run(engine, p::Name::turbo, at4k, {{std::string(kOther), "off"}});
        require(value_of(overridden, kLever) == "a8", "an override of another FAST lever makes the TURBO row stale");
    }
    {
        // --allow-pending runs only cells no measurement decides: measured slower / within noise stay below.
        for (const double speedup : {0.98, 1.005}) {
            const std::array rows{measured_over({kLever, "a8", p::Name::fast, p::Phase::prefill, "dev", 1, 4096, 1, 1, speedup,
                                                 0.01, "r/a8-loser"}, kExactArm)};
            const p::Engine engine{"bench", "", kRows, false, "bench-v1", {}, {}, rows};
            const auto res = run(engine, p::Name::fast, at4k, {}, true);
            require(value_of(res, kLever) == "off" &&
                        route_of(res, kLever).find("unqualified benchmark") == std::string::npos,
                    "allow-pending never runs a measured FAST loser x" + std::to_string(speedup));
            const auto unmeasured = run(engine, p::Name::fast, {p::Phase::prefill, 8192, 1, "dev"}, {}, true);
            require(value_of(unmeasured, kLever) == "a8" &&
                        route_of(unmeasured, kLever).find("unqualified benchmark") != std::string::npos,
                    "allow-pending runs the unmeasured regime as a benchmark arm");
        }
    }
    {
        // Startup (context 0): no timing band holds an unknown regime, so a grant keeps its cell (timed or not) and an
        // ungranted timed cell stays below; once the prompt is known the rows decide.
        const std::array granted{p::incumbent_cell(kLever, "a8", p::Name::fast, "r/grant")};
        const p::Engine grant_only{"grant", "", kRows, false, "grant-v1", {}, {}, granted};
        const p::Regime startup = p::startup_prefill_regime("dev");
        require(value_of(run(grant_only, p::Name::fast, startup), kLever) == "a8", "untimed grant keeps its cell at startup");
        const std::array timed_grant{p::incumbent_cell(kLever, "a8", p::Name::fast, "r/grant"), fast_a8};
        const p::Engine timed{"grant", "", kRows, false, "grant-v1", {}, {}, timed_grant};
        require(value_of(run(timed, p::Name::fast, startup), kLever) == "a8", "timed grant keeps its cell at startup");
        const std::array timed_only{fast_a8};
        const p::Engine ungranted{"grant", "", kRows, false, "grant-v1", {}, {}, timed_only};
        require(value_of(run(ungranted, p::Name::fast, startup), kLever) == "off", "ungranted timed cell stays below at startup");
        require(value_of(run(ungranted, p::Name::fast, at4k), kLever) == "a8", "the row decides once the prompt is known");
        require(value_of(run(timed, p::Name::fast, {p::Phase::prefill, 16384, 1, "dev"}), kLever) == "a8",
                "outside the timed bands the grant keeps its cell");
    }
    {
        // r51: the reference names every lever with its effective value, builtins included. A FAST row measured over
        // the builtin off stops deciding once the builtin itself is another execution.
        constexpr std::array rows{fast_a8};
        const p::Engine engine{"builtin", "", kRows, false, "builtin-v1", {}, {}, rows};
        require(value_of(run(engine, p::Name::fast, at4k), kLever) == "a8", "FAST row over the builtin off applies");
        auto rebuilt = kRows;
        rebuilt[0].builtin = "a2";
        const p::Engine changed{"builtin", "", rebuilt, false, "builtin-v1", {}, {}, rows};
        const auto res = run(changed, p::Name::fast, at4k);
        require(value_of(res, kLever) == "a2" &&
                    route_of(res, kLever).find("timing rows measured over another default arm") != std::string::npos,
                "a changed builtin makes the FAST receipt stale: " + route_of(res, kLever));
    }
    {
        // An exact schedule lever is part of the reference: overriding it changes the timing denominator.
        constexpr std::string_view kSchedule = "fixture-schedule@1";
        std::array<p::Lever, 2> table{kRows[0], kRows[0]};
        table[1].name = kSchedule;
        table[1].env = "STRIX_FIXTURE_SCHEDULE";
        table[1].value = {"", "", "", ""};
        table[1].builtin = "1";
        table[1].off = {};
        table[1].kind = p::Kind::exact;
        const std::array rows{measured_over(fast_a8, "ladder-arm/v2;fixture-w4a4@1=off,fixture-schedule@1=1")};
        const p::Engine engine{"schedule", "", table, false, "schedule-v1", {}, {}, rows};
        require(value_of(run(engine, p::Name::fast, at4k), kLever) == "a8", "FAST row over the default schedule applies");
        const auto res = run(engine, p::Name::fast, at4k, {{std::string(kSchedule), "2"}});
        require(value_of(res, kLever) == "off" && value_of(res, kSchedule) == "2",
                "an exact schedule override makes the FAST receipt stale: " + route_of(res, kLever));
        require(res.find(kLever)->ladder_reference == "ladder-arm/v2;fixture-w4a4@1=off,fixture-schedule@1=2",
                "the reference spells the override: " + res.find(kLever)->ladder_reference);
    }
    {
        // A timing row names its value: an empty value covers no cell (it would also match a6).
        p::LadderRow any = fast_a8;
        any.value = {};
        const std::array rows{any};
        const p::Engine engine{"any", "", kRows, false, "any-v1", {}, {}, rows};
        require(value_of(run(engine, p::Name::fast, at4k), kLever) == "off" && !p::ladder_defect(engine).empty(),
                "an ordinary row without a value never decides and is a defect");
    }
    {
        // A TURBO row explicitly measured over the builtin off applies when the FAST a8 candidate is measured slower
        // and FAST runs off (the reference that actually runs; no projected-column recheck).
        const std::array rows{measured_over({kLever, "a8", p::Name::fast, p::Phase::prefill, "dev", 1, 8192, 1, 1, 0.95,
                                             0.01, "r/a8-loses"}, kExactArm),
                              measured_over({kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 1, 8192, 1, 1, 1.30,
                                             0.02, "r/a4-over-off"}, kExactArm)};
        const p::Engine engine{"over-off", "", kRows, false, "over-off-v1", {}, {}, rows};
        require(value_of(run(engine, p::Name::fast, at4k), kLever) == "off", "FAST a8 measured slower runs off");
        require(value_of(run(engine, p::Name::turbo, at4k), kLever) == "a4", "TURBO row over the running off applies");
    }
    {
        // Two isolated TURBO wins over the same FAST arm do not admit their unmeasured combination: cells of one tier
        // compose in table order, so the second cell decides only on a row measured over the first.
        constexpr std::string_view kSecond = "fixture-second@1";
        std::array<p::Lever, 2> table{kRows[0], kRows[0]};
        table[0].value = {"", "", "", "a4"};
        table[1].name = kSecond;
        table[1].env = "STRIX_FIXTURE_SECOND";
        table[1].value = {"", "", "", "b4"};
        constexpr std::string_view kBase = "ladder-arm/v2;fixture-w4a4@1=off,fixture-second@1=off";
        constexpr std::string_view kWithFirst = "ladder-arm/v2;fixture-w4a4@1=a4,fixture-second@1=off";
        const auto turbo_row = [](std::string_view lever, std::string_view value, double speedup, std::string_view over,
                                  std::string_view receipt) {
            return measured_over({lever, value, p::Name::turbo, p::Phase::prefill, "dev", 4096, 4096, 1, 1, speedup, 0.02,
                                  receipt}, over);
        };
        const std::array isolated{turbo_row(kLever, "a4", 1.10, kBase, "r/a4-alone"),
                                  turbo_row(kSecond, "b4", 1.10, kBase, "r/b4-alone")};
        const p::Engine alone{"compose", "", table, false, "compose-v1", {}, {}, isolated};
        const auto two = run(alone, p::Name::turbo, at4k);
        require(value_of(two, kLever) == "a4" && value_of(two, kSecond) == "off" &&
                    route_of(two, kSecond).find("timing rows measured over another fast arm") != std::string::npos,
                "two isolated wins: only the first measured cell runs: " + route_of(two, kSecond));
        const std::array losing{isolated[0], isolated[1], turbo_row(kSecond, "b4", 0.95, kWithFirst, "r/b4-joint-loses")};
        const p::Engine joint_loses{"compose", "", table, false, "compose-v1", {}, {}, losing};
        const auto lost = run(joint_loses, p::Name::turbo, at4k);
        require(value_of(lost, kLever) == "a4" && value_of(lost, kSecond) == "off" &&
                    route_of(lost, kSecond).find("slower") != std::string::npos,
                "a losing joint result keeps the combination off: " + route_of(lost, kSecond));
        const std::array winning{isolated[0], turbo_row(kSecond, "b4", 1.05, kWithFirst, "r/b4-joint-wins")};
        const p::Engine joint_wins{"compose", "", table, false, "compose-v1", {}, {}, winning};
        const auto both = run(joint_wins, p::Name::turbo, at4k);
        require(value_of(both, kLever) == "a4" && value_of(both, kSecond) == "b4",
                "a combination measured over the first cell runs both");
    }
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc < 3) return 2;
    if (argc > 3 && std::string_view(argv[3]) == "--write") {
        incumbents(argv[2], true);
        return 0;
    }
    incumbents(argv[2], false);
    verdicts();
    bound_evidence();
    defects();
    custom_adapters();
    cuda_cli_server();
    namespace q = strix::qwen;
    namespace qd = strix::qwen35;
    namespace mm = strix::minimax::defaults;
    const p::Engine engines[] = {
        {"qwen", "", q::profile::levers(), false, q::profile::kTurboVersion, {}, {}, q::profile::ladder_rows},
        {"qwen-cuda", "", q::cuda_profile::levers(), false, q::profile::kTurboVersion, {}, {}, q::cuda_profile::ladder_rows},
        {"deepseek41-pair", "", strix::deepseek41::pair_defaults::profile_table, false, "ds41-turbo-v1", {}, {},
         strix::deepseek41::pair_defaults::ladder_rows},
        {"deepseek41-rtx", "", strix::ds41::rtx::profile::shared_rows()},
        strix::glm::rank_defaults::product_profile_engine,
        mm::profile_engine(),
        {"qwen35", "", qd::profile::table, false, qd::profile::kTurboVersion},
        qd::hip_profile::engine(qd::hip_profile::App::decode),
        qd::hip_profile::engine(qd::hip_profile::App::spec),
        strix::hy4::profile::engine(),
        strix::inkling2::kMiniProfile,
    };
    for (const auto& registered : strix::model_registry::load(argv[1])) {
        const auto tag = registered.profile_tag;
        require(tag == "gemma4" || std::ranges::any_of(engines, [&](const auto& engine) {
                    return engine.tag == tag || (tag == "hy4" && engine.tag == "hy4-hip");
                }), "profile registry entry must be covered: " + tag);
    }
    compatibility_opt_outs(engines);
    paired_rows();
    hy4_new_quality_rows();
    for (const auto& engine : engines) walk(engine);
    if (failures) return 1;
    std::puts("profile ladder tests: PASS");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
