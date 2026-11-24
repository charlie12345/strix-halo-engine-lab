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

void verdicts() {
    constexpr p::LadderRow fast_pf{kLever, "a8", p::Name::fast, p::Phase::prefill, "dev", 1, 8192, 1, 1, 1.20, 0.02, "r/fast-pf"};
    constexpr p::LadderRow turbo_pf{kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 2049, 4096, 1, 1, 1.30, 0.02,
                                    "r/turbo-pf-4k"};
    constexpr p::LadderRow turbo_pf_noise{kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 4097, 8192, 1, 1, 1.003,
                                          0.02, "r/turbo-pf-8k"};
    constexpr p::LadderRow turbo_dec_slow{kLever, "a4", p::Name::turbo, p::Phase::decode, "dev", 1, 65536, 8, 16, 0.95,
                                          0.02, "r/turbo-dec"};
    constexpr p::LadderRow fast_dec_slow{kLever, "a8", p::Name::fast, p::Phase::decode, "dev", 1, 65536, 1, 1, 0.97, 0.01,
                                         "r/fast-dec"};
    constexpr p::LadderRow turbo_dec_b1{kLever, "a4", p::Name::turbo, p::Phase::decode, "dev", 1, 65536, 1, 1, 1.25, 0.02,
                                        "r/turbo-dec-b1"};
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
    check(p::Name::turbo, {p::Phase::decode, 4096, 1, "dev"}, "off", "measured over fast=a8");
    check(p::Name::fast, {p::Phase::prefill, 4096, 1, "dev"}, "a8", "-> fast=a8");
    check(p::Name::standard, {p::Phase::prefill, 4096, 1, "dev"}, "off", "");

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
    const p::LadderRow good{kLever, "a4", p::Name::turbo, p::Phase::prefill, "dev", 1, 4096, 1, 1, 1.3, 0.02, "r"};
    require(defect(good).empty(), "good row");
    auto row = good;
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
    const p::LadderRow timing{"qsa_mma", "1", p::Name::fast, p::Phase::prefill, "fixture", 4096, 4096,
                              1, 1, 1.02, 0.005, "fixture-qsa"};
    qo::Request request;
    request.engine.model_gguf = "model.gguf";
    request.engine.tables_dir = "tables";
    request.engine.rope_table = "rope";
    request.inputs.env = [](const char*) -> const char* { return nullptr; };
    request.inputs.ladder = std::span(&timing, 1);
    request.profile = p::Name::fast;
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
}

void custom_adapters() {
    constexpr p::Regime regime{p::Phase::prefill, 4096, 1, "dev"};
    constexpr p::LadderRow sample{kLever, "a8", p::Name::fast, p::Phase::prefill, "dev", 4096, 4096, 1, 1,
                                 1.008, 0.002, "r/fast"};
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
// (tests/golden/profile_selections.txt) holds the lines of the tree before the ladder rule; a change that moves a
// default selection rewrites it on purpose (strix-profile-ladder-tests REGISTRY GOLDEN --write) and says why.
// --allow-pending is outside it: an unqualified cell needs a timing row since the ladder rule.
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
    p::LadderRow plain = joint;
    plain.joint = {};
    plain.receipt = "shadow alone";
    const p::LadderRow both[] = {joint, plain};
    require(p::ladder_judge(std::span(&plain, 1), "shadow", "fp8", p::Name::turbo, regime, "reader").rung !=
                p::Rung::applies, "a single-cell row admitted the pair");
    require(p::ladder_judge(std::span(&joint, 1), "shadow", "fp8", p::Name::turbo, regime).rung != p::Rung::applies,
            "a joint row decided the plain cell");
    const auto verdict = p::ladder_judge(both, "shadow", "fp8", p::Name::turbo, regime, "reader");
    require(verdict.rung == p::Rung::applies && verdict.row == &both[0], "the joint row decides the pair");
    require(p::ladder_judge(both, "shadow", "fp8", p::Name::turbo, regime, "other").rung != p::Rung::applies,
            "a joint row decided another partner");
    p::LadderRow slow = joint;
    slow.speedup = 0.97;
    require(p::ladder_judge(std::span(&slow, 1), "shadow", "fp8", p::Name::turbo, regime, "reader").rung == p::Rung::slower,
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
    // the matching timing row admits it
    p::LadderRow timing{"prompt_rows", "4096", p::Name::fast, p::Phase::prefill, "gfx1151", 32569, 65536, 1, 1, 1.05, 0.01,
                        "fixture timing"};
    auto timed = h::resolve(beyond, in, rows);
    std::vector<p::LadderRow> ladder(timed->ladder);
    ladder.push_back(timing);
    timed->engine.ladder = std::span<const p::LadderRow>(ladder);
    in.regime = {p::Phase::prefill, 40000, 1, "gfx1151"};
    const auto resolved = p::resolve(timed->engine, in);
    require(resolved.find("prompt_rows")->value == "4096", "a matching timing row admits the new band");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    if (argc > 3 && std::string_view(argv[3]) == "--write") {
        incumbents(argv[2], true);
        return 0;
    }
    incumbents(argv[2], false);
    verdicts();
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
}
