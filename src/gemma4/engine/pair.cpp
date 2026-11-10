// Two-node Gemma 4 route (see pair.hpp). Transport, pipe, staging and CLI are the shared pair modules
// (strix/pair/*.hpp, xport/hip_pipe_staging.hpp); this file only binds them to the coordinator.
#include "pair.hpp"
#include "gemma4/prefill/pair.hpp"
#include "gemma4/prefill/prefill.hpp"
#include "strix/pair/link.hpp"
#include "xport/hip_pipe_staging.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace strix::gemma4::engine {
namespace {
using Json = strix::serve::Json;
constexpr std::string_view prefix = "STRIX_G4_PAIR";
constexpr std::uint16_t tcp_port = 29873;
unsigned env_cut(const ModelConfig& config) {
    const char* value = std::getenv("STRIX_G4_PAIR_CUT");
    if (!value || !*value) return pair_default_cut;
    char* end = nullptr;
    const auto cut = std::strtoul(value, &end, 10);
    if (*end || cut == 0 || cut >= config.layers) throw std::invalid_argument("STRIX_G4_PAIR_CUT must be in [1, layers)");
    return unsigned(cut);
}
bool env_on(const char* name) { const char* v = std::getenv(name); return v && std::string_view(v) == "1"; }
std::uint64_t fnv(std::uint64_t hash, std::string_view bytes) {
    for (const unsigned char b : bytes) { hash ^= b; hash *= 1099511628211ULL; }
    return hash;
}
struct StageReset { ~StageReset() { prefill::pair_stage = {}; } };
} // namespace
bool pair_enabled() { return env_on("STRIX_G4_PAIR"); }
std::size_t pair_chunk_count(std::size_t prompt, std::size_t chunk) {
    std::size_t count = 0, processed = 0;
    while (const auto n = prompt_chunk_rows(prompt, processed, processed, chunk)) { processed += n; ++count; }
    return count;
}
int run_pair(const Options& options, const Runtime& runtime, std::span<const std::uint32_t> prompt,
             const std::function<void(std::span<const std::uint32_t>, strix::serve::Json)>& emit) {
    using Clock = std::chrono::steady_clock;
    const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
    auto& backend = *runtime.backend;
    const auto& config = backend.config();
    const auto chunk = backend.prefill_chunk_rows();
    if (!chunk || options.streams != 1) throw std::invalid_argument("pair route needs the device many-row prefill and --streams 1");
    const auto rank = pair::env_rank(prefix);
    const auto cut = env_cut(config);
    const auto count = pair_chunk_count(prompt.size(), chunk);
    if (!count) throw std::invalid_argument("pair route needs a prompt longer than one decode tail (> 9 tokens)");
    // Both ranks must run the same model, tables, profile and cut: their words go in every seam header.
    std::uint64_t identity = fnv(14695981039346656037ULL, backend.identity().dump());
    identity = fnv(identity, std::to_string(cut) + ":" + std::to_string(prompt.size()));
    auto link = pair::open_link(rank, prefix, tcp_port);
    const auto linked = Clock::now();
    const auto capacity = prefill::pair_seam_bytes(config, cut, unsigned(chunk));
    pair::PipeOptions pipe_options;
    pipe_options.profile_label = "g4-pair-profile";
    pipe_options.profile = env_on("STRIX_G4_PAIR_PROFILE");
    pipe_options.validate = [&, identity, count, cut](const pair::PipeHeader& h, std::uint64_t sequence) {
        using W = prefill::PairWord;
        if (h[W::magic] != prefill::pair_seam_magic || h[1] != sequence || h[42] != 3U || h[41] != count ||
            h[W::cut] != cut || h[W::identity] != identity || h[W::rows] == 0U || h[W::rows] > chunk ||
            h[W::bytes] != prefill::pair_seam_bytes(config, cut, unsigned(h[W::rows])))
            throw std::runtime_error("gemma4 pair seam header mismatch (model, profile, cut, prompt or chunk)");
        return std::size_t(h[W::bytes]);
    };
    pair::Pipe pipe(rank, count,
        [&](std::span<const std::byte> data) { link->send(data); },
        [&](std::span<std::byte> data) { link->receive(data); },
        [&] { link->abort("gemma4 pair prefill stopped"); }, std::move(pipe_options),
        [&] { return pair::make_hip_pipe_staging(capacity, [] { return static_cast<hipStream_t>(prefill::pair_stage.queue); }); },
        [](std::span<const std::byte> data) { return dist::crc32c(data); });
    StageReset reset;
    prefill::pair_stage = {&pipe, rank, cut, 0, identity, nullptr, 0, 0};
    auto pair_json = Json::object();
    pair_json.set("rank", std::uint64_t(rank)); pair_json.set("cut", std::uint64_t(cut));
    pair_json.set("chunks", std::uint64_t(count)); pair_json.set("seam_bytes_per_chunk", std::uint64_t(capacity));
    if (rank == 0U) {
        // Stage A: the coordinator's chunk schedule for this prompt, layers [0, cut) only. No decode.
        auto state = backend.create_stream(std::uint32_t(prompt.size() + options.request.max_tokens));
        std::size_t processed = 0;
        auto chunks = Json::array();
        const auto start = Clock::now();
        while (const auto n = prompt_chunk_rows(prompt.size(), processed, state->frontier(), chunk)) {
            const auto begin = Clock::now();
            auto tx = state->begin();
            (void)backend.prefill_rows(*tx, prompt.subspan(processed, n), false, false);
            tx->commit(n);
            processed += n;
            chunks.push(ms(Clock::now() - begin));
        }
        pipe.finish(count);
        pair_json.set("stage_ms", ms(Clock::now() - start)); pair_json.set("chunk_ms", chunks);
        pair_json.set("seal_wait_ms", prefill::pair_stage.seal_ms);
        pair_json.set("link_to_done_ms", ms(Clock::now() - linked));
        auto receipt = Json::object(); receipt.set("pair", pair_json); receipt.set("prompt_tokens", std::uint64_t(prompt.size()));
        emit({}, receipt);
        return 0;
    }
    Coordinator coordinator(backend, runtime.drafts);
    auto request = options.request; request.prompt.assign(prompt.begin(), prompt.end());
    const auto slot = coordinator.submit(std::move(request));
    coordinator.run();
    pipe.finish(count);
    const auto& result = coordinator.result(slot);
    auto receipt = coordinator.receipt(slot); receipt.set("stream", std::uint64_t(slot));
    pair_json.set("seam_wait_ms", prefill::pair_stage.wait_ms);
    pair_json.set("link_to_done_ms", ms(Clock::now() - linked));
    receipt.set("pair", pair_json);
    emit(result.ids, receipt);
    return result.status == Status::failed ? 1 : 0;
}
} // namespace strix::gemma4::engine
