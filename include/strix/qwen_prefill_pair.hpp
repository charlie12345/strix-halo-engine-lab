#pragma once
#include "strix/qwen_prefill_pipeline.hpp"
#include "strix/pair/pipe.hpp"
#include <array>
#include <algorithm>
#include <memory>
#include <cstdlib>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <span>
#include <thread>
#include <chrono>
#include <cstdio>

namespace strix::qwen_pp {
inline bool pipe_enabled() {
    const char* value = std::getenv("STRIX_QWEN_PF_PIPE");
    return value && std::string_view(value) == "1";
}
inline bool pipe_profile_enabled() {
    const char* value = std::getenv("STRIX_QWEN_PF_PIPE_PROFILE");
    return value && std::string_view(value) == "1";
}
// The pipe, staging contract and CRC are the shared two-stage seam (strix/pair/pipe.hpp); Qwen adds
// only its seam header contract: magic, rows, cut and its HC words.
using pair::seam_piece_bytes;
using pair::SeamCrc;
using pair::PipeStaging;
using pair::PipeFrame;
std::shared_ptr<PipeStaging> make_hip_pipe_staging(std::size_t capacity);
class Pipe : public pair::Pipe {
public:
    Pipe(unsigned rank, std::uint64_t count, Send send, Receive receive, std::function<void()> abort,
        std::uint32_t cut = 24, std::function<std::shared_ptr<PipeStaging>()> staging = {}, Checksum checksum = {},
        TransferPiece transfer_piece = {})
        : pair::Pipe(rank, count, std::move(send), std::move(receive), std::move(abort),
              options(checked_cut(cut), count), std::move(staging), std::move(checksum), std::move(transfer_piece)) {}
private:
    static std::uint32_t checked_cut(std::uint32_t cut) {
        if (cut == 0U || cut >= 48U) throw std::invalid_argument("pair prefill invalid cut");
        return cut;
    }
    static pair::PipeOptions options(std::uint32_t cut, std::uint64_t count) {
        pair::PipeOptions value;
        value.profile_label = "qwen-pair-profile";
        value.profile = pipe_profile_enabled();
        value.validate = [cut, count](const pair::PipeHeader& header, std::uint64_t sequence) {
            if (header[0] != Seam::magic || header[1] != sequence || header[42] != 3U ||
                header[41] != count || header[3] == 0U || header[3] > 16384U ||
                header[4] != cut || header[7] > 32U)
                throw std::runtime_error("pair prefill invalid seam header or cut mismatch");
            return static_cast<std::size_t>(header[3]) * Seam::row_bytes;
        };
        return value;
    }
};
// One pipeline per process; all native tile owners share its ordered chunk sequence.
struct PipeContext {
    Pipe* pipe = nullptr;
    unsigned rank = 0;
    std::uint64_t sequence = 0;
    std::uint64_t identity = 0;
    std::uint32_t cut = 24;
};
inline PipeContext pipe_context;
inline std::uint64_t token_hash(std::span<const std::uint32_t> tokens) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto token : tokens) { hash ^= token; hash *= 1099511628211ULL; }
    return hash;
}
} // namespace strix::qwen_pp
