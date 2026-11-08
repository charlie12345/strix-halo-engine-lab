#include <hip/hip_runtime.h>
namespace strix { inline hipStream_t hip_execution_stream() { return nullptr; } }
#include "../src/hip_qwen_prefill_pair_staging.hpp"
#include <iostream>

namespace {
void require(bool condition) {
    if (!condition) throw std::runtime_error("PP seam publication assertion failed");
}
std::size_t stress(bool device_only) {
    using namespace strix::qwen_pp;
    seam_visibility::device_only = device_only;
    std::mt19937 random(0x505033U);
    std::size_t mismatches = 0;
    for (unsigned trial = 0; trial < 24U; ++trial) {
        // Random short pieces, full pieces and partial final pieces. Two live
        // staging slots model immutable snapshots while source compute resumes.
        const auto size = static_cast<std::size_t>(random() % 65535U) + 1U +
            (trial % 3U == 0U ? seam_piece_bytes : 0U);
        std::array<std::shared_ptr<PipeStaging>, 2> slots;
        std::vector<std::byte> source(size);
        for (auto& byte : source) byte = static_cast<std::byte>(random() & 255U);
        const auto expected = source;
        for (auto& slot : slots) {
            slot = make_hip_pipe_staging(size);
            auto host = slot->resize(size);
            slot->copy(true, 0U, source.data(), size);
            slot->seal(size);
            // The last partial copy is allowed to complete before earlier
            // pieces; completion still does not publish coarse host memory.
            seam_visibility::complete_shuffled(seam_visibility::last_stream, random);
            require(std::all_of(seam_visibility::last_stream->copies.begin(),
                seam_visibility::last_stream->copies.end(), [](const auto& copy) { return copy.complete; }));
            std::fill(source.begin(), source.end(), std::byte{0x5a});
            std::vector<std::size_t> offsets;
            for (std::size_t offset = 0; offset < size; offset += seam_piece_bytes) offsets.push_back(offset);
            std::shuffle(offsets.begin(), offsets.end(), random);
            for (const auto offset : offsets) {
                slot->await_piece(offset);
                const auto length = std::min(seam_piece_bytes, size - offset);
                const auto piece = host.subspan(offset, length);
                SeamCrc actual_crc, expected_crc;
                actual_crc.update(piece);
                expected_crc.update(std::span(expected).subspan(offset, length));
                if (actual_crc.value() != expected_crc.value() ||
                    !std::equal(piece.begin(), piece.end(), expected.begin() + static_cast<std::ptrdiff_t>(offset))) ++mismatches;
            }
            source = expected;
        }
    }
    return mismatches;
}
}
int main() {
    try {
        const auto before = stress(true);
        const auto after = stress(false);
        require(before > 0U && after == 0U);
        std::cout << "PASS: 24 seeded sizes x 2 slots; shuffled copy/fence order, source reuse and partial tails; device-only mismatches="
            << before << ", system-release mismatches=" << after << " (host model, not device qualification)\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
