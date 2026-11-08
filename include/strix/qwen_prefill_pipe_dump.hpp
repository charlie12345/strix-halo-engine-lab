#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::qwen_pp {
inline bool dump_enabled() {
    const char* directory = std::getenv("STRIX_QWEN_PF_PIPE_DUMP");
    return directory && directory[0] != '\0';
}
inline std::uint64_t dump_hash(std::span<const std::byte> bytes,
    std::uint64_t hash = 14695981039346656037ULL) {
    for (const auto value : bytes) {
        hash ^= std::to_integer<std::uint8_t>(value);
        hash *= 1099511628211ULL;
    }
    return hash;
}
inline std::string dump_stem(std::string_view owner, std::uint64_t position, std::uint32_t rows) {
    return std::string(owner) + "-p" + std::to_string(position) + "-n" + std::to_string(rows);
}
// The caller supplies only generated filenames; the directory is a private debug artifact destination.
inline void dump_write(const std::string& name, std::span<const std::byte> bytes) {
    const char* directory = std::getenv("STRIX_QWEN_PF_PIPE_DUMP");
    if (!directory || directory[0] == '\0') return;
    std::filesystem::create_directories(directory);
    if (bytes.size() > static_cast<std::size_t>((std::numeric_limits<std::streamsize>::max)()))
        throw std::runtime_error("PP dump exceeds stream size");
    std::ofstream output(std::filesystem::path(directory) / name, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open PP dump");
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    output.close();
    if (!output) throw std::runtime_error("cannot write PP dump");
}
inline void dump_text(const std::string& name, const std::string& value) {
    dump_write(name, std::as_bytes(std::span<const char>(value.data(), value.size())));
}
} // namespace strix::qwen_pp
