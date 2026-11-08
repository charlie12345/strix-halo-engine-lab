#pragma once
#include "glm/engine/glm_kvshard_layout.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>

namespace strix::glm::kvshard {
// Diagnostic files are private run artifacts. One file per epoch/layer/position
// and rank avoids inter-rank writes and retains global selection order.
inline void write_decode_dump(const std::string& directory, unsigned rank, std::uint64_t epoch,
                              unsigned layer, std::uint64_t position,
                              const std::vector<std::uint32_t>& tokens,
                              const std::vector<std::uint32_t>& keys) {
    const auto root = std::filesystem::path(directory) / ("rank-" + std::to_string(rank));
    std::filesystem::create_directories(root);
    const auto stem = "epoch-" + std::to_string(epoch) + "-layer-" + std::to_string(layer) +
                      "-pos-" + std::to_string(position);
    std::ofstream selected(root / (stem + "-selected.csv"));
    std::ofstream scores(root / (stem + "-pools.csv"));
    if (!selected || !scores) throw std::runtime_error("kvshard: cannot open debug dump");
    selected << "# " << numerical_identity << '\n' << "order,global_id,owner,local_token\n";
    for (std::size_t i = 0; i < tokens.size(); ++i)
        selected << i << ',' << tokens[i] << ',' << owner(tokens[i]) << ',' << local_token(tokens[i]) << '\n';
    scores << "# " << numerical_identity << '\n' << "global_pool,owner,local_pool,key_hex,score_hexfloat\n";
    for (std::size_t pool = 0; pool < keys.size(); ++pool) {
        const auto key = keys[pool];
        const std::uint32_t bits = (key & 0x80000000U) ? key & 0x7fffffffU : ~key;
        float score = 0.0F;
        std::memcpy(&score, &bits, sizeof(score));
        scores << pool << ',' << pool % 2 << ',' << pool / 2 << ',' << std::hex << std::setw(8) <<
            std::setfill('0') << key << std::dec << ',' << std::hexfloat << score << '\n';
    }
    selected.close(); scores.close();
    if (!selected || !scores) throw std::runtime_error("kvshard: cannot write debug dump");
}
// First 16 contiguous local-head MLA output values, before the output projection.
// Raw IEEE-754 bits preserve signed zero, infinities and NaN payloads exactly.
inline void write_attention_dump(const std::string& directory, unsigned rank, std::uint64_t epoch,
                                 unsigned layer, std::uint64_t position, const std::array<float, 16>& values) {
    const auto root = std::filesystem::path(directory) / ("rank-" + std::to_string(rank));
    std::filesystem::create_directories(root);
    const auto stem = "epoch-" + std::to_string(epoch) + "-layer-" + std::to_string(layer) +
                      "-pos-" + std::to_string(position);
    std::ofstream output(root / (stem + "-attention.csv"));
    if (!output) throw std::runtime_error("kvshard: cannot open attention dump");
    output << "# " << numerical_identity << '\n' << "index,fp32_hex\n";
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &values[i], sizeof(bits));
        output << i << ',' << std::hex << std::setw(8) << std::setfill('0') << bits << std::dec << '\n';
    }
    output.close();
    if (!output) throw std::runtime_error("kvshard: cannot write attention dump");
}
} // namespace strix::glm::kvshard
