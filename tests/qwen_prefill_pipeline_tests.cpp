#include "strix/qwen_prefill_pipeline.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>

int main() {
    using strix::qwen_pp::Seam;
    const auto check = [](bool pass) { if (!pass) throw std::runtime_error("layer-split layout check failed"); };
    check(Seam::row_bytes == 25616U && Seam::hc_bytes == 20480U && Seam::residual_bytes == 5120U);
    Seam seam(3);
    check(seam.bytes.size() == 76848U);
    for (std::uint32_t row = 0; row < 3U; ++row) {
        for (std::size_t element = 0; element < (Seam::hc_bytes + Seam::residual_bytes) / 2U; ++element) {
            const auto bf16_bits = static_cast<std::uint16_t>((row << 13U) ^ element);
            std::memcpy(seam.row(row) + element * 2U, &bf16_bits, sizeof(bf16_bits));
        }
        for (std::size_t coefficient = 0; coefficient < 4U; ++coefficient) {
            const float value = static_cast<float>(row * 4U + coefficient);
            std::memcpy(seam.row(row) + Seam::hc_bytes + Seam::residual_bytes + coefficient * 4U, &value, sizeof(value));
        }
    }
    const auto saved = seam.bytes;
    Seam restored(3);
    std::memcpy(restored.bytes.data(), saved.data(), saved.size());
    check(restored.bytes == saved);
    for (std::uint32_t row = 0; row < 3U; ++row) {
        check(restored.row(row) == restored.bytes.data() + static_cast<std::size_t>(row) * 25616U);
        for (std::size_t element = 0; element < 12800U; ++element) {
            std::uint16_t actual = 0;
            std::memcpy(&actual, restored.row(row) + element * 2U, sizeof(actual));
            check(actual == static_cast<std::uint16_t>((row << 13U) ^ element));
        }
        for (std::size_t coefficient = 0; coefficient < 4U; ++coefficient) {
            float actual = 0;
            std::memcpy(&actual, restored.row(row) + 25600U + coefficient * 4U, sizeof(actual));
            check(actual == static_cast<float>(row * 4U + coefficient));
        }
    }
    check(!strix::qwen_pp::parse_range(nullptr, 48).enabled);
    const auto range = strix::qwen_pp::parse_range("0:24", 48);
    check(range.enabled && range.begin == 0 && range.end == 24);
    check(strix::qwen_pp::parse_range("24:48", 48).begin == 24);
    for (const char* bad : {"", "24", "-1:24", "24:24", "25:24", "0:49", "0:24x", "0:24:48", " 0:24"}) {
        bool rejected = false;
        try { static_cast<void>(strix::qwen_pp::parse_range(bad, 48)); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected);
    }
    std::cout << "layer-split seam layout and range parser PASS\n";
}
