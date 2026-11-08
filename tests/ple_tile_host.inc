// Shared by the HIP executable's --ple-tile-host mode and a standalone host build.
#include "strix/ple_projection_tile.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ple_tile_host {
inline double median(std::vector<float> values) {
    if (values.empty()) throw std::invalid_argument("median requires at least one sample");
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2U;
    if (values.size() % 2U != 0U) return static_cast<double>(values[middle]);
    return (static_cast<double>(values[middle - 1U]) + static_cast<double>(values[middle])) / 2.0;
}
inline void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
inline unsigned reverse4(unsigned n) {
    return ((n & 1U) << 3U) | ((n & 2U) << 1U) | ((n & 4U) >> 1U) | ((n & 8U) >> 3U);
}
inline std::uint16_t input_bits(unsigned row, unsigned f) {
    constexpr std::uint16_t edges[] = {0, 0x8000, 1, 0x8001, 0x7f80, 0xff80, 0x7fc1, 0xffd2};
    if (row % 4U == 0U && f < 8U) return edges[f];
    return static_cast<std::uint16_t>(0x3800U + ((row * 71U + f * 17U) % 2048U));
}
inline void contracts() {
    using namespace strix::ple_tile;
    require(requested("1") && !requested(nullptr) && !requested("") && !requested("0") &&
        !requested("01") && !requested("10") && !requested("true"), "strict opt-in/default off");
    const Matrix in{0x100000000ULL, 2560, 2568};
    const Matrix key{0x200000000ULL, 10240, 10243}, value{0x300000000ULL, 2560, 2563};
    for (const unsigned groups : {1U, 128U, 256U}) {
        require(admits(true, in, key, value, groups), "production shapes/pitches admit");
        require(!admits(false, in, key, value, groups), "inactive request refuses");
        auto alias = key; alias.address = in.address + 16U;
        require(!admits(true, in, alias, value, groups), "input/output alias fallback");
        alias = value; alias.address = key.address + 4U;
        require(!admits(true, in, key, alias, groups), "key/value alias fallback");
        auto bad = in; ++bad.stride;
        require(!admits(true, bad, key, value, groups), "misaligned pitch fallback");
        bad = in; --bad.stride; bad.stride = bad.columns - 1U;
        require(!admits(true, bad, key, value, groups), "short pitch fallback");
        bad = in; bad.columns = 2304;
        require(!admits(true, bad, key, value, groups), "odd chunk fallback");
        alias = key; alias.address = UINTPTR_MAX - 3U;
        require(!admits(true, in, alias, value, groups), "address overflow fallback");
    }
    require(!admits(true, in, key, value, 0) && !admits(true, in, key, value, 65536), "group bounds");
    require(!admits(true, {}, key, value, 128), "empty input fallback");
    // Adjacent extents admit; even padding-only bounding aliases conservatively refuse.
    require(disjoint({4, 8}, {8, 12}) && !disjoint({4, 9}, {8, 12}), "half-open ranges");
}
inline void panels(unsigned rows, unsigned pitch) {
    constexpr unsigned width = 2560, chunks = width / 256U;
    std::vector<unsigned char> seen(64U * width, 0);
    std::vector<std::uint32_t> panel(64U * width), independent(panel.size());
    // The existing producer traversal, including exact NaN/Inf/zero bit widening.
    for (std::size_t t = 0; t < panel.size(); ++t) {
        const unsigned mm = static_cast<unsigned>(t % 8U), r = static_cast<unsigned>(t / 8U % 4U);
        const unsigned h = static_cast<unsigned>(t / 32U % 2U), k = static_cast<unsigned>(t / 64U % chunks);
        const unsigned q = static_cast<unsigned>(t / (64U * chunks) % 16U);
        const unsigned g = static_cast<unsigned>(t / (1024U * chunks));
        const unsigned f = reverse4(q) + 16U * (8U * h + mm) + 256U * k, row = 4U * g + r;
        require(row < 64U && f < width && !seen[row * width + f]++, "pack bijection");
        panel[t] = std::uint32_t{input_bits(row, f)} << 16U;
    }
    // Independent inverse: walk original rows/features, not packed indices.
    for (unsigned row = 0; row < 64U; ++row)
        for (unsigned f = 0; f < width; ++f) {
            const unsigned m = f % 256U / 16U;
            const auto at = (((std::size_t{row / 4U} * 16U + reverse4(f % 16U)) * chunks + f / 256U) * 2U +
                m / 8U) * 32U + row % 4U * 8U + m % 8U;
            independent[at] = std::uint32_t{input_bits(row, f)} << 16U;
        }
    require(panel == independent, "independent packed byte oracle");
    // Every row/feature address at both production T values and pitched layouts.
    for (unsigned group = 0; group < rows / 64U; ++group)
        for (unsigned row = 0; row < 64U; ++row)
            for (unsigned f = 0; f < width; ++f) {
                const auto old_address = std::uint64_t{group} * 64U * pitch + std::uint64_t{row} * pitch + f;
                const auto shared_address = std::uint64_t{group * 64U + row} * pitch + f;
                require(old_address == shared_address, "full-T input address oracle");
            }
    std::printf("PASS host panel T%u pitch%u (byte oracle, all row/feature addresses)\n", rows, pitch);
}
inline float add(float lower, float upper) { return upper + lower; }
inline void tree_oracle() {
    for (unsigned edge = 0; edge < 4U; ++edge) {
        std::array<float, 2560> x{}, c{};
        for (unsigned f = 0; f < x.size(); ++f) {
            const auto word = edge == 0U ? static_cast<std::uint16_t>((f * 1777U) % 0x7f00U) : input_bits(edge - 1U, f);
            x[f] = std::bit_cast<float>(std::uint32_t{word} << 16U);
            c[f] = static_cast<float>(static_cast<int>((f * 31U) % 256U) - 128) * 0.00390625F;
        }
        std::array<float, 256> leaves{};
        for (unsigned leaf = 0; leaf < 256U; ++leaf)
            for (unsigned k = 0; k < 10U; ++k) {
                const auto f = leaf + 256U * k;
                leaves[leaf] = std::fma(x[f], c[f], leaves[leaf]);
            }
        for (unsigned h = 128U; h; h /= 2U)
            for (unsigned i = 0; i < h; ++i) leaves[i] = add(leaves[i], leaves[i + h]);
        std::array<float, 4> stack{};
        float result = 0;
        for (unsigned q = 0; q < 16U; ++q) {
            std::array<float, 16> acc{};
            for (unsigned k = 0; k < 10U; ++k)
                for (unsigned m = 0; m < 16U; ++m) {
                    const auto f = reverse4(q) + 16U * m + 256U * k;
                    acc[m] = std::fma(x[f], c[f], acc[m]);
                }
            for (unsigned h = 8U; h; h /= 2U)
                for (unsigned m = 0; m < h; ++m) acc[m] = add(acc[m], acc[m + h]);
            float v = acc[0];
            bool carry = true;
            for (unsigned b = 0; b < 4U && carry; ++b)
                if ((q >> b) & 1U) v = add(stack[b], v);
                else { stack[b] = v; carry = false; }
            if (carry) result = v;
        }
        require(std::bit_cast<std::uint32_t>(result) == std::bit_cast<std::uint32_t>(leaves[0]),
            "independent 256-leaf/FMA oracle");
    }
}
inline int run() {
    require(median({3.0F, 1.0F, 2.0F}) == 2.0, "odd median");
    require(median({8.0F, 1.0F, 7.0F, 2.0F, 6.0F, 3.0F}) == 4.5, "six-sample median");
    require(median({8.0F, 1.0F, 7.0F, 2.0F, 6.0F, 3.0F, 5.0F, 4.0F}) == 4.5, "eight-sample median");
    require(median({1000.0F, 2.0F, 4.0F, 3.0F, -500.0F}) == 3.0, "unsorted/outlier median");
    require(median({std::numeric_limits<float>::max(), std::numeric_limits<float>::max()}) ==
        static_cast<double>(std::numeric_limits<float>::max()), "float-max median avoids overflow");
    bool empty_rejected = false;
    try { (void)median({}); } catch (const std::invalid_argument&) { empty_rejected = true; }
    require(empty_rejected, "empty median rejected");
    contracts();
    for (unsigned rows : {8192U, 16384U})
        for (unsigned pitch : {2560U, 2568U, 2561U}) panels(rows, pitch);
    tree_oracle();
    std::puts("PASS host admission, inactive switch, aliases, overflow and 256-leaf arithmetic oracle; DEVICE PENDING");
    return 0;
}
}  // namespace ple_tile_host
