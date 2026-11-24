// Host invariants of the grouped rotated A4W4 expert class (moe_core/grouped_iu4a4_contract.hpp):
//  1. iu4a4_rot::paired_column is a bijection of [0, 2 rows) and tiles 2 c / 2 c + 1 are gate / up of outputs 16 c ..;
//  2. ExpertSource::value == the BF16 operand of the EXACT tile (PackedOperand::load_tile16 + decode_bf16), NVFP4 and
//     MXFP8, contiguous and interleaved row order;
//  3. a paired image column n == the plain image column paired_column(n) of the same matrix (codes and step).
#include "moe_core/grouped_iu4a4_contract.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
namespace gi = strix::moe_core::grouped_iu4a4;
namespace rot = strix::core::iu4a4_rot;
namespace packed = strix::weight::microscaled;

static_assert(gi::image_code_bytes(128U, 32U) == 2048U);
static_assert(gi::image_step_floats(128U, 32U) == 32U);

int fails = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++fails;
    }
}
std::uint32_t bits(float v) {
    std::uint32_t b;
    std::memcpy(&b, &v, 4);
    return b;
}

void paired_order(unsigned rows) {
    std::vector<unsigned> seen(2U * rows, 0U);
    bool tiles = true;
    for (unsigned n = 0; n < 2U * rows; ++n) {
        const unsigned c = rot::paired_column(n, rows);
        if (c < 2U * rows) ++seen[c];
        const unsigned t = n / 16U, j = n % 16U;
        tiles = tiles && c == (t % 2U) * rows + 16U * (t / 2U) + j;
    }
    bool bijective = true;
    for (const unsigned s : seen) bijective = bijective && s == 1U;
    expect(bijective && tiles, "paired_column order");
}

template <packed::MatrixFormat F>
void decode_and_pair(packed::RowOrder order, unsigned seed) {
    constexpr bool nv = F == packed::MatrixFormat::nvfp4;
    const unsigned rows = 64U, in = 256U, code_row = nv ? in / 2U : in, scale_row = in / (nv ? 16U : 32U);
    std::mt19937 g(seed);
    std::vector<std::uint8_t> codes(std::size_t(rows) * code_row), scales(std::size_t(rows) * scale_row);
    for (auto& c : codes) c = static_cast<std::uint8_t>(g() & 0xffU);
    for (auto& s : scales) s = static_cast<std::uint8_t>(nv ? 0x20U + g() % 32U : 119U + g() % 5U);
    const packed::PackedMatrix m{codes.data(), scales.data(), rows, in, nv ? 0.75F : 1.0F, order};
    const packed::PackedOperand<F> op{m};
    const gi::ExpertSource<F> plain{m, 0U}, paired{m, rows / 2U};
    unsigned bad = 0;
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned k0 = 0; k0 < in; k0 += 16U) {
            std::uint32_t pairs[8];
            op.decode_bf16(op.load_tile16(r, k0), pairs);
            for (unsigned j = 0; j < 16U; ++j) {
                const std::uint32_t want = ((j & 1U) != 0U ? pairs[j / 2U] >> 16U : pairs[j / 2U] & 0xffffU) << 16U;
                const float got = plain.value(r, k0 + j);
                bad += bits(got) != want ? 1U : 0U;
            }
        }
    expect(bad == 0U, nv ? "nvfp4 value == EXACT operand" : "mxfp8 value == EXACT operand");
    // 3: paired column n == plain column paired_column(n).
    std::vector<std::uint8_t> pc(gi::image_code_bytes(in, rows), 0), qc(gi::image_code_bytes(in, rows), 0);
    unsigned bad_pair = 0;
    for (unsigned b = 0; b < in / rot::kBlock; ++b)
        for (unsigned n = 0; n < rows; ++n) {
            const unsigned src = rot::paired_column(n, rows / 2U);
            const float sp = paired.requantise(n, b, rows, pc.data()), sq = plain.requantise(src, b, rows, qc.data());
            bad_pair += bits(sp) != bits(sq);
            for (unsigned i = 0; i < rot::kBlock; ++i)
                bad_pair += rot::pack_code(pc.data(), b * rot::kBlock + i, n, rows) != rot::pack_code(qc.data(), b * rot::kBlock + i, src, rows);
        }
    expect(bad_pair == 0U, "paired image column == plain image column");
}
}  // namespace

int main() {
    paired_order(16U);
    paired_order(64U);
    paired_order(1152U);
    decode_and_pair<packed::MatrixFormat::nvfp4>(packed::RowOrder::contiguous, 1U);
    decode_and_pair<packed::MatrixFormat::nvfp4>(packed::RowOrder::interleaved_halves, 2U);
    decode_and_pair<packed::MatrixFormat::mxfp8>(packed::RowOrder::contiguous, 3U);
    std::printf("%s\n", fails == 0 ? "ALL PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
