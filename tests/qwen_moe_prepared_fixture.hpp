#pragma once
#include "strix/qwen_moe_prepared.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

namespace qwen_moe_prepared_fixture {
namespace qp = strix::qwen_moe_prepared;
inline void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
inline qp::Spec spec(qp::Format format) {
    return {format, format == qp::Format::q5_1 ? qp::Role::down : qp::Role::gate, 2, 32, 768};
}
inline qp::Source source() {
    qp::Source result;
    for (unsigned i = 0; i < 32; ++i) result.model_hash.bytes[i] = static_cast<std::uint8_t>(i + 1);
    result.tensor_index = 17;
    result.file_offset = 37;
    return result;
}
inline std::vector<std::byte> input(std::size_t bytes, unsigned seed = 0) {
    std::vector<std::byte> result(bytes);
    std::uint32_t state = seed + 1;
    for (auto& b : result) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        b = std::byte(state & 255U);
    }
    return result;
}
inline unsigned byte(std::byte b) { return std::to_integer<unsigned>(b); }
// Independent scalar source-format interpretation. No production inverse/map
// helper is used in the coordinate oracle below.
inline unsigned original_code(qp::Format f, std::span<const std::byte> raw, unsigned row, unsigned k) {
    if (f == qp::Format::q5_1) {
        const unsigned block = row * 192 + k / 32 * 24, j = k % 32;
        const unsigned lo = (byte(raw[block + 8 + j % 16]) >> (4 * (j / 16))) & 15U;
        const unsigned hi = (byte(raw[block + 4 + j / 8]) >> (j % 8)) & 1U;
        return lo | (hi << 4);
    }
    const unsigned base = row * (f == qp::Format::q4_k ? 144U : 176U);
    const unsigned lo = (byte(raw[base + (f == qp::Format::q4_k ? 16U : 48U) + k / 64 * 32 + k % 32]) >> (4 * ((k / 32) % 2))) & 15U;
    return lo | (f == qp::Format::q5_k ? ((byte(raw[base + 16 + k % 32]) >> (k / 32)) & 1U) << 4 : 0U);
}
inline unsigned prepared_code(qp::Format f, std::span<const std::byte> packed, unsigned row, unsigned k) {
    const unsigned headers = f == qp::Format::q5_1 ? 512U : 256U;
    const unsigned high = f == qp::Format::q4_k ? 0U : 512U;
    const unsigned group = k / 32, j = k % 32;
    const unsigned lo = (byte(packed[headers + high + (group * 16 + row) * 16 + j / 2]) >> (4 * (j % 2))) & 15U;
    if (f == qp::Format::q4_k) return lo;
    const unsigned hi = f == qp::Format::q5_k ? (byte(packed[headers + row * 32 + j]) >> group) & 1U
        : (byte(packed[headers + (group * 16 + row) * 4 + j / 8]) >> (j % 8)) & 1U;
    return lo | (hi << 4);
}
inline void unit_oracle(qp::Format f, std::span<const std::byte> raw, std::span<const std::byte> packed) {
    const auto layout = qp::describe(spec(f));
    std::vector<std::byte> inverse(layout.unit_bytes);
    qp::inverse_unit(f, packed, inverse);
    require(std::equal(inverse.begin(), inverse.end(), raw.begin(), raw.end()), "inverse bytes differ");
    for (unsigned row = 0; row < 16; ++row) {
        const unsigned count = f == qp::Format::q5_1 ? 8U : 1U;
        const unsigned bytes = f == qp::Format::q5_1 ? 4U : 16U;
        for (unsigned group = 0; group < count; ++group)
            for (unsigned b = 0; b < bytes; ++b)
                require(raw[row * layout.row_bytes + group * 24 + b] == packed[(group * 16 + row) * bytes + b], "header byte differs");
        if (f == qp::Format::q5_k)
            for (unsigned b = 0; b < 32; ++b) require(raw[row * 176 + 16 + b] == packed[256 + row * 32 + b], "Q5_K high plane differs");
        if (f == qp::Format::q5_1)
            for (unsigned group = 0; group < 8; ++group)
                for (unsigned b = 0; b < 4; ++b)
                    require(raw[row * 192 + group * 24 + 4 + b] == packed[512 + (group * 16 + row) * 4 + b], "Q5_1 high plane differs");
    }
    // All coordinates of the current B fragments: 32 hardware lanes, including
    // lane+16 replicas, all four K64 stages/four ascending K16 planes/16 halves.
    // This is a layout oracle, not a certificate for an unimplemented shuffle.
    for (unsigned lane = 0; lane < 32; ++lane)
        for (unsigned stage = 0; stage < 4; ++stage)
            for (unsigned plane = 0; plane < 4; ++plane)
                for (unsigned element = 0; element < 16; ++element) {
                    const unsigned k = stage * 64 + plane * 16 + element;
                    require(original_code(f, raw, lane % 16, k) == prepared_code(f, packed, lane % 16, k), "B fragment code coordinate differs");
                }
}
inline void direct_b_word_oracle(std::span<const std::byte> raw,std::span<const std::byte> packed) {
    for(unsigned lane=0;lane<32;++lane)for(unsigned step=0;step<4;++step)
        for(unsigned plane=0;plane<4;++plane)for(unsigned element=0;element<16;++element) {
            const unsigned i=lane%16,word=(plane%2)*2+element/8;
            const unsigned at=256+((2*step+plane/2)*16+i)*16+word*4+(element%8)/2;
            const unsigned code=(std::to_integer<unsigned>(packed[at])>>(4*(element%2)))&15U;
            const unsigned replica=(lane^16U)%16;
            require(replica==i && code==original_code(qp::Format::q4_k,raw,replica,step*64+plane*16+element),
                "direct B adjacent-nibble / lane exchange fragment differs");
        }
}
inline void tensor_oracle(qp::Spec s, std::span<const std::byte> raw, std::span<const std::byte> packed) {
    const auto layout = qp::describe(s);
    std::vector<std::byte> unit(layout.unit_bytes);
    std::size_t destination = 0;
    for (unsigned expert = 0; expert < s.experts; ++expert)
        for (unsigned output = 0; output < s.outputs; output += 16)
            for (unsigned block = 0; block < s.input / 256; ++block) {
                for (unsigned row = 0; row < 16; ++row) {
                    const auto old_at = (static_cast<std::size_t>(expert) * s.outputs + output + row) * (s.input / 256) * layout.row_bytes + block * layout.row_bytes;
                    std::copy_n(raw.begin() + static_cast<std::ptrdiff_t>(old_at), layout.row_bytes,
                                unit.begin() + static_cast<std::ptrdiff_t>(row * layout.row_bytes));
                }
                unit_oracle(s.format, unit, packed.subspan(destination, layout.unit_bytes));
                destination += layout.unit_bytes;
            }
    require(destination == packed.size(), "tensor unit count differs");
}
} // namespace qwen_moe_prepared_fixture
