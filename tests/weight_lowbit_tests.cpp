// Low-bit resident row storage (strix/weight/host_lowbit.hpp): the host encoder, the loader that produces it from BF16
// rows (whole and partial reads), and the device chunk decoder of the dense rows family (core/weight_decode.hpp,
// host build) agree element by element; codes stay in range; the chosen scale never reconstructs worse than the
// unclipped scale; the Q8_0 loader is unchanged by the shared block source.
#include "core/iu4_onfly.hpp"
#include "core/weight_decode.hpp"
#include "strix/weight/host_lowbit.hpp"
#include "strix/weight/host_q8.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <vector>

namespace {

namespace wei = strix::weight;
namespace wd = strix::core::weight_decode;

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

std::uint16_t to_bf16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, 4);
    return static_cast<std::uint16_t>((x + 0x7fffU + ((x >> 16U) & 1U)) >> 16U);
}

std::vector<std::uint16_t> random_matrix(std::size_t n, std::size_t k, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 0.02F);
    std::vector<std::uint16_t> bits(n * k);
    for (std::size_t i = 0; i < bits.size(); ++i) {
        float v = normal(rng);
        if (i % 97U == 5U) v *= 12.0F;  // outliers exercise the clip search
        bits[i] = to_bf16(v);
    }
    // One all-zero group (row 0, group 1) and one single-signed group (row 1, group 0).
    for (std::size_t j = 64; j < 128; ++j) bits[j] = 0;
    for (std::size_t j = 0; j < 64; ++j) bits[k + j] = to_bf16(std::fabs(strix::core::onfly::bf16_bits_to_f32(bits[k + j])));
    return bits;
}

template <class Storage, wd::Format F>
void check_format(const char* name) {
    constexpr std::size_t n = 3, k = 384;
    const auto bits = random_matrix(n, k, 7U + Storage::high_bytes);
    const wei::Shape shape{{n, k, 0, 0}, 2};
    const auto storage = F == wd::Format::q4_g64 ? wei::kQ4G64WeightStorage : wei::kQ6G64WeightStorage;

    wei::MatrixLoadGeometry geometry{};
    check(bool(wei::matrix_load_geometry(storage, shape, geometry)), "geometry");
    check(geometry.row_bytes == k / 128U * Storage::bytes && geometry.bytes == n * geometry.row_bytes, "row bytes");
    check(geometry.row_bytes == wd::row_bytes(F, k), "dense family row bytes");
    wei::MatrixLoadGeometry odd{};
    check(!wei::matrix_load_geometry(storage, {{n, 192, 0, 0}, 2}, odd), "K % 128 refused");

    const auto source = wei::memory_source(std::as_bytes(std::span(bits)));
    const wei::Slice slice{source, {0, bits.size() * 2U}};
    std::vector<std::byte> loaded(geometry.bytes);
    check(bool(wei::load_bf16_matrix(slice, shape, storage, loaded)), "load");

    // Direct encoder == loader; reference decode == device chunk decoder; codes and error bounds.
    std::vector<std::uint8_t> direct(geometry.bytes);
    std::vector<float> w(n * k);
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = strix::core::onfly::bf16_bits_to_f32(bits[i]);
    for (std::size_t b = 0; b < w.size() / 128U; ++b)
        check(wei::lowbit::encode_block<Storage>(w.data() + b * 128U, direct.data() + b * Storage::bytes), "encode");
    check(std::memcmp(direct.data(), loaded.data(), direct.size()) == 0, "loader bytes == encoder bytes");

    for (std::size_t r = 0; r < n; ++r) {
        const auto* row = direct.data() + r * geometry.row_bytes;
        for (unsigned i = 0; i < k; i += 16U) {
            const wd::Chunk<F> chunk(row, i);
            for (unsigned e = 0; e < 16U; ++e) {
                const unsigned col = i + e;
                const auto* block = row + (col / 128U) * Storage::bytes;
                const float ref = wei::lowbit::decode<Storage>(block, col % 128U);
                const float dev = chunk.at(e);
                check(std::memcmp(&ref, &dev, 4) == 0, "chunk decode == reference decode");
                const int q = Storage::code(block, col % 128U);
                check(q >= Storage::qmin && q <= Storage::qmax, "code range");
            }
        }
        for (unsigned g = 0; g < k / 64U; ++g) {
            const auto* block = row + (g / 2U) * Storage::bytes;
            const float s = wei::lowbit::f16_bits_to_f32(Storage::scale_bits(block, g % 2U));
            const float* src = w.data() + r * k + g * 64U;
            float pos = 0, neg = 0, err = 0, err_full = 0;
            for (unsigned j = 0; j < 64; ++j) {
                pos = std::max(pos, src[j]);
                neg = std::max(neg, -src[j]);
            }
            const float base = std::max(pos / float(Storage::qmax), neg / float(-Storage::qmin));
            if (base == 0.0F) {
                check(s == 0.0F, "zero group has zero scale");
                continue;
            }
            const float full = wei::lowbit::f16_bits_to_f32(wei::lowbit::f32_to_f16_bits(base));
            for (unsigned j = 0; j < 64; ++j) {
                const float d = src[j] - wei::lowbit::decode<Storage>(block, (g % 2U) * 64U + j);
                err += d * d;
                const float df = src[j] - full * float(wei::lowbit::quantise<Storage>(src[j], 1.0F / full));
                err_full += df * df;
            }
            check(err <= err_full, "chosen scale no worse than unclipped");
        }
    }

    // Partial reads of the staged source (start and end inside blocks) equal the whole image.
    wei::Slice packed{};
    check(bool(wei::prepare_bf16_matrix_slice(slice, shape, storage, packed)), "prepare");
    check(packed.range.bytes == geometry.bytes, "packed size");
    for (std::uint64_t offset : {0ULL, 1ULL, 67ULL, 99ULL, 150ULL}) {
        const std::uint64_t count = std::min<std::uint64_t>(geometry.bytes - offset, 211ULL);
        std::vector<std::byte> part(count);
        check(bool(packed.source->read({offset, count}, part)), "partial read");
        check(std::memcmp(part.data(), loaded.data() + offset, count) == 0, "partial read bytes");
    }

    // Non-finite input is refused.
    auto bad = bits;
    bad[200] = 0x7f80U;
    const auto bad_source = wei::memory_source(std::as_bytes(std::span(bad)));
    check(!wei::load_bf16_matrix({bad_source, {0, bad.size() * 2U}}, shape, storage, loaded), "nonfinite refused");
    std::cout << name << " ok\n";
}

void check_q8_unchanged() {
    constexpr std::size_t n = 2, k = 96;
    const auto bits = random_matrix(n, k + 32U, 3U);  // reuse the generator, keep the first n * k values
    std::vector<std::uint16_t> m(bits.begin(), bits.begin() + static_cast<std::ptrdiff_t>(n * k));
    const wei::Shape shape{{n, k, 0, 0}, 2};
    const auto source = wei::memory_source(std::as_bytes(std::span(m)));
    const wei::Slice slice{source, {0, m.size() * 2U}};
    const auto expected = strix::core::onfly::bf16_to_q8_0_host(m.data(), n, k);
    std::vector<std::byte> loaded(expected.size());
    check(bool(wei::load_bf16_matrix(slice, shape, wei::kQ8WeightStorage, loaded)), "q8 load");
    check(std::memcmp(loaded.data(), expected.data(), expected.size()) == 0, "q8 bytes unchanged");
    wei::Slice packed{};
    check(bool(wei::prepare_bf16_matrix_slice(slice, shape, wei::kQ8WeightStorage, packed)), "q8 prepare");
    std::vector<std::byte> part(50);
    check(bool(packed.source->read({17, 50}, part)), "q8 partial read");
    check(std::memcmp(part.data(), expected.data() + 17, 50) == 0, "q8 partial bytes unchanged");
}

}  // namespace

int main() {
    check_format<wei::Q4G64Storage, wd::Format::q4_g64>("q4_g64");
    check_format<wei::Q6G64Storage, wd::Format::q6_g64>("q6_g64");
    check_q8_unchanged();
    if (failures) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "weight lowbit tests passed\n";
    return 0;
}
