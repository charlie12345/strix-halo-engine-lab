#pragma once
// Host encoder of NVFP4 rows (E2M1 codes, one E4M3 scale per 16 columns, one FP32 outer scale per matrix; decode:
// microscaled.hpp nvfp4_value = (e2m1 x scale) x outer) and the MXFP8 -> NVFP4 transcode of a matrix: the narrow pack of a
// microscaled bank (4.5 instead of 8.25 bits per weight). Bit-changing: a packed bank is a FAST or TURBO weight lever, decided by
// gate v2 on the model; the per-matrix error returned here is the load-time evidence (and the per-bank fallback input).
//
// Block scale rule (mse@2): the candidates are the E4M3 rounding of amax_block / (6 outer) and its two neighbours on each
// side (codes clamped to [0, 0x7e]); the candidate with the smallest squared error of the decoded block wins, ties to the
// lower code. Codes: nearest E2M1 of value / (scale outer), ties to even, saturating at +-6.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "strix/weight/formats/microscaled.hpp"
#include "strix/weight/formats/scalar_float.hpp"

namespace strix::weight::microscaled {

// Outer scale that lets the largest block scale (448) times the largest code (6) reach amax.
inline float nvfp4_outer_scale(float amax) {
    return amax > 0.0F && std::isfinite(amax) ? amax / (6.0F * 448.0F) : 1.0F;
}

// Squared error of one 16-column block encoded with scale code `scale`; writes the 8 code bytes when `codes` is set.
inline double nvfp4_block_error(const float* x, float outer, std::uint8_t scale, std::uint8_t* codes) {
    const float step = scalar_float::e4m3(scale) * outer;
    double error = 0.0;
    for (std::size_t i = 0; i < kNvfp4BlockColumns; ++i) {
        const std::uint8_t code = step > 0.0F ? scalar_float::e2m1_encode(x[i] / step) : std::uint8_t{0};
        const double d = double{nvfp4_value(code, scalar_float::e4m3(scale), outer)} - double{x[i]};
        error += d * d;
        if (codes) {
            auto& byte = codes[i / 2U];
            byte = i % 2U == 0U ? code : static_cast<std::uint8_t>(byte | (code << 4U));
        }
    }
    return error;
}

// Encode one row of finite values (size a multiple of 16) with the matrix outer scale.
inline void encode_nvfp4_row(std::span<const float> x, float outer, std::span<std::uint8_t> codes,
                             std::span<std::uint8_t> scales) {
    if (x.empty() || x.size() % kNvfp4BlockColumns != 0U || codes.size() != x.size() / 2U ||
        scales.size() != x.size() / kNvfp4BlockColumns || !(outer > 0.0F) || !std::isfinite(outer))
        throw std::invalid_argument("NVFP4 encode: inconsistent row");
    for (std::size_t b = 0; b < scales.size(); ++b) {
        const float* block = x.data() + b * kNvfp4BlockColumns;
        float amax = 0.0F;
        for (std::size_t i = 0; i < kNvfp4BlockColumns; ++i) {
            if (!std::isfinite(block[i])) throw std::invalid_argument("NVFP4 encode: non-finite value");
            amax = std::fmax(amax, std::fabs(block[i]));
        }
        const int center = scalar_float::e4m3_encode(amax / (6.0F * outer)) & 0x7F;
        std::uint8_t best = 0U;
        double best_error = -1.0;
        for (int s = center - 2; s <= center + 2; ++s) {
            if (s < 0 || s > 0x7E) continue;
            const double error = nvfp4_block_error(block, outer, static_cast<std::uint8_t>(s), nullptr);
            if (best_error < 0.0 || error < best_error) {
                best = static_cast<std::uint8_t>(s);
                best_error = error;
            }
        }
        scales[b] = best;
        nvfp4_block_error(block, outer, best, codes.data() + b * kNvfp4BlockColumns / 2U);
    }
}

// Error of a packed matrix against its source: sum of squared errors, source energy, largest absolute error.
struct PackError {
    double sse{}, energy{};
    float max_abs{};
    double relative() const { return energy > 0.0 ? std::sqrt(sse / energy) : 0.0; }
};

struct Nvfp4Matrix {
    std::vector<std::uint8_t> codes, scales;  // rows x columns / 2, rows x columns / 16, the source row order
    float outer{1.0F};
    PackError error;
};

// MXFP8 matrix (rows x columns E4M3 codes, rows x columns / 32 E8M0 scales) -> NVFP4 with one outer scale (from the
// matrix amax). Throws on malformed input (decode_mxfp8_row).
inline Nvfp4Matrix transcode_mxfp8_to_nvfp4(std::span<const std::uint8_t> codes, std::span<const std::uint8_t> scales,
                                           std::size_t rows, std::size_t columns) {
    if (rows == 0U || columns == 0U || columns % kMxfp8BlockColumns != 0U || codes.size() / columns != rows ||
        codes.size() % columns != 0U || scales.size() != rows * (columns / kMxfp8BlockColumns))
        throw std::invalid_argument("MXFP8 transcode: inconsistent matrix");
    std::vector<float> values(rows * columns);
    float amax = 0.0F;
    for (std::size_t r = 0; r < rows; ++r) {
        std::span<float> row(values.data() + r * columns, columns);
        decode_mxfp8_row(codes.subspan(r * columns, columns),
                         scales.subspan(r * (columns / kMxfp8BlockColumns), columns / kMxfp8BlockColumns), row);
        for (const float v : row) amax = std::fmax(amax, std::fabs(v));
    }
    Nvfp4Matrix out;
    out.outer = nvfp4_outer_scale(amax);
    out.codes.resize(rows * columns / 2U);
    out.scales.resize(rows * (columns / kNvfp4BlockColumns));
    for (std::size_t r = 0; r < rows; ++r) {
        const std::span<const float> row(values.data() + r * columns, columns);
        const std::span<std::uint8_t> row_codes(out.codes.data() + r * columns / 2U, columns / 2U);
        const std::span<std::uint8_t> row_scales(out.scales.data() + r * (columns / kNvfp4BlockColumns),
                                                 columns / kNvfp4BlockColumns);
        encode_nvfp4_row(row, out.outer, row_codes, row_scales);
        for (std::size_t c = 0; c < columns; ++c) {
            const auto byte = row_codes[c / 2U];
            const auto code = static_cast<std::uint8_t>(c % 2U == 0U ? byte & 0x0FU : byte >> 4U);
            const float decoded = nvfp4_value(code, scalar_float::e4m3(row_scales[c / kNvfp4BlockColumns]), out.outer);
            const double d = double{decoded} - double{row[c]};
            out.error.sse += d * d;
            out.error.energy += double{row[c]} * double{row[c]};
            out.error.max_abs = std::fmax(out.error.max_abs, static_cast<float>(std::fabs(d)));
        }
    }
    return out;
}

}  // namespace strix::weight::microscaled
