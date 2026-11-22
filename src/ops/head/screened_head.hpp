#pragma once

#include "strix/head_argmax.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRIX_SCREEN_HD __host__ __device__
#else
#define STRIX_SCREEN_HD
#endif

namespace strix::ops::screened_head {
namespace ha = ::strix::head_argmax;

inline constexpr std::uint32_t kTileRows = 32;
inline constexpr double kInfinity = std::numeric_limits<double>::infinity();

// R1 BF16 weights, FP32 activations and raw FP32 scores. The incumbent assigns
// eight consecutive columns to each of 32 lanes, then applies five XOR adds.
class Plan {
public:
    [[nodiscard]] static std::optional<Plan> make(std::uint32_t width, std::uint32_t vocab,
                                                  std::uint32_t first = 0) {
        if (width == 0 || width % 8 != 0 || vocab == 0 || vocab > ha::kNone - first ||
            std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) / width < vocab) {
            return std::nullopt;
        }
        const auto path = 1U + 8U * ((width - 1U) / 256U + 1U) + 5U;
        if (path > (1U << 23U)) return std::nullopt;
        return Plan{width, vocab, first, path};
    }

    [[nodiscard]] std::uint32_t width() const { return width_; }
    [[nodiscard]] std::uint32_t vocab() const { return vocab_; }
    [[nodiscard]] std::uint32_t first() const { return first_; }
    [[nodiscard]] std::uint32_t tiles() const { return (vocab_ - 1U) / kTileRows + 1U; }
    [[nodiscard]] std::size_t packed_stride() const { return width_ / 2U; }
    [[nodiscard]] std::size_t packed_bytes() const { return packed_stride() * vocab_; }
    [[nodiscard]] std::size_t weight_count() const { return std::size_t{width_} * vocab_; }
    [[nodiscard]] std::size_t partial_count() const { return tiles(); }

    // Coefficients for 2*m*u and 2*N*eta in the proof below. Both are exact.
    [[nodiscard]] double relative_error() const { return static_cast<double>(path_) * 0x1p-23; }
    [[nodiscard]] double underflow_error() const {
        return static_cast<double>(2ULL * width_ + 31U) * 0x1p-149;
    }

private:
    Plan(std::uint32_t width, std::uint32_t vocab, std::uint32_t first, std::uint32_t path)
        : width_(width), vocab_(vocab), first_(first), path_(path) {}
    std::uint32_t width_;
    std::uint32_t vocab_;
    std::uint32_t first_;
    std::uint32_t path_;
};

struct Result {
    std::uint32_t id;
    float score;
    std::uint32_t runner_up;  // Runner-up ID. The existing kNone sentinel covers missing ranks.
    std::uint32_t nonfinite;
};
static_assert(sizeof(Result) == 16);

STRIX_SCREEN_HD inline Result emit(const ha::Pick2& pick) {
    return {pick.first.index, pick.first.value, pick.index2, pick.first.nonfinite};
}

struct RowCertificate {
    double scale;
    double weight_l1;
    double residual_l1;
};

struct Screen {
    std::vector<std::uint8_t> packed;
    std::vector<RowCertificate> certificates;
};

inline float widen_bf16(std::uint16_t word) {
    return std::bit_cast<float>(std::uint32_t{word} << 16U);
}

STRIX_SCREEN_HD inline int unpack(std::uint8_t pair, std::uint32_t column) {
    return static_cast<int>((pair >> ((column & 1U) * 4U)) & 15U) - 8;
}

// The host reference expands both endpoints by one binary64 step after each
// operation. The CUDA body uses directed binary64 instructions for tighter bounds.
struct HostOutward {
    static double add_up(double a, double b) { return std::nextafter(a + b, kInfinity); }
    static double add_down(double a, double b) { return std::nextafter(a + b, -kInfinity); }
    static double mul_up(double a, double b) { return std::nextafter(a * b, kInfinity); }
    static double mul_down(double a, double b) { return std::nextafter(a * b, -kInfinity); }
};

// This image is coupled to these exact immutable BF16 rows. Rebuild it whenever
// the weights or their order change. A nonfinite row has no certificate.
inline Screen build_screen(const Plan& plan, std::span<const std::uint16_t> weights) {
    if (weights.size() != plan.weight_count()) throw std::invalid_argument("screened head weight size");
    Screen screen{std::vector<std::uint8_t>(plan.packed_bytes()),
                  std::vector<RowCertificate>(plan.vocab())};
    for (std::uint32_t row = 0; row < plan.vocab(); ++row) {
        const auto source = weights.subspan(std::size_t{row} * plan.width(), plan.width());
        double peak = 0;
        for (const auto word : source) {
            const auto w = static_cast<double>(widen_bf16(word));
            if (!std::isfinite(w)) {
                peak = kInfinity;
                break;
            }
            peak = std::max(peak, std::abs(w));
        }
        auto& cert = screen.certificates[row];
        if (!std::isfinite(peak)) {
            cert = {0, kInfinity, kInfinity};
            continue;
        }

        // Use a power of two so every decoded q*scale is exact in binary64.
        int exponent = 0;
        const double fraction = std::frexp(peak / 7.0, &exponent);
        const double scale = peak == 0 ? 1.0 : std::ldexp(1.0, exponent - (fraction == 0.5 ? 1 : 0));
        cert = {scale, 0, 0};
        auto* packed = screen.packed.data() + std::size_t{row} * plan.packed_stride();
        for (std::uint32_t col = 0; col < plan.width(); ++col) {
            const auto w = static_cast<double>(widen_bf16(source[col]));
            const auto q = std::clamp(static_cast<int>(std::round(w / scale)), -7, 7);
            const auto code = static_cast<unsigned>(q + 8);
            packed[col / 2U] |= static_cast<std::uint8_t>(code << ((col & 1U) * 4U));
            const auto decoded = static_cast<double>(q) * scale;
            const auto residual = std::max(HostOutward::add_up(w, -decoded),
                                           HostOutward::add_up(decoded, -w));
            cert.weight_l1 = HostOutward::add_up(cert.weight_l1, std::abs(w));
            cert.residual_l1 = HostOutward::add_up(cert.residual_l1, residual);
        }
    }
    return screen;
}

struct Interval {
    double lower = -kInfinity;
    double upper = kInfinity;
};

// Let A bound sum |w*x|. With unit roundoff u = 2^-24 and underflow
// allowance eta = 2^-150, every FP32 operation obeys |fl(z)-z| <= u|z|+eta
// while staying in range. In the incumbent tree each term crosses at most m
// rounding sites (one product, lane additions, five warp adds), and the tree
// uses at most N operations. Expanding the error recurrence gives
// ((1+u)^m-1)*A + N*eta*(1+u)^m <= 2*m*u*A + 2*N*eta when m*u <= 1/2.
// This also bounds every subtree. If A plus this error fits in FP32, induction
// in the same tree excludes overflow at every site. The L1 residual times
// ||x||inf bounds the difference of the real dots. Add both errors outward to
// the outward screen dot to enclose the incumbent score, including underflow.
template <class Outward>
STRIX_SCREEN_HD Interval certify(const RowCertificate& row, double x_max, double relative,
                                 double underflow, Interval integer_dot) {
    if (x_max == kInfinity || row.weight_l1 == kInfinity) return {};
    const double magnitude = Outward::mul_up(row.weight_l1, x_max);
    const double round = Outward::add_up(Outward::mul_up(relative, magnitude), underflow);
    if (Outward::add_up(magnitude, round) > 0x1.fffffep127) return {};
    const double error = Outward::add_up(Outward::mul_up(row.residual_l1, x_max), round);
    return {Outward::add_down(Outward::mul_down(row.scale, integer_dot.lower), -error),
            Outward::add_up(Outward::mul_up(row.scale, integer_dot.upper), error)};
}

// Two distinct certified rows at or above the cutoff exclude every row whose
// upper bound is strictly below it. Equality must survive for the lower-ID
// tie rule. An uncertified row has +infinity as its upper bound and always
// reaches exact rescoring, so pruning cannot hide nonfinite scores.
STRIX_SCREEN_HD inline double second_lower(const Interval* intervals, std::uint32_t count) {
    double first = -kInfinity;
    double second = -kInfinity;
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto lower = intervals[i].lower;
        if (lower > first) {
            second = first;
            first = lower;
        } else if (lower > second) {
            second = lower;
        }
    }
    return second;
}

STRIX_SCREEN_HD inline bool survives(Interval interval, double cutoff) {
    return interval.upper >= cutoff;
}

}  // namespace strix::ops::screened_head

#undef STRIX_SCREEN_HD
