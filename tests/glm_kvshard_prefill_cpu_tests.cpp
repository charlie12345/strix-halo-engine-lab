#include "glm/prefill/glm_kvshard_prefill.hpp"
#include "glm/engine/glm_kvshard_layout.hpp"
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

// MSVC spelling adapters only; use the production arithmetic on every host.
#if defined(_MSC_VER) && !defined(__clang__)
#define __builtin_bit_cast(T, x) std::bit_cast<T>(x)
#define __builtin_huge_valf() std::numeric_limits<float>::infinity()
#define __builtin_rintf(x) std::nearbyint(x)
#define __builtin_sqrtf(x) std::sqrt(x)
#define __builtin_fmaf(a, b, c) std::fma(a, b, c)
#endif
#include "glm/common/glm_math.hpp"
#if defined(_MSC_VER) && !defined(__clang__)
#undef __builtin_bit_cast
#undef __builtin_huge_valf
#undef __builtin_rintf
#undef __builtin_sqrtf
#undef __builtin_fmaf
#endif

namespace P = strix::glm::pf;
namespace K = strix::glm::kvshard;
using namespace strix::glm;
namespace {
constexpr unsigned stride = kIdxTopk + kKpool - 1;
constexpr P::KvShardPrefillLimits old_limits{8192, 128, P::kKvShardPrefillTemporaryBytes};
static_assert(P::kKvShardPrefillTemporaryBytes == (std::uint64_t{64} << 20));
static_assert(P::kvshard_prefill_bytes(2051, 128, stride) <= P::kKvShardPrefillTemporaryBytes);
unsigned checks = 0, oracle_rows = 0;
void check(bool good) {
    ++checks;
    if (!good) throw std::runtime_error("bounded prefill oracle mismatch");
}
template<class F> void rejects(F f) {
    bool threw = false;
    try { f(); } catch (const std::invalid_argument&) { threw = true; }
    check(threw);
}
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Same R13 host oracle as hip_glm_prefill_tests: ascending score FMA, declared
// exp, tree256 denominator and tree256 o-bar. Project all 256 output dimensions
// with fixed synthetic weights. This checks address invariance, not HIP codegen.
struct Oracle { std::vector<float> probabilities, obar, output; };
Oracle attention(const std::vector<std::uint16_t>& cache, const std::uint32_t* tokens,
                 unsigned count, unsigned seed) {
    Oracle result;
    std::vector<float> qt(kKvLora);
    for (unsigned dim = 0; dim < kKvLora; ++dim)
        qt[dim] = static_cast<float>(static_cast<int>((dim * 7 + seed) % 23) - 11) / 128.0F;
    auto latent = [&](unsigned item, unsigned dim) {
        return bf16_to_float(cache[static_cast<std::size_t>(tokens[item]) * kKvLora + dim]);
    };
    auto& p = result.probabilities;
    p.resize(count);
    for (unsigned item = 0; item < count; ++item)
        p[item] = chain_dot([&](unsigned dim) { return qt[dim]; },
                           [&](unsigned dim) { return latent(item, dim); }, kKvLora) * kMlaScale;
    float maximum = p[0];
    for (unsigned item = 1; item < count; ++item) maximum = p[item] > maximum ? p[item] : maximum;
    for (auto& value : p) value = declared_exp(value - maximum);
    const float sum = tree_sum<256>([&](unsigned item) { return p[item]; }, count);
    for (auto& value : p) value /= sum;
    result.obar.resize(kKvLora);
    for (unsigned dim = 0; dim < kKvLora; ++dim)
        result.obar[dim] = tree_fma<256>([&](unsigned item) { return p[item]; },
                                      [&](unsigned item) { return latent(item, dim); }, count);
    result.output.resize(kMlaDim);
    for (unsigned dim = 0; dim < kMlaDim; ++dim)
        result.output[dim] = chain_dot([&](unsigned i) { return result.obar[i]; }, [&](unsigned i) {
            return static_cast<float>(static_cast<int>((i + 3 * dim) % 19) - 9) / 256.0F;
        }, kKvLora);
    return result;
}

void planning(unsigned rows, unsigned pattern, std::uint64_t capacity, std::uint64_t budget) {
    const auto limits = P::kvshard_prefill_limits(budget, capacity, rows, stride);
    check(limits.bytes <= budget);
    std::vector<std::uint32_t> tokens(static_cast<std::size_t>(rows) * stride, 0xdeadbeefU), counts(rows);
    for (unsigned row = 0; row < rows; ++row) {
        counts[row] = pattern == 2 ? 1 + row % stride : stride;
        for (unsigned item = 0; item < counts[row]; ++item) {
            const auto token = pattern == 0 ? (row * 4 + item) % 8192 : row * stride + item;
            tokens[static_cast<std::size_t>(row) * stride + item] = pattern == 3
                ? static_cast<std::uint32_t>(capacity - 1 - token) : token;
        }
    }
    unsigned windows = 0;
    for (unsigned begin = 0; begin < rows;) {
        const unsigned batch = std::min(limits.rows, rows - begin);
        auto w = P::kvshard_prefill_window(tokens.data() + static_cast<std::size_t>(begin) * stride,
                                         counts.data() + begin, batch, stride, capacity, limits);
        check(w.rows > 0 && w.rows <= batch && w.latents.size() <= limits.latents);
        for (unsigned row = 0; row < w.rows; ++row)
            for (unsigned item = 0; item < counts[begin + row]; ++item) {
                const auto at = static_cast<std::size_t>(row) * stride + item;
                check(w.tokens[at] < w.latents.size());
                check(w.latents[w.tokens[at]] == tokens[static_cast<std::size_t>(begin + row) * stride + item]);
            }
        begin += w.rows;
        ++windows;
    }
    if (pattern == 0 && limits.latents >= 8192) check(windows == (rows + limits.rows - 1) / limits.rows);
    if (pattern == 1) {
        const unsigned per_page = limits.rows / (limits.latents / stride) + (limits.rows % (limits.latents / stride) != 0);
        const unsigned tail = rows % limits.rows;
        const unsigned per_tail = tail / (limits.latents / stride) + (tail % (limits.latents / stride) != 0);
        check(windows == rows / limits.rows * per_page + per_tail);
        if (budget == P::kKvShardPrefillMaximumBytes && rows == 128) check(windows == 1);
    }
    std::printf("plan rows=%u pattern=%u capacity=%llu windows=%u PASS\n", rows, pattern,
                static_cast<unsigned long long>(capacity), windows);
}

void numerical_oracle(const P::KvShardPrefillLimits& limits) {
    constexpr unsigned capacity = 65540, rows = 11;
    const auto storage = static_cast<std::size_t>(K::storage_tokens(capacity));
    std::vector<std::uint16_t> full(static_cast<std::size_t>(capacity) * kKvLora);
    std::array<std::vector<std::uint16_t>, 2> compact;
    for (auto& rank : compact) rank.resize(storage * kKvLora, 0x7fc1U); // poison padding
    for (unsigned token = 0; token < capacity; ++token)
        for (unsigned dim = 0; dim < kKvLora; ++dim) {
            const auto bits = static_cast<std::uint16_t>(((token + dim) % 2) * 0x8000U |
                ((121 + (token * 3 + dim) % 8) << 7) | ((token + 7 * dim) % 128));
            const auto value = dim % 127 == 0 ? static_cast<std::uint16_t>(bits & 0x8000U) : bits;
            full[static_cast<std::size_t>(token) * kKvLora + dim] = value;
            compact[K::owner(token)][static_cast<std::size_t>(K::local_token(token)) * kKvLora + dim] = value;
        }
    std::vector<std::uint32_t> tokens(static_cast<std::size_t>(rows) * stride, 0xdeadbeefU), counts(rows);
    for (unsigned row = 0; row < rows; ++row) {
        counts[row] = row == rows - 1 ? 3U : stride;
        for (unsigned item = 0; item < counts[row]; ++item)
            tokens[static_cast<std::size_t>(row) * stride + item] = row < 4 ? row * 4 + item :
                row == rows - 1 ? capacity - 3 + item : 16380 + (row - 4) * stride + item;
    }
    unsigned windows = 0;
    for (unsigned begin = 0; begin < rows;) {
        auto w = P::kvshard_prefill_window(tokens.data() + static_cast<std::size_t>(begin) * stride,
                                          counts.data() + begin, rows - begin, stride, capacity, limits);
        const auto elements = w.latents.size() * kKvLora;
        std::array<std::vector<std::uint16_t>, 2> packed;
        for (auto& rank : packed) rank.resize(elements, 0);
        std::vector<std::uint16_t> merged(elements, 0x7fc1U);
        // Emulate both existing pack/merge kernels, including zero unowned data.
        for (unsigned rank = 0; rank < 2; ++rank)
            for (std::size_t element = 0; element < elements; ++element) {
                const auto token = w.latents[element / kKvLora];
                if (K::owner(token) == rank)
                    packed[rank][element] = compact[rank][static_cast<std::size_t>(K::local_token(token)) * kKvLora + element % kKvLora];
            }
        for (std::size_t element = 0; element < elements; ++element) {
            const auto token = w.latents[element / kKvLora];
            merged[element] = packed[K::owner(token)][element];
            check(merged[element] == full[static_cast<std::size_t>(token) * kKvLora + element % kKvLora]);
        }
        for (unsigned row = 0; row < w.rows; ++row) {
            const auto reference = attention(full, tokens.data() + static_cast<std::size_t>(begin + row) * stride,
                                             counts[begin + row], begin + row);
            const auto bounded = attention(merged, w.tokens.data() + static_cast<std::size_t>(row) * stride,
                                           counts[begin + row], begin + row);
            check(same(reference.probabilities, bounded.probabilities));
            check(same(reference.obar, bounded.obar));
            check(same(reference.output, bounded.output));
            ++oracle_rows;
        }
        begin += w.rows;
        ++windows;
    }
    if (limits.latents == old_limits.latents) check(windows > 1);
    std::printf("R13 full-image vs bounded union: %u rows, %u windows, probabilities/512 obar/256 output BITWISE\n", oracle_rows, windows);
}
void budget_geometry() {
    // Actual 8K row workspace: upper/lower window counts use disjoint/full-reuse
    // selections. These are planning bounds, not device throughput predictions.
    for (auto capacity : {8192U, 65536U, 1048576U, 2850000U}) {
        for (auto budget : {P::kKvShardPrefillTemporaryBytes, P::kKvShardPrefillMaximumBytes}) {
            const auto lim = P::kvshard_prefill_limits(budget, capacity, 8192, stride);
            check(lim.bytes <= budget && lim.rows >= 128 && lim.latents >= stride);
            check(lim.bytes == P::kvshard_prefill_bytes(lim.latents, lim.rows, stride));
            const unsigned disjoint_rows = lim.latents / stride;
            const unsigned full_pages = 8192 / lim.rows, tail = 8192 % lim.rows;
            const unsigned low = full_pages + (tail != 0);
            const unsigned high = full_pages * ((lim.rows + disjoint_rows - 1) / disjoint_rows) +
                (tail + disjoint_rows - 1) / disjoint_rows;
            std::printf("budget cap=%u MiB=%llu L=%u R=%u peakMiB=%llu windows/8K=%u..%u\n", capacity,
                static_cast<unsigned long long>(budget >> 20), lim.latents, lim.rows,
                static_cast<unsigned long long>(lim.bytes >> 20), low, capacity <= lim.latents ? low : high);
        }
    }
    // Every binary-search boundary and rounded owner stays inside the admitted
    // bytes. A slightly smaller budget cannot retain the same maximal geometry.
    for (unsigned mib = 64; mib <= 2048; mib += 7) {
        const auto budget = std::uint64_t{mib} << 20;
        const auto lim = P::kvshard_prefill_limits(budget, 2850000, 8192, stride);
        check(lim.bytes <= budget);
        check(lim.rows == 8192 || P::kvshard_prefill_bytes(lim.latents, lim.rows + 1, stride) > budget);
        const unsigned target = 128 * stride;
        check(lim.latents >= target || P::kvshard_prefill_bytes(lim.latents + 1, 128, stride) > budget);
        check(lim.latents == std::min<std::uint64_t>(2850000, std::uint64_t{lim.rows} * stride) ||
              P::kvshard_prefill_bytes(lim.latents + 1, lim.rows, stride) > budget);
    }
    rejects([&] { (void)P::kvshard_prefill_limits(1, 2850000, 8192, stride); });
    rejects([&] { (void)P::kvshard_prefill_limits(P::kKvShardPrefillMaximumBytes, 0, 8192, stride); });
    const std::array<std::uint32_t, 2> tokens{0, 1};
    const std::uint32_t count = 2;
    rejects([&] { (void)P::kvshard_prefill_window(tokens.data(), &count, 1, 2, 2, {1, 1, 0}); });
    const auto small = P::kvshard_prefill_limits(P::kKvShardPrefillTemporaryBytes, 2, 1, stride);
    check(P::kvshard_prefill_window(tokens.data(), &count, 1, 2, 2, small).rows == 1);
    // Forecast only: independent uniformly distributed row selections. Actual
    // selected IDs determine the count; this is deliberately no timing gate.
    for (auto length : {8192U, 65536U, 1048576U}) {
        for (auto budget : {P::kKvShardPrefillTemporaryBytes, P::kKvShardPrefillMaximumBytes}) {
            unsigned reuse = 0, uniform = 0, disjoint = 0;
            for (unsigned context = 8192; context <= length; context += 8192) {
                const auto lim = P::kvshard_prefill_limits(budget, context, 8192, stride);
                unsigned mean_rows = lim.rows, worst_rows = lim.rows;
                if (lim.latents < context) {
                    const double r = std::log1p(-static_cast<double>(lim.latents) / context) /
                                     std::log1p(-static_cast<double>(stride) / context);
                    mean_rows = std::max(1U, std::min(lim.rows, static_cast<unsigned>(r)));
                    worst_rows = std::min(lim.rows, lim.latents / stride);
                }
                for (unsigned begin = 0; begin < 8192; begin += lim.rows) {
                    const unsigned page_rows = std::min(lim.rows, 8192 - begin);
                    ++reuse;
                    uniform += (page_rows + mean_rows - 1) / mean_rows;
                    disjoint += (page_rows + worst_rows - 1) / worst_rows;
                }
            }
            std::printf("forecast tokens=%u MiB=%llu reuse=%u independent=%u disjoint-bound=%u windows/layer\n",
                length, static_cast<unsigned long long>(budget >> 20), reuse, uniform, disjoint);
        }
    }
}
} // namespace

int main() {
    try {
        for (auto budget : {P::kKvShardPrefillTemporaryBytes, P::kKvShardPrefillMaximumBytes}) {
            planning(257, 0, 8192, budget); planning(128, 1, 655360, budget);
            planning(129, 2, 655360, budget); planning(128, 3, 2850000, budget);
        }
        std::uint32_t token = 0, count = 1;
        rejects([&] { (void)P::kvshard_prefill_window(&token, &count, 0, stride, 8192, old_limits); });
        count = stride + 1;
        rejects([&] { (void)P::kvshard_prefill_window(&token, &count, 1, stride, 8192, old_limits); });
        count = 0;
        rejects([&] { (void)P::kvshard_prefill_window(&token, &count, 1, stride, 8192, old_limits); });
        count = 1; token = 8192;
        rejects([&] { (void)P::kvshard_prefill_window(&token, &count, 1, stride, 8192, old_limits); });
        numerical_oracle(old_limits); // previous address plan
        numerical_oracle(P::kvshard_prefill_limits(P::kKvShardPrefillTemporaryBytes, 65540, 11, stride));
        numerical_oracle(P::kvshard_prefill_limits(P::kKvShardPrefillMaximumBytes, 65540, 11, stride));
        budget_geometry();
        std::printf("PASS: %u checks; device/model-logit gates NOT RUN\n", checks);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
