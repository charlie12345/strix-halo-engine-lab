#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace strix::core {

// Resident BF16 weights [output][input] and tight FP32 output [row][output].
struct Bf16RowProjection {
    const std::uint16_t* weights{};
    unsigned outputs{};
    float* output{};
    bool round_bf16{};  // Round the completed FP32 sum and store it widened to FP32.
    std::size_t row_bytes{};  // Zero uses input_width * sizeof(std::uint16_t).

    constexpr bool contiguous(unsigned input_width) const {
        return row_bytes == 0U ||
            (row_bytes % sizeof(std::uint16_t) == 0U && row_bytes / sizeof(std::uint16_t) == input_width);
    }
};

// Value class of a dense activation operand, declared by the caller that produced it.
// bf16_exact: every element is a finite FP32 value exactly representable in BF16 (the
// producer stored RN_bf16 of its result, or widened a BF16 tensor).
enum class Bf16Operand : unsigned char { fp32, bf16_exact };

// Activation planes the exact BF16 split needs. The split writes hi = RN_bf16(x),
// mid = RN_bf16(x - hi), lo = RN_bf16(x - hi - mid), and each K16 step of the tall
// chain issues lo, mid, hi into one accumulator that starts at +0. For a BF16-exact
// finite x, mid and lo are +0, so the two leading MMAs add only zero products to an
// accumulator that is never -0: one plane (hi) gives the same bits. This is an EXACT
// schedule of the three-term arithmetic, not a FAST operand class.
constexpr int exact_bf16_planes(Bf16Operand operand) {
    return operand == Bf16Operand::bf16_exact ? 1 : 3;
}

template <auto Limit, class Size, class Status, class Launch>
Status for_each_dense_chunk(Size total, Status success, const Launch& launch) {
    static_assert(Limit > 0);
    constexpr Size limit = Limit;
    for (Size first = 0; first < total;) {
        const Size count = std::min(total - first, limit);
        if (const auto status = launch(first, count); status != success) return status;
        first += count;
    }
    return success;
}

template <std::size_t Limit, class Status, class Launch>
Status for_each_bf16_projection_group(std::span<const Bf16RowProjection> projections, Status success,
                                      const Launch& launch) {
    return for_each_dense_chunk<Limit>(projections.size(), success, [&](std::size_t first, std::size_t count) {
        return launch(projections.subspan(first, count));
    });
}

// Route of one projection of a call whose FP32 rows may be stored once as a BF16 operand image. Every route stores the
// bits the FP32 route stores (the image holds the operand bits the FP32 staging makes of the same values).
enum class ImageRoute : unsigned char { fp32_rows, packed, image };

// Backend tile data of the image routes (not model data): the height from which the packed schedule is faster, its K
// stage and tile width, and the packed columns one call needs to pay for its image pass (the image costs rows x K x 6
// bytes; the packed schedule saves a share of each whole tile's FP32-route time; both scale with rows x K).
struct ImageRoutePolicy {
    unsigned min_rows{};
    unsigned k_step{};
    unsigned tile_columns{};
    unsigned min_packed_columns{};
};

// Routes of the projections of one call on FP32 rows [rows][input_width] (routes.size() >= projections.size()):
// packed = whole tiles within max_outputs, 16-byte aligned weights, a row pitch of whole 16-byte units no shorter than the row;
// image = what image_admits(projection) accepts; fp32_rows otherwise. Returns false, every route fp32_rows, when the
// call keeps its FP32 rows: below min_rows, K not a multiple of the stage, or fewer packed columns than min_packed_columns.
template <class ImageAdmits>
bool plan_image_routes(const ImageRoutePolicy& policy, unsigned input_width, unsigned rows,
                       std::span<const Bf16RowProjection> projections, unsigned max_outputs,
                       std::span<ImageRoute> routes, const ImageAdmits& image_admits) {
    if (routes.size() < projections.size()) return false;
    std::fill_n(routes.begin(), projections.size(), ImageRoute::fp32_rows);
    if (policy.k_step == 0U || policy.tile_columns == 0U || rows == 0U || rows < policy.min_rows || input_width == 0U ||
        input_width % policy.k_step != 0U)
        return false;
    const auto natural = std::size_t{input_width} * sizeof(std::uint16_t);
    std::size_t packed_columns = 0;
    for (std::size_t p = 0; p < projections.size(); ++p) {
        const auto& projection = projections[p];
        const auto pitch = projection.row_bytes != 0U ? projection.row_bytes : natural;
        if (projection.weights && projection.output && projection.outputs != 0U && projection.outputs <= max_outputs &&
            projection.outputs <= std::numeric_limits<std::size_t>::max() / sizeof(float) / rows &&
            projection.outputs % policy.tile_columns == 0U && pitch >= natural && pitch % 16U == 0U &&
            reinterpret_cast<std::uintptr_t>(projection.weights) % 16U == 0U) {
            routes[p] = ImageRoute::packed;
            packed_columns += projection.outputs;
        } else if (image_admits(projection)) {
            routes[p] = ImageRoute::image;
        }
    }
    if (packed_columns >= policy.min_packed_columns) return true;
    std::fill_n(routes.begin(), projections.size(), ImageRoute::fp32_rows);
    return false;
}

// The launches of a routed call: for each route (packed, image, fp32_rows) the projections of that route in call order,
// in runs of at most Limit that share the rounding class and the row pitch; launch(route, run).
template <std::size_t Limit, class Status, class Launch>
Status for_each_image_route_run(std::span<const Bf16RowProjection> projections, std::span<const ImageRoute> routes,
                                unsigned input_width, Status success, const Launch& launch) {
    static_assert(Limit > 0);
    const auto natural = std::size_t{input_width} * sizeof(std::uint16_t);
    const auto pitch = [natural](const Bf16RowProjection& p) { return p.row_bytes != 0U ? p.row_bytes : natural; };
    for (const auto route : {ImageRoute::packed, ImageRoute::image, ImageRoute::fp32_rows}) {
        Bf16RowProjection run[Limit]{};
        std::size_t count = 0;
        const auto flush = [&]() {
            const auto status = count == 0U ? success : launch(route, std::span<const Bf16RowProjection>(run, count));
            count = 0;
            return status;
        };
        for (std::size_t p = 0; p < projections.size(); ++p) {
            if (routes[p] != route) continue;
            if (count != 0U && (count == Limit || run[0].round_bf16 != projections[p].round_bf16 ||
                                pitch(run[0]) != pitch(projections[p])))
                if (const auto status = flush(); status != success) return status;
            run[count++] = projections[p];
        }
        if (const auto status = flush(); status != success) return status;
    }
    return success;
}

}  // namespace strix::core
