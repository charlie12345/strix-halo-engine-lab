#pragma once
// Per-family vision adapter: everything model-specific about an image tower. The shared runtime (encoder.hpp) owns
// the graph and the arithmetic; an adapter only names tensors, states its resize rule and patch order, runs its
// projector through DenseOps and describes the span the language model sees.
#include "strix/vision/encoder.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace strix::vision {

struct Image;  // decoded pixels (preprocess library)

// Resize contract of a family, as data: sides are multiples of `multiple`, the area is kept inside
// [min_pixels, max_pixels], the row count after the projector never exceeds max_rows.
struct ResizeRule {
    unsigned multiple = 0;
    std::uint64_t min_pixels = 0, max_pixels = 0;
    unsigned max_rows = 0;
    float max_aspect = 0.0F;  // 0 = unlimited
};

// Tensor lookup of one checkpoint (GGUF, safetensors, pack). Views stay valid while the source lives.
class WeightSource {
public:
    virtual ~WeightSource() = default;
    virtual bool has(std::string_view name) const = 0;
    virtual Tensor tensor(std::string_view name) = 0;  // throws std::out_of_range when absent
};

// What the language model sees for one image: one entry per spliced row.
struct SpanLayout {
    enum class Role : std::uint8_t { image = 0, start = 1, newline = 2, end = 3 };
    std::vector<Role> role;
    std::vector<std::array<std::uint32_t, 3>> lm_pos;  // (t, h, w) relative to the span start; 1D families use [0]
    std::uint32_t real_token_id = 0;                   // id the family's per-row extras treat these rows as
    unsigned grid_h = 0, grid_w = 0;                   // merged grid
    unsigned rows() const { return static_cast<unsigned>(role.size()); }
};

class VisionAdapter {
public:
    virtual ~VisionAdapter() = default;
    virtual EncoderConfig config() const = 0;
    virtual std::string identity() const = 0;  // weights digest + arithmetic id: cache key and capability hash
    virtual ResizeRule resize() const = 0;
    virtual PatchGrid patches(const Image&) const = 0;
    virtual void bind(WeightSource&, EncoderWeights&) = 0;  // tensor-name mapping only
    virtual PositionInputs positions(const EncoderConfig& c, const EncoderWeights& w, const PatchGrid& g) const {
        return default_positions(c, w, g);
    }
    // Merger / unshuffle / pool + MLP and special rows; linears go through `dense`.
    virtual void project(EncoderOutput&, DenseOps& dense, ProjectedRows& out) = 0;
    virtual SpanLayout layout(const PatchGrid&) const = 0;
};

}  // namespace strix::vision
