#include "inkling2/inkling2_vision.hpp"

#include "strix/serve/json.hpp"
#include "strix/vision/preprocess.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace strix::inkling2 {
namespace {

[[noreturn]] void bad(const std::string& what) { throw std::invalid_argument("Inkling vision: " + what); }

std::string bits(float v) {
    std::ostringstream s;
    s << std::hex << std::bit_cast<std::uint32_t>(v);
    return s.str();
}

unsigned cells(unsigned pixels, unsigned cell) { return (pixels + cell - 1U) / cell; }

// transformers vision_utils._interpolation_axis_taps_weights, bilinear, align_corners=False, border: the two taps of
// target index i on an axis of n patches resampled from a table side of `side`, and their weights.
struct AxisTaps {
    unsigned tap[2];
    float weight[2];
};
AxisTaps axis_taps(unsigned i, unsigned n, unsigned side) {
    const float src = (static_cast<float>(i) + 0.5F) * static_cast<float>(side) / static_cast<float>(n) - 0.5F;
    const float floor = std::floor(src);
    AxisTaps t{};
    for (unsigned k = 0; k < 2U; ++k) {
        const long raw = static_cast<long>(floor) + static_cast<long>(k);
        t.tap[k] = static_cast<unsigned>(std::clamp(raw, 0L, static_cast<long>(side) - 1L));
        const float distance = std::fabs(src - floor - static_cast<float>(k));
        t.weight[k] = std::max(1.0F - distance, 0.0F);
    }
    return t;
}

// rows x cols FP32 copy of a tensor view (any element type), scaled per output row by `scale` (nullptr: none).
std::vector<float> scaled_rows(const vision::Tensor& t, const vision::Tensor* scale) {
    std::vector<float> out(t.size());
    for (std::size_t r = 0; r < t.rows; ++r) {
        const float s = scale ? scale->at(r) : 1.0F;
        for (std::size_t c = 0; c < t.cols; ++c) out[r * t.cols + c] = t.at(r * t.cols + c) * s;
    }
    return out;
}

vision::Tensor owned(vision::EncoderWeights& w, std::vector<float> data, std::size_t rows, std::size_t cols) {
    if (data.size() != rows * cols) bad("internal tensor size");
    auto& stored = w.owned.emplace_back(std::move(data));
    return vision::Tensor{stored.data(), vision::Tensor::Type::f32, rows, cols};
}

vision::Tensor need(vision::WeightSource& source, const std::string& name, std::size_t rows, std::size_t cols) {
    if (!source.has(name)) bad("missing tensor " + name);
    auto t = source.tensor(name);
    if (t.rows * t.cols != rows * cols) bad("tensor " + name + " has an unexpected shape");
    t.rows = rows;
    t.cols = cols;
    return t;
}

}  // namespace

VisionSpec parse_vision_spec(const std::string& text) {
    const auto root = serve::parse_json(text);
    const auto* vc = root.find("vision_config");
    if (!vc || !vc->is_object()) bad("config.json has no vision_config");
    const auto num = [&](const char* key, double fallback) {
        const auto* v = vc->find(key);
        if (!v) return fallback;
        if (!v->is_number()) bad(std::string("vision_config.") + key + " is not a number");
        return v->as_double();
    };
    VisionSpec spec;
    const VisionSpec want;
    const auto check = [&](const char* key, double value, double expected) {
        if (value != expected) bad(std::string("unsupported vision_config.") + key);
    };
    const auto* type = vc->find("model_type");
    if (!type || !type->is_string() || type->as_string() != "inkling2_vision") bad("vision_config.model_type is not inkling2_vision");
    check("hidden_size", num("hidden_size", 1024), want.width);
    check("num_hidden_layers", num("num_hidden_layers", 23), want.layers);
    check("num_attention_heads", num("num_attention_heads", 16), want.heads);
    check("mlp_ratio", num("mlp_ratio", 4.0), 4.0);
    check("native_patch_size", num("native_patch_size", num("patch_size", 20)), want.patch);
    check("adapter_merge_size", num("adapter_merge_size", num("spatial_merge_size", 2)), want.merge);
    check("posemb_grid_size", num("posemb_grid_size", 32), want.pos_grid);
    check("projection_hidden_dim", num("projection_hidden_dim", 4096), want.projection);
    check("text_hidden_size", num("text_hidden_size", 2304), want.text_width);
    check("temporal_patch_size", num("temporal_patch_size", 2), 2);  // video only; images use one frame
    if (const auto* eps = vc->find("layer_norm_eps")) spec.eps = static_cast<float>(eps->as_double());
    if (spec.eps != want.eps) bad("unsupported vision_config.layer_norm_eps");
    if (const auto* pb = vc->find("projector_bias"); pb && (!pb->is_bool() || pb->as_bool())) bad("projector biases are not supported");
    if (const auto* rp = vc->find("rope_parameters"); rp && rp->is_object()) {
        if (const auto* th = rp->find("rope_theta")) spec.rope_theta = static_cast<float>(th->as_double());
    }
    if (spec.rope_theta != want.rope_theta) bad("unsupported vision rope_theta");
    return spec;
}

std::vector<weight::TensorSpec> vision_tensor_specs(const VisionSpec& s) {
    const std::uint64_t D = s.width, F = s.mlp_width;
    std::vector<weight::TensorSpec> v;
    const auto add = [&](std::string name, std::vector<std::uint64_t> extents) {
        v.push_back({std::move(name), "BF16", std::move(extents), true, {}});
    };
    const std::string e = "model.visual.backbone.embeddings.";
    add(e + "class_embedding", {D});
    add(e + "cls_position_embedding", {D});
    add(e + "patch_embedding.weight", {D, 3ULL * s.patch * s.patch});
    add(e + "position_embedding", {std::uint64_t{s.pos_grid} * s.pos_grid, D});
    add("model.visual.backbone.ln_pre.weight", {D});
    add("model.visual.backbone.ln_pre.bias", {D});
    for (unsigned l = 0; l < s.layers; ++l) {
        const std::string p = "model.visual.backbone.layers." + std::to_string(l) + ".";
        for (const char* n : {"ln_1", "ln_2"}) {
            add(p + n + ".weight", {D});
            add(p + n + ".bias", {D});
        }
        add(p + "ls_1.gamma", {D});
        add(p + "ls_2.gamma", {D});
        add(p + "mlp.c_fc.weight", {F, D});
        add(p + "mlp.c_fc.bias", {F});
        add(p + "mlp.c_proj.weight", {D, F});
        add(p + "mlp.c_proj.bias", {D});
        for (const char* n : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
            add(p + "self_attn." + n + ".weight", {D, D});
            add(p + "self_attn." + n + ".bias", {D});
        }
    }
    add("model.visual.proj_in.weight", {s.projection, 4U * D});
    add("model.visual.proj_out.weight", {s.text_width, s.projection});
    return v;
}

unsigned vision_tokens(unsigned width, unsigned height, unsigned cell) {
    if (width == 0U || height == 0U || cell == 0U) bad("empty image or cell");
    return cells(width, cell) * cells(height, cell);
}

VisionSize vision_fit(unsigned width, unsigned height, unsigned max_tokens, unsigned cell) {
    if (max_tokens == 0U || cell == 0U) bad("zero token limit or cell");
    if (std::uint64_t{cells(width, cell)} * cells(height, cell) <= max_tokens) return {width, height};
    // Largest scale that keeps the padded grid within the limit (aspect kept; each side at least one pixel).
    double lo = 0.0, hi = 1.0;
    for (int it = 0; it < 64; ++it) {
        const double mid = 0.5 * (lo + hi);
        const auto w = std::max(1U, static_cast<unsigned>(std::floor(width * mid)));
        const auto h = std::max(1U, static_cast<unsigned>(std::floor(height * mid)));
        (std::uint64_t{cells(w, cell)} * cells(h, cell) <= max_tokens ? lo : hi) = mid;
    }
    return {std::max(1U, static_cast<unsigned>(std::floor(width * lo))),
            std::max(1U, static_cast<unsigned>(std::floor(height * lo)))};
}

VisionAdapter::VisionAdapter(VisionSpec spec, std::string weights_digest, unsigned max_tokens)
    : spec_(spec), digest_(std::move(weights_digest)), max_tokens_(max_tokens) {
    if (max_tokens_ == 0U || max_tokens_ > 16384U) bad("token limit outside 1..16384");
    if (spec_.patch == 0U || spec_.merge == 0U || spec_.width == 0U || spec_.heads == 0U || spec_.width % spec_.heads)
        bad("invalid tower geometry");
    if (digest_.empty()) bad("the weights digest is part of the cache identity and must be stated");
}

vision::EncoderConfig VisionAdapter::config() const {
    vision::EncoderConfig c;
    c.layers = spec_.layers;
    c.width = spec_.width;
    c.heads = spec_.heads;
    c.mlp_width = spec_.mlp_width;
    c.norm = vision::EncoderConfig::Norm::layernorm;
    c.eps = spec_.eps;
    c.mlp = vision::EncoderConfig::Mlp::gelu;
    c.qkv_bias = c.out_bias = c.mlp_bias = true;
    c.patch_bias = false;
    c.pos = vision::EncoderConfig::Pos::rope2d_plus_learned;
    c.rope_theta = spec_.rope_theta;
    c.rope_pairing = vision::EncoderConfig::RopePairing::adjacent;
    c.pos_grid = spec_.pos_grid;
    c.patch = spec_.patch;
    c.temporal = 1U;
    c.channels = 3U;
    c.final_norm = false;
    c.embed_norm = true;
    return c;
}

std::string VisionAdapter::identity() const {
    return "inkling2-vision-v1|" + digest_ + "|cell=" + std::to_string(spec_.patch * spec_.merge) +
           "|max_tokens=" + std::to_string(max_tokens_) + "|eps=" + bits(spec_.eps) + "|theta=" + bits(spec_.rope_theta) +
           "|ls=folded-f32|pad=-1|alpha=checker8";
}

vision::ResizeRule VisionAdapter::resize() const {
    vision::ResizeRule r;
    const std::uint64_t cell = std::uint64_t{spec_.patch} * spec_.merge;
    r.multiple = static_cast<unsigned>(cell);
    r.min_pixels = 1U;
    r.max_pixels = std::uint64_t{max_tokens_} * cell * cell;
    r.max_rows = max_tokens_;
    return r;
}

VisionAdapter::TokenGrid VisionAdapter::token_grid(const vision::PatchGrid& g) const {
    if (g.grid_h == 0U || g.grid_w == 0U || g.grid_h % spec_.merge || g.grid_w % spec_.merge)
        bad("patch grid is not a whole number of cells");
    if (std::uint64_t{g.grid_h} * g.grid_w + 1U != g.rows) bad("patch grid rows differ from 1 + grid_h * grid_w");
    const TokenGrid t{g.grid_h / spec_.merge, g.grid_w / spec_.merge};
    if (std::uint64_t{t.rows} * t.cols > max_tokens_) bad("image over the token limit");
    return t;
}

vision::PatchGrid VisionAdapter::patches(const vision::Image& input) const {
    if (input.width <= 0 || input.height <= 0 || input.rgb.size() != input.pixels() * 3U) bad("invalid image");
    const unsigned cell = spec_.patch * spec_.merge;
    const auto fit = vision_fit(static_cast<unsigned>(input.width), static_cast<unsigned>(input.height), max_tokens_, cell);
    vision::Image scaled;
    const bool resized = fit.width != static_cast<unsigned>(input.width) || fit.height != static_cast<unsigned>(input.height);
    if (resized) {
        scaled = vision::resize(input, static_cast<int>(fit.width), static_cast<int>(fit.height), vision::Resampler{});
    }
    const vision::Image& img = resized ? scaled : input;
    const unsigned W = static_cast<unsigned>(img.width), H = static_cast<unsigned>(img.height);
    const unsigned rows_t = cells(H, cell), cols_t = cells(W, cell), m = spec_.merge, P = spec_.patch;
    vision::check_grid_axes(std::uint64_t{rows_t} * m, std::uint64_t{cols_t} * m);
    vision::PatchGrid g;
    g.grid_h = rows_t * m;
    g.grid_w = cols_t * m;
    g.patch_dim = 3U * P * P;
    g.rows = 1U + g.grid_h * g.grid_w;
    g.data.assign(std::size_t{g.rows} * g.patch_dim, 0.0F);
    g.y.assign(g.rows, 0U);
    g.x.assign(g.rows, 0U);
    // Inkling2ImageProcessor: pad with -1 (0..255 scale), then (v - 127.5) / 127.5 in FP32 (fused rescale+normalise).
    const float pad = (-1.0F - 127.5F) / 127.5F;
    std::size_t row = 1;
    for (unsigned ty = 0; ty < rows_t; ++ty)
        for (unsigned tx = 0; tx < cols_t; ++tx)
            for (unsigned dy = 0; dy < m; ++dy)
                for (unsigned dx = 0; dx < m; ++dx, ++row) {
                    const unsigned py0 = (ty * m + dy) * P, px0 = (tx * m + dx) * P;
                    float* out = g.data.data() + row * g.patch_dim;
                    for (unsigned py = 0; py < P; ++py)
                        for (unsigned px = 0; px < P; ++px)
                            for (unsigned c = 0; c < 3U; ++c) {
                                const unsigned y = py0 + py, x = px0 + px;
                                float v = pad;
                                if (y < H && x < W)
                                    v = (static_cast<float>(img.rgb[(std::size_t{y} * W + x) * 3U + c]) - 127.5F) / 127.5F;
                                out[(py * P + px) * 3U + c] = v;
                            }
                    g.y[row] = static_cast<std::uint16_t>(ty * m + dy);
                    g.x[row] = static_cast<std::uint16_t>(tx * m + dx);
                }
    return g;
}

void VisionAdapter::bind(vision::WeightSource& source, vision::EncoderWeights& w) {
    const auto& s = spec_;
    const std::string root = "model.visual.backbone.";
    const std::size_t D = s.width, F = s.mlp_width;
    w.patch_w = need(source, root + "embeddings.patch_embedding.weight", D, 3U * s.patch * s.patch);
    w.pos_embed = need(source, root + "embeddings.position_embedding", std::size_t{s.pos_grid} * s.pos_grid, D);
    pos_embed_ = w.pos_embed;
    const auto cls = need(source, root + "embeddings.class_embedding", D, 1);
    const auto cls_pos = need(source, root + "embeddings.cls_position_embedding", D, 1);
    class_row_.resize(D);
    for (std::size_t i = 0; i < D; ++i) class_row_[i] = cls.at(i) + cls_pos.at(i);
    w.embed_norm_w = need(source, root + "ln_pre.weight", D, 1);
    w.embed_norm_b = need(source, root + "ln_pre.bias", D, 1);
    w.blocks.resize(s.layers);
    for (unsigned l = 0; l < s.layers; ++l) {
        const std::string p = root + "layers." + std::to_string(l) + ".";
        auto& b = w.blocks[l];
        b.norm1_w = need(source, p + "ln_1.weight", D, 1);
        b.norm1_b = need(source, p + "ln_1.bias", D, 1);
        b.norm2_w = need(source, p + "ln_2.weight", D, 1);
        b.norm2_b = need(source, p + "ln_2.bias", D, 1);
        std::vector<float> qkv(3U * D * D), qkv_b(3U * D);
        const char* names[3] = {"q_proj", "k_proj", "v_proj"};
        for (unsigned j = 0; j < 3U; ++j) {
            const auto wt = need(source, p + "self_attn." + names[j] + ".weight", D, D);
            const auto bt = need(source, p + "self_attn." + names[j] + ".bias", D, 1);
            for (std::size_t i = 0; i < D * D; ++i) qkv[j * D * D + i] = wt.at(i);
            for (std::size_t i = 0; i < D; ++i) qkv_b[j * D + i] = bt.at(i);
        }
        b.qkv_w = owned(w, std::move(qkv), 3U * D, D);
        b.qkv_b = owned(w, std::move(qkv_b), 3U * D, 1);
        // residual + lambda * (W x + bias) == residual + (lambda W) x + lambda bias
        const auto ls1 = need(source, p + "ls_1.gamma", D, 1);
        const auto ls2 = need(source, p + "ls_2.gamma", D, 1);
        const auto out_w = need(source, p + "self_attn.out_proj.weight", D, D);
        const auto out_b = need(source, p + "self_attn.out_proj.bias", D, 1);
        b.out_w = owned(w, scaled_rows(out_w, &ls1), D, D);
        b.out_b = owned(w, scaled_rows(out_b, &ls1), D, 1);
        b.mlp1_w = need(source, p + "mlp.c_fc.weight", F, D);
        b.mlp1_b = need(source, p + "mlp.c_fc.bias", F, 1);
        const auto fc2 = need(source, p + "mlp.c_proj.weight", D, F);
        const auto fc2_b = need(source, p + "mlp.c_proj.bias", D, 1);
        b.mlp2_w = owned(w, scaled_rows(fc2, &ls2), D, F);
        b.mlp2_b = owned(w, scaled_rows(fc2_b, &ls2), D, 1);
    }
    proj_in_ = need(source, "model.visual.proj_in.weight", s.projection, 4U * D);
    proj_out_ = need(source, "model.visual.proj_out.weight", s.text_width, s.projection);
    vision::validate(config(), w);
}

vision::PositionInputs VisionAdapter::positions(const vision::EncoderConfig& c, const vision::EncoderWeights& w,
                                                const vision::PatchGrid& g) const {
    (void)token_grid(g);
    if (class_row_.size() != c.width || !w.pos_embed) bad("positions before bind");
    vision::PositionInputs p;
    const unsigned D = c.width, G = c.pos_grid;
    p.addend.resize(std::size_t{g.rows} * D);
    std::copy(class_row_.begin(), class_row_.end(), p.addend.begin());
    for (unsigned r = 1; r < g.rows; ++r) {
        const auto ty = axis_taps(g.y[r], g.grid_h, G), tx = axis_taps(g.x[r], g.grid_w, G);
        std::size_t idx[4];
        float wt[4];
        for (unsigned a = 0; a < 2U; ++a)
            for (unsigned b = 0; b < 2U; ++b) {
                idx[a * 2U + b] = std::size_t{ty.tap[a]} * G + tx.tap[b];
                wt[a * 2U + b] = ty.weight[a] * tx.weight[b];
            }
        float* out = p.addend.data() + std::size_t{r} * D;
        for (unsigned k = 0; k < D; ++k) {
            float acc = 0.0F;
            for (unsigned j = 0; j < 4U; ++j) acc += w.pos_embed.at(idx[j] * D + k) * wt[j];
            out[k] = acc;
        }
    }
    // Inkling2VisionRotaryEmbedding: inv_freq over head_dim / 2 = 32 channels (16 frequencies), width angles first.
    const unsigned half = c.head_dim() / 2U, nf = half / 2U;
    std::vector<float> inv(nf);
    for (unsigned f = 0; f < nf; ++f)
        inv[f] = 1.0F / std::pow(c.rope_theta, static_cast<float>(2U * f) / static_cast<float>(half));
    p.angles.assign(std::size_t{g.rows} * half, 0.0F);  // the class row sits at (0, 0)
    for (unsigned r = 1; r < g.rows; ++r)
        for (unsigned f = 0; f < nf; ++f) {
            p.angles[std::size_t{r} * half + f] = static_cast<float>(g.x[r] + 1U) * inv[f];
            p.angles[std::size_t{r} * half + nf + f] = static_cast<float>(g.y[r] + 1U) * inv[f];
        }
    return p;
}

void VisionAdapter::project(vision::EncoderOutput& hidden, const vision::PatchGrid& g, vision::DenseOps& dense,
                            vision::ProjectedRows& out) {
    const auto t = token_grid(g);
    const unsigned D = spec_.width, tokens = t.rows * t.cols;
    if (hidden.width != D || hidden.rows != g.rows || hidden.hidden.size() != std::size_t{g.rows} * D)
        bad("encoder output does not match the patch grid");
    if (!proj_in_ || !proj_out_) bad("project before bind");
    // Drop the class row; the four patch rows of a cell are consecutive, so the merged input is the contiguous rest.
    const float* merged = hidden.hidden.data() + D;
    std::vector<float> mid(std::size_t{tokens} * spec_.projection);
    dense.linear(proj_in_, {}, merged, tokens, mid.data());
    for (auto& v : mid) v = vision::activate(vision::Activation::gelu, v);
    std::vector<float> rows(std::size_t{tokens} * spec_.text_width);
    dense.linear(proj_out_, {}, mid.data(), tokens, rows.data());
    for (const float v : rows)
        if (!std::isfinite(v)) throw std::runtime_error("Inkling vision: non-finite image rows");
    out.rows = tokens;
    out.width = spec_.text_width;
    out.bf16 = vision::to_bf16_rows(rows.data(), rows.size());
    out.deep.clear();
}

vision::SpanLayout VisionAdapter::layout(const vision::PatchGrid& g) const {
    const auto t = token_grid(g);
    vision::SpanLayout s;
    const unsigned n = t.rows * t.cols;
    s.role.assign(n, vision::SpanLayout::Role::image);
    s.lm_pos.resize(n);
    for (unsigned i = 0; i < n; ++i) s.lm_pos[i] = {i, 0U, 0U};
    s.real_token_id = kVisionImageTokenId;
    s.grid_h = t.rows;
    s.grid_w = t.cols;
    return s;
}

std::vector<std::pair<vision::Tensor, vision::Tensor>> VisionAdapter::projector_linears() const {
    if (!proj_in_ || !proj_out_) return {};
    return {{proj_in_, {}}, {proj_out_, {}}};
}

}  // namespace strix::inkling2
