// Inkling 2 vision gate on the real checkpoint against the dumps of tools/vision/inkling2_vision_ref.py (transformers
// 5.24 Inkling2VisionModel run in FP32 on the checkpoint's own vision tensors).
//
//   strix-inkling2-vision-gate[-hip] --checkpoint DIR --ref DIR [--backend host|hip] [--max-rel-l2 X] NAME...
//
// Per NAME (DIR/NAME.png with DIR/NAME.{grid.json,pixels.f32,hidden.f32,rows.f32}):
//   patches   decode + adapter.patches() against the reference pixel_values (exact)
//   encoder   adapter positions -> tower -> hidden rows (class row first) against the reference hidden states
//   rows      adapter.project() -> image rows against the reference merger output (relative L2, worst row cosine)
// Exit status 0 when every image passes the patch check and its rows stay within --max-rel-l2 (default 0.02).
#include "inkling2/inkling2_vision.hpp"
#include "strix/serve/json.hpp"
#include "strix/vision/compare.hpp"
#include "strix/vision/encoder.hpp"
#include "strix/vision/image.hpp"
#include "strix/vision/lookup_weights.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace strix;

std::string read_text(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + p.string());
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
std::vector<float> read_f32(const std::filesystem::path& p) {
    const auto s = read_text(p);
    if (s.size() % 4U) throw std::runtime_error("bad float dump " + p.string());
    std::vector<float> v(s.size() / 4U);
    std::memcpy(v.data(), s.data(), s.size());
    return v;
}

struct RowStats {
    double rel_l2 = 0.0, worst_cos = 1.0, max_abs = 0.0;
};
RowStats compare_rows(const std::vector<float>& got, const std::vector<float>& want, std::size_t width) {
    RowStats r;
    double num = 0.0, den = 0.0;
    for (std::size_t row = 0; row * width < want.size(); ++row) {
        double dot = 0.0, a2 = 0.0, b2 = 0.0;
        for (std::size_t c = 0; c < width; ++c) {
            const double a = got[row * width + c], b = want[row * width + c];
            dot += a * b;
            a2 += a * a;
            b2 += b * b;
            num += (a - b) * (a - b);
            r.max_abs = std::max(r.max_abs, std::fabs(a - b));
        }
        den += b2;
        if (a2 > 0.0 && b2 > 0.0) r.worst_cos = std::min(r.worst_cos, dot / std::sqrt(a2 * b2));
    }
    r.rel_l2 = den > 0.0 ? std::sqrt(num / den) : 0.0;
    return r;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) try {
    std::string checkpoint, ref, backend_name = "host";
    double max_rel_l2 = 0.02;
    std::vector<std::string> names;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return argv[++i];
        };
        if (a == "--checkpoint") checkpoint = value();
        else if (a == "--ref") ref = value();
        else if (a == "--backend") backend_name = value();
        else if (a == "--max-rel-l2") max_rel_l2 = std::stod(value());
        else names.push_back(a);
    }
    if (checkpoint.empty() || ref.empty() || names.empty()) {
        std::fprintf(stderr, "usage: %s --checkpoint DIR --ref DIR [--backend host|hip] [--max-rel-l2 X] NAME...\n", argv[0]);
        return 2;
    }
    const std::filesystem::path dir(checkpoint), refdir(ref);
    const auto spec = inkling2::parse_vision_spec(read_text(dir / "config.json"));
    const auto specs = inkling2::vision_tensor_specs(spec);
    weight::TensorLookup lookup;
    const weight::TensorSchema schema{weight::Container::safetensors, "safetensors@1", specs, inkling2::kVisionIgnoredPrefix};
    auto t0 = std::chrono::steady_clock::now();
    if (const auto st = lookup.open_safetensors((dir / "model-00001-of-00001.safetensors").string(), schema); !st)
        throw std::runtime_error("vision tensors do not match the schema");
    vision::LookupWeights source(lookup);
    inkling2::VisionAdapter adapter(spec, "gate");
    vision::EncoderWeights weights;
    adapter.bind(source, weights);
    const auto cfg = adapter.config();
    std::printf("bind %.0f ms\n", ms_since(t0));
    std::unique_ptr<vision::EncoderBackend> backend;
#if defined(STRIX_INKLING2_VISION_GATE_HIP)
    if (backend_name == "hip") {
        t0 = std::chrono::steady_clock::now();
        auto enc = std::make_unique<vision::Encoder>(cfg, adapter.device_options());
        enc->load(weights);
        std::printf("hip load %.0f ms (%s)\n", ms_since(t0), enc->arithmetic());
        backend = std::move(enc);
    }
#endif
    if (!backend) {
        if (backend_name != "host") throw std::invalid_argument("backend " + backend_name + " is not built in this tool");
        backend = std::make_unique<vision::ReferenceEncoder>(cfg, weights);
    }
    bool ok = true;
    for (const auto& name : names) {
        const auto grid = serve::parse_json(read_text(refdir / (name + ".grid.json")));
        const auto bytes = read_text(refdir / (name + ".png"));
        const auto image = vision::decode(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()),
                                          vision::DecodeLimits{}, inkling2::VisionAdapter::color_policy());
        const auto g = adapter.patches(image);
        const auto tg = adapter.token_grid(g);
        const auto want_rows_t = static_cast<unsigned>(grid.find("rows")->as_int());
        const auto want_cols_t = static_cast<unsigned>(grid.find("cols")->as_int());
        // 1. patches: reference pixel_values [token][40][40][3], ours row 1 + (token * 4 + dy * 2 + dx), (py, px, c)
        const auto pixels = read_f32(refdir / (name + ".pixels.f32"));
        double patch_max = 0.0;
        bool shape = tg.rows == want_rows_t && tg.cols == want_cols_t && pixels.size() == std::size_t{tg.rows} * tg.cols * 4800U;
        if (shape) {
            for (unsigned t = 0; t < tg.rows * tg.cols; ++t)
                for (unsigned dy = 0; dy < 2U; ++dy)
                    for (unsigned dx = 0; dx < 2U; ++dx) {
                        const float* row = g.data.data() + (1U + t * 4U + dy * 2U + dx) * std::size_t{g.patch_dim};
                        for (unsigned py = 0; py < 20U; ++py)
                            for (unsigned px = 0; px < 20U; ++px)
                                for (unsigned c = 0; c < 3U; ++c) {
                                    const float want = pixels[((std::size_t{t} * 40U + dy * 20U + py) * 40U + dx * 20U + px) * 3U + c];
                                    patch_max = std::max(patch_max, std::fabs(double(row[(py * 20U + px) * 3U + c]) - want));
                                }
                    }
        }
        // 2. encoder + 3. rows
        t0 = std::chrono::steady_clock::now();
        const auto pos = adapter.positions(cfg, weights, g);
        auto out = backend->encode(g, pos);
        const double encode_ms = ms_since(t0);
        const auto hid = compare_rows(out.hidden, read_f32(refdir / (name + ".hidden.f32")), cfg.width);
        t0 = std::chrono::steady_clock::now();
        vision::ProjectedRows rows;
        adapter.project(out, g, *backend, rows);
        const double project_ms = ms_since(t0);
        std::vector<float> got(rows.bf16.size());
        for (std::size_t i = 0; i < got.size(); ++i) got[i] = vision::from_bf16(rows.bf16[i]);
        const auto want = read_f32(refdir / (name + ".rows.f32"));
        const bool rows_shape = want.size() == got.size();
        const auto rs = rows_shape ? compare_rows(got, want, rows.width) : RowStats{1.0, 0.0, 1e30};
        const bool pass = shape && patch_max == 0.0 && rows_shape && rs.rel_l2 <= max_rel_l2;
        ok = ok && pass;
        std::printf("%-22s %ux%u tokens %u | patches max|d| %.3g%s | hidden relL2 %.3e worst cos %.6f | rows relL2 %.3e "
                    "worst cos %.6f max|d| %.3g | encode %.1f ms project %.1f ms | %s\n",
                    name.c_str(), tg.rows, tg.cols, rows.rows, patch_max, shape ? "" : " (shape!)", hid.rel_l2, hid.worst_cos,
                    rs.rel_l2, rs.worst_cos, rs.max_abs, encode_ms, project_ms, pass ? "PASS" : "FAIL");
    }
    return ok ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "inkling2 vision gate: %s\n", e.what());
    return 2;
}
