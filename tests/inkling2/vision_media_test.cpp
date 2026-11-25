#include "inkling2_mini_media.hpp"
#include "vision_test_dir.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace {

using namespace strix;
namespace ink = strix::inkling2;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void write_tower(const std::filesystem::path& path, const ink::VisionSpec& spec) {
    auto header = serve::Json::object();
    std::string data;
    std::uint32_t seed = 7;
    for (const auto& tensor : ink::vision_tensor_specs(spec)) {
        auto dims = serve::Json::array();
        std::size_t count = 1;
        for (const auto dim : tensor.extents) {
            dims.push(serve::Json(dim));
            count *= static_cast<std::size_t>(dim);
        }
        auto offsets = serve::Json::array();
        offsets.push(serve::Json(std::uint64_t{data.size()}));
        for (std::size_t i = 0; i < count; ++i) {
            seed = 1664525U * seed + 1013904223U;
            const float value = (static_cast<float>(seed >> 24U) - 128.0F) / 512.0F;
            const auto bits = vision::to_bf16(value + (tensor.name.find("ln_") != std::string::npos &&
                                                       tensor.name.ends_with(".weight") ? 1.0F : 0.0F));
            data.push_back(static_cast<char>(bits & 0xffU));
            data.push_back(static_cast<char>(bits >> 8U));
        }
        offsets.push(serve::Json(std::uint64_t{data.size()}));
        auto entry = serve::Json::object();
        entry.set("dtype", serve::Json(std::string(tensor.wire_dtype)));
        entry.set("shape", std::move(dims));
        entry.set("data_offsets", std::move(offsets));
        header.set(tensor.name, std::move(entry));
    }
    auto json = header.dump();
    while (json.size() % 8U) json.push_back(' ');
    std::ofstream output(path, std::ios::binary);
    const std::uint64_t size = json.size();
    for (unsigned shift = 0; shift < 64U; shift += 8U)
        output.put(static_cast<char>((size >> shift) & 0xffU));
    output << json << data;
    require(static_cast<bool>(output), "write tiny safetensors tower");
}

void test_serial_rgba() {
    const auto empty_env = [](const char*) -> const char* { return nullptr; };
    const auto axial_env = [](const char* key) -> const char* {
        return std::string_view(key) == "STRIX_INKLING2_VISION_RECIPE" ? "axial@1" : nullptr;
    };
    const auto incumbent = ink::resolve_vision_identity(empty_env);
    const auto selected = ink::resolve_vision_identity(axial_env);
    require(incumbent == ink::VisionIdentity::legacy, "incumbent image recipe");
    require(selected == ink::VisionIdentity::axial_v1, "explicit axial image recipe");

    // Real CLI media owner, real schema and PNG, deterministic host tower.
    ink::VisionSpec spec;
    spec.layers = 1;
    spec.width = 16;
    spec.heads = 2;
    spec.mlp_width = 24;
    spec.patch = 4;
    spec.pos_grid = 4;
    spec.projection = 20;
    spec.text_width = 12;
    VisionTestDir fixture;
    const auto shard = fixture.path / "tower.safetensors";
    write_tower(shard, spec);
    auto serial = ink::cli::make_mini_media(shard.string(), spec, selected);
    auto legacy_serial = ink::cli::make_mini_media(shard.string(), spec, incumbent);
    auto& bridge = serial->adapter();
    require(bridge.identity().find("inkling2-vision-axial@1|") == 0, "explicit media identity");
    require(legacy_serial->adapter().identity().find("inkling2-vision-v1|") == 0, "legacy media identity");
    std::ifstream png(std::filesystem::path(STRIX_VISION_FIXTURE_DIR) / "rgba_noise.png", std::ios::binary);
    serve::media::MediaItem item;
    item.bytes.assign(std::istreambuf_iterator<char>(png), std::istreambuf_iterator<char>());
    require(!item.bytes.empty(), "read RGBA fixture");
    const auto span = bridge.encode(item, 4096);
    auto legacy_span = legacy_serial->adapter().encode(item, 4096);

    weight::TensorLookup lookup;
    const auto specs = ink::vision_tensor_specs(spec);
    require(static_cast<bool>(lookup.open_safetensors(shard.string(),
            {weight::Container::safetensors, "safetensors@1", specs, ink::kVisionIgnoredPrefix})), "open fixture tower");
    vision::LookupWeights source(lookup);
    const std::span png_bytes(reinterpret_cast<const std::uint8_t*>(item.bytes.data()), item.bytes.size());
    vision::ColorPolicy checker;
    checker.alpha = vision::AlphaPolicy::checkerboard;
    const auto rgb = vision::decode(png_bytes);
    const auto composited = vision::decode(png_bytes, {}, checker);
    const auto encode = [&](ink::VisionIdentity identity, const vision::Image& image) {
        ink::VisionAdapter adapter(spec, "fixture", identity);
        vision::EncoderWeights weights;
        adapter.bind(source, weights);
        vision::ReferenceEncoder encoder(adapter.config(), weights);
        const auto grid = adapter.patches(image);
        auto hidden = encoder.encode(grid, adapter.positions(adapter.config(), weights, grid));
        vision::ProjectedRows projected;
        adapter.project(hidden, grid, encoder, projected);
        return projected.bf16;
    };
    require(span.lease.rows().data == encode(selected, composited), "CLI RGBA rows must use the checkerboard");
    require(span.lease.rows().data != encode(selected, rgb), "RGBA fixture must distinguish alpha policies");
    require(legacy_span.lease.rows().data == encode(incumbent, rgb), "legacy media must keep alpha drop");
}

}  // namespace

int main() try {
    test_serial_rgba();
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
