#include "inkling2_mini_media.hpp"
#include "vision_test_dir.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#ifndef _WIN32
#include <sys/stat.h>
#endif

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
    const weight::ConfinedDirectory directory(fixture.path);
    const auto assets = ink::cli::read_mini_vision_checkpoint(directory, shard.filename());
    auto serial = ink::cli::make_mini_media(assets.shard, spec, selected);
    auto legacy_serial = ink::cli::make_mini_media(assets.shard, spec, incumbent);
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

void test_checkpoint_admission() {
    namespace fs = std::filesystem;
    const auto write = [](const fs::path& name, std::string_view bytes) {
        std::ofstream stream(name, std::ios::binary);
        stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        require(static_cast<bool>(stream), "write checkpoint fixture");
    };
    const auto rejects = [](const auto& call, std::string_view expected) {
        try {
            call();
        } catch (const std::exception& error) {
            return std::string_view(error.what()).find(expected) != std::string_view::npos;
        }
        return false;
    };
    constexpr std::string_view valid = R"({"vision_config":{"model_type":"inkling2_vision"}})";
    VisionTestDir fixture;
    const auto model_dir = fixture.path / "model";
    fs::create_directory(model_dir);
    const auto config = model_dir / "config.json";
    write(model_dir / "tower.safetensors", "original");
    const weight::ConfinedDirectory directory(model_dir);
    const auto open = [&] { return ink::cli::read_mini_vision_checkpoint(directory, "tower.safetensors"); };
    require(open().spec.width == 1024U, "a shard alone uses the fixed spec");
    require(rejects([&] { (void)ink::cli::open_mini_media((model_dir / "tower.safetensors").string()); },
                    "tower tensors"), "the production opener reaches shard validation without a sidecar");
    std::string capped(valid);
    capped.resize(ink::cli::kMiniVisionConfigBytes, ' ');
    write(config, capped);
    require(open().spec.width == 1024U, "accept config at the one-MiB boundary");
    capped.push_back(' ');
    write(config, capped);
    require(rejects([&] { (void)open(); }, "length"), "refuse config over one MiB");
    write(config, valid.substr(0, valid.size() - 1U));
    require(rejects([&] { (void)open(); }, ""), "refuse a truncated config document");
    write(config, R"({"vision_config":{"model_type":"inkling2_vision","hidden_size":2}})");
    require(rejects([&] { (void)open(); }, "hidden_size"), "refuse unsupported geometry");

    fs::remove(config);
    fs::create_directory(config);
    require(rejects([&] { (void)open(); }, "config.json"), "refuse a directory in the config slot");
    fs::remove(config);
#ifndef _WIN32
    require(::mkfifo(config.c_str(), 0600) == 0, "make fixture FIFO");
    require(rejects([&] { (void)open(); }, "config.json"), "refuse a FIFO in the config slot");
    fs::remove(config);
#endif
    write(fixture.path / "outside.json", valid);
    std::error_code link_error;
    fs::create_symlink(fixture.path / "outside.json", config, link_error);
#ifndef _WIN32
    require(!link_error, "create sidecar symlink fixture");
#endif
    if (!link_error) {
        require(rejects([&] { (void)open(); }, "config.json"), "refuse a config symlink");
        fs::create_symlink(fixture.path / "outside.json", model_dir / "shard-link");
        require(rejects([&] { (void)ink::cli::read_mini_vision_checkpoint(directory, "shard-link"); },
                        "shard-link"), "refuse a shard symlink");
        fs::remove(config);
        fs::create_symlink(fixture.path / "absent.json", config);
        require(rejects([&] { (void)open(); }, "config.json"), "a dangling config symlink is not an absent sidecar");
        fs::remove(config);
    } else {
        std::fprintf(stderr, "symbolic link fixture unavailable: %s\n", link_error.message().c_str());
    }

    write(config, valid);
    fs::rename(model_dir, fixture.path / "held");
    fs::create_directory(model_dir);
    write(config, "{}");
    write(model_dir / "tower.safetensors", "replacement");
    const auto assets = open();
    std::string contents(8, '\0');
    require(assets.spec.width == 1024U &&
            static_cast<bool>(assets.shard->read({0, contents.size()},
                                                {reinterpret_cast<weight::Byte*>(contents.data()), contents.size()})) &&
            contents == "original", "config and shard must open within the same held directory");
}

}  // namespace

int main() try {
    test_serial_rgba();
    test_checkpoint_admission();
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
