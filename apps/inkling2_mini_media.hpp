#pragma once
// Inkling 2 Mini image input of the serial CLI: the family's VisionAdapter, its tower inventory and the encoder
// backend of this build (HIP: vision::Encoder, CUDA: vision::CudaEncoder, CPU: the FP32 host reference), handed to the
// shared vision::SerialMedia. Model-specific only in what it names.
#include "inkling2/inkling2_mini_config.hpp"
#include "inkling2/inkling2_vision.hpp"
#include "strix/serve/inkling2_chat.hpp"
#include "strix/vision/serial_media.hpp"
#include "strix/vision/encoder.hpp"
#include "strix/weight/confined_file.hpp"
#if defined(__CUDACC__)
#include "strix/vision/encoder_cuda.hpp"
#endif

#include <filesystem>
#include <memory>
#include <string>

namespace strix::inkling2::cli {

inline constexpr std::size_t kMiniVisionConfigBytes = 1U << 20U;

struct MiniVisionCheckpoint {
    VisionSpec spec;
    std::shared_ptr<const weight::Source> shard;
};

inline MiniVisionCheckpoint read_mini_vision_checkpoint(const weight::ConfinedDirectory& directory,
                                                         const std::filesystem::path& basename) {
    auto shard = directory.open(basename);
    const auto sidecar = directory.read("config.json", kMiniVisionConfigBytes, true);
    return {sidecar ? parse_vision_spec(*sidecar) : VisionSpec{}, std::move(shard)};
}

inline std::unique_ptr<serve::SerialMediaInput> make_mini_media(std::shared_ptr<const weight::Source> checkpoint,
                                                               const VisionSpec& config,
                                                               VisionIdentity identity) {
    vision::SerialMediaSpec spec;
    const auto shard_bytes = checkpoint->size();
    spec.shard_source = std::move(checkpoint);
    spec.tensors = vision_tensor_specs(config);
    spec.ignored_prefix = std::string(kVisionIgnoredPrefix);
    spec.placeholder = std::string(serve::kInkling2ImagePlaceholder);
    spec.lm_width = config.text_width;
    // Weight identity of the embed rows: the shard's size (no cross-run cache, so no content hash is needed).
    auto adapter = std::make_unique<VisionAdapter>(
        config, "shard-bytes-" + std::to_string(shard_bytes), identity);
    return std::make_unique<vision::SerialMedia>(std::move(adapter), std::move(spec),
        [](const vision::EncoderConfig& c, const vision::EncoderWeights& w) -> std::unique_ptr<vision::EncoderBackend> {
#if defined(__HIPCC__)
            auto encoder = std::make_unique<vision::Encoder>(c);
            encoder->load(w);
            return encoder;
#elif defined(__CUDACC__)
            auto encoder = std::make_unique<vision::CudaEncoder>(c);
            encoder->load(w);
            return encoder;
#else
            return std::make_unique<vision::ReferenceEncoder>(c, w);  // the CPU reference CLI
#endif
        });
}

inline std::unique_ptr<serve::SerialMediaInput> open_mini_media(const std::string& checkpoint) {
    const std::filesystem::path model(checkpoint);
    const auto parent = model.has_parent_path() ? model.parent_path() : std::filesystem::path(".");
    const weight::ConfinedDirectory directory(std::filesystem::canonical(parent));
    auto assets = read_mini_vision_checkpoint(directory, model.filename());
    return make_mini_media(std::move(assets.shard), assets.spec, resolve_vision_identity());
}

}  // namespace strix::inkling2::cli
