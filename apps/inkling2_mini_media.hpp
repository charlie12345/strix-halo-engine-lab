#pragma once
// Inkling 2 Mini image input of the serial CLI: the family's VisionAdapter, its tower inventory and the encoder
// backend of this build (HIP: vision::Encoder, CUDA: vision::CudaEncoder, CPU: the FP32 host reference), handed to the
// shared vision::SerialMedia. Model-specific only in what it names.
#include "inkling2/inkling2_mini_config.hpp"
#include "inkling2/inkling2_vision.hpp"
#include "strix/serve/inkling2_chat.hpp"
#include "strix/vision/serial_media.hpp"
#include "strix/vision/encoder.hpp"
#if defined(__CUDACC__)
#include "strix/vision/encoder_cuda.hpp"
#endif

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>

namespace strix::inkling2::cli {

// The tower geometry of the checkpoint's own config.json (vision_config, checked by parse_vision_spec). The checkpoint
// and its config are one authorized set: config.json must be a regular file, not a link, at most 1 MiB, in the same
// real directory as the checkpoint.
inline VisionSpec read_mini_vision_spec(const std::string& checkpoint) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::canonical(checkpoint).parent_path();
    const fs::path path = dir / "config.json";
    std::error_code ec;
    if (fs::is_symlink(path, ec) || !fs::is_regular_file(path, ec))
        throw std::runtime_error("Inkling 2 image input needs config.json (a regular file) beside the checkpoint");
    if (fs::canonical(path, ec).parent_path() != dir || ec)
        throw std::runtime_error("Inkling 2 image input: config.json resolves outside the checkpoint directory");
    const auto size = fs::file_size(path, ec);
    if (ec || size > (std::uintmax_t{1} << 20U))
        throw std::runtime_error("Inkling 2 image input: config.json is unreadable or larger than 1 MiB");
    std::ifstream in(path, std::ios::binary);
    std::string text(static_cast<std::size_t>(size), ' ');
    if (!in.read(text.data(), static_cast<std::streamsize>(size)))
        throw std::runtime_error("Inkling 2 image input: short read of config.json");
    return parse_vision_spec(text);
}

inline std::unique_ptr<serve::SerialMediaInput> open_mini_media(const std::string& checkpoint) {
    const VisionSpec config = read_mini_vision_spec(checkpoint);
    vision::SerialMediaSpec spec;
    spec.shard = checkpoint;
    spec.tensors = vision_tensor_specs(config);
    spec.ignored_prefix = std::string(kVisionIgnoredPrefix);
    spec.placeholder = std::string(serve::kInkling2ImagePlaceholder);
    spec.lm_width = config.text_width;
    // Weight identity of the embed rows: the shard's size (no cross-run cache, so no content hash is needed).
    auto adapter = std::make_unique<VisionAdapter>(
        config, "shard-bytes-" + std::to_string(std::filesystem::file_size(checkpoint)));
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

}  // namespace strix::inkling2::cli
