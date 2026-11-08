#pragma once

#include "strix/model_runtime_binding.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string_view>

namespace strix::qwen_moe_prepared {

inline constexpr std::uint32_t layout_version = 1;
inline constexpr std::uint32_t staging_units = 64;
inline constexpr std::string_view control = "STRIX_QPF2K_MOE_DIRECT_B";
enum class Format : std::uint32_t { q4_k = 12, q5_k = 13, q5_1 = 7 };
enum class Role : std::uint32_t { gate = 1, up = 2, down = 3 };
struct Spec {
    Format format = Format::q4_k;
    Role role = Role::gate;
    std::uint32_t experts = 0, outputs = 0, input = 0;
};
struct Layout {
    std::uint32_t version = layout_version;
    std::uint32_t row_bytes = 0, header_bytes = 0, high_bytes = 0, unit_bytes = 0;
    std::uint64_t units = 0, tensor_bytes = 0;
};
// Complete output16/K256 units only. Unsupported formats, roles, tails and
// overflow are rejected before any I/O or allocation; original loaders remain usable.
[[nodiscard]] Layout describe(Spec);
struct Selection {
    bool prepare = false;
    bool execute = false;
    std::string_view reason;
};
// Pure layout/consumer inspection. nullptr/"0" is OFF; only "1" requests
// preparation. Q4_K gate/up consumers require the additional runtime guards.
[[nodiscard]] Selection select(const char* requested, Spec);
// Paired local Q4_K gate/up tensors only; excludes incomplete/mixed pairs.
[[nodiscard]] std::optional<Spec> admitted_spec(const ModelManifest&, std::uint32_t index);

// Source identity must come from the model loader's verified full-content hash.
// The loader retains/rechecks its source snapshot through preparation. Paths and
// pointer values never enter the asset key. Index and offset disambiguate tensors.
struct Source {
    ModelContentIdentity model_hash;
    std::uint32_t tensor_index = 0;
    std::uint64_t file_offset = 0;
};
[[nodiscard]] ModelContentIdentity asset_identity(Spec, const Source&);

// A unit's source is 16 consecutive gathered output rows, each covering K256.
// All header/high-plane bytes are preserved. Low nibbles are packed in natural
// K32 order, adjacent K values in low/high nibble. Buffers must not overlap.
void repack_unit(Format, std::span<const std::byte> source, std::span<std::byte> destination);
void inverse_unit(Format, std::span<const std::byte> source, std::span<std::byte> destination);

class Allocation {
public:
    virtual ~Allocation() = default;
    [[nodiscard]] virtual std::uintptr_t address() const noexcept = 0;
    [[nodiscard]] virtual std::uint64_t bytes() const noexcept = 0;
    // Must complete before returning; never retain staging pointers, even on
    // failure. The final allocation is unpublished until every upload succeeds.
    virtual void upload(std::uint64_t offset, std::span<const std::byte>) = 0;
};
using Allocate = std::function<std::unique_ptr<Allocation>(std::uint64_t)>;
// Complete the exact requested read or throw. Offsets are absolute file offsets.
using Read = std::function<void(std::uint64_t, std::span<std::byte>)>;
struct PreparedTensor {
    Spec spec;
    Layout layout;
    ModelContentIdentity identity;
    std::unique_ptr<Allocation> allocation;
};
// Exactly one final allocation; two bounded slabs plus one gather unit
// (<=396288 staging bytes). No tensor-sized host shadow, resident original, or global in-place
// permutation. Setup only: do not replace pointers used by live/captured work.
[[nodiscard]] PreparedTensor prepare(Spec, const Source&, std::uint64_t file_bytes,
                                     const Read&, const Allocate&);
[[nodiscard]] PreparedTensor prepare_file(Spec, const Source&, const std::filesystem::path&,
                                          const Allocate&);

} // namespace strix::qwen_moe_prepared
