#include "strix/qwen_moe_prepared.hpp"
#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strix::qwen_moe_prepared {
namespace {
std::uint64_t multiply(std::uint64_t a, std::uint64_t b) {
    if (b && a > std::numeric_limits<std::uint64_t>::max() / b)
        throw std::overflow_error("prepared MoE layout size overflow");
    return a * b;
}
Layout unit_layout(Format format) {
    switch (format) {
    case Format::q4_k: return {layout_version, 144, 256, 0, 2304, 0, 0};
    case Format::q5_k: return {layout_version, 176, 256, 512, 2816, 0, 0};
    case Format::q5_1: return {layout_version, 192, 512, 512, 3072, 0, 0};
    }
    throw std::invalid_argument("unsupported prepared MoE format");
}
void validate_buffers(const Layout& layout, std::span<const std::byte> src, std::span<std::byte> dst) {
    if (src.size() != layout.unit_bytes || dst.size() != layout.unit_bytes)
        throw std::invalid_argument("prepared MoE unit buffer size mismatch");
    const auto a = reinterpret_cast<std::uintptr_t>(src.data());
    const auto b = reinterpret_cast<std::uintptr_t>(dst.data());
    if ((a <= b ? b - a : a - b) < layout.unit_bytes)
        throw std::invalid_argument("prepared MoE unit buffers overlap");
}
unsigned byte(std::byte b) { return std::to_integer<unsigned>(b); }
void permute(Format format, std::span<const std::byte> src, std::span<std::byte> dst, bool inverse) {
    const auto layout = unit_layout(format);
    validate_buffers(layout, src, dst);
    std::fill(dst.begin(), dst.end(), std::byte{0});
    const auto copy = [&](unsigned old_at, unsigned new_at, unsigned count) {
        if (inverse) std::memcpy(dst.data() + old_at, src.data() + new_at, count);
        else std::memcpy(dst.data() + new_at, src.data() + old_at, count);
    };
    for (unsigned row = 0; row < 16; ++row) {
        const unsigned row_at = row * layout.row_bytes;
        if (format == Format::q5_1) {
            // Eight independent K32 blocks, each d/m (4), qh (4), qs (16).
            for (unsigned group = 0; group < 8; ++group) {
                const unsigned old_at = row_at + group * 24;
                copy(old_at, (group * 16 + row) * 4, 4);
                copy(old_at + 4, layout.header_bytes + (group * 16 + row) * 4, 4);
                for (unsigned k = 0; k < 32; ++k) {
                    const unsigned old_q = old_at + 8 + k % 16;
                    const unsigned old_shift = 4 * (k / 16);
                    const unsigned new_q = layout.header_bytes + layout.high_bytes + (group * 16 + row) * 16 + k / 2;
                    const unsigned new_shift = 4 * (k % 2);
                    if (inverse) dst[old_q] |= std::byte(((byte(src[new_q]) >> new_shift) & 15U) << old_shift);
                    else dst[new_q] |= std::byte(((byte(src[old_q]) >> old_shift) & 15U) << new_shift);
                }
            }
        } else {
            copy(row_at, row * 16, 16);
            if (format == Format::q5_k) copy(row_at + 16, layout.header_bytes + row * 32, 32);
            const unsigned qs_at = format == Format::q5_k ? 48U : 16U;
            for (unsigned k = 0; k < 256; ++k) {
                const unsigned old_q = row_at + qs_at + (k / 64) * 32 + k % 32;
                const unsigned old_shift = 4 * ((k / 32) % 2);
                const unsigned new_q = layout.header_bytes + layout.high_bytes + ((k / 32) * 16 + row) * 16 + (k % 32) / 2;
                const unsigned new_shift = 4 * (k % 2);
                if (inverse) dst[old_q] |= std::byte(((byte(src[new_q]) >> new_shift) & 15U) << old_shift);
                else dst[new_q] |= std::byte(((byte(src[old_q]) >> old_shift) & 15U) << new_shift);
            }
        }
    }
}
} // namespace

Layout describe(Spec spec) {
    auto result = unit_layout(spec.format);
    if ((spec.format == Format::q5_1 && spec.role != Role::down) ||
        (spec.format != Format::q5_1 && spec.role != Role::gate && spec.role != Role::up))
        throw std::invalid_argument("unsupported prepared MoE tensor role");
    if (!spec.experts || !spec.outputs || spec.outputs % 16 || !spec.input || spec.input % 256)
        throw std::invalid_argument("prepared MoE requires complete expert/output16/K256 units");
    result.units = multiply(multiply(spec.experts, spec.outputs / 16), spec.input / 256);
    result.tensor_bytes = multiply(result.units, result.unit_bytes);
    if (result.tensor_bytes > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("prepared MoE layout exceeds address space");
    return result;
}

Selection select(const char* requested, Spec spec) {
    if (!requested || std::string_view(requested) == "0") return {false, false, "off"};
    if (std::string_view(requested) != "1") return {false, false, "invalid-switch"};
    try { (void)describe(spec); }
    catch (const std::invalid_argument&) { return {false, false, "unsupported-geometry"}; }
    catch (const std::overflow_error&) { return {false, false, "size-overflow"}; }
    if (spec.format != Format::q4_k) return {false, false, "format-consumer-pending"};
    if (spec.outputs % 128 || spec.input > 2560 || spec.outputs > 640 || spec.experts > 512)
        return {false, false, "consumer-geometry"};
    return {true, true, "candidate"};
}

std::optional<Spec> admitted_spec(const ModelManifest& manifest,std::uint32_t index) {
    if(index>=manifest.tensors.size())return {};
    const auto& t=manifest.tensors[index];
    const bool gate=t.name.ends_with(".ffn_gate_exps.weight"),up=t.name.ends_with(".ffn_up_exps.weight");
    if((!gate&&!up)||t.placement.kind!=PlacementKind::local||t.ggml_type!=12||t.dimensions.size()!=3)return {};
    if(t.dimensions[0]>UINT32_MAX||t.dimensions[1]>UINT32_MAX||t.dimensions[2]>UINT32_MAX)return {};
    const Spec spec{Format::q4_k,gate?Role::gate:Role::up,static_cast<std::uint32_t>(t.dimensions[2]),
        static_cast<std::uint32_t>(t.dimensions[1]),static_cast<std::uint32_t>(t.dimensions[0])};
    if(!select("1",spec).prepare||describe(spec).tensor_bytes!=t.byte_size)return {};
    const auto other=t.name.substr(0,t.name.size()-std::string_view(gate?"ffn_gate_exps.weight":"ffn_up_exps.weight").size())+
        (gate?"ffn_up_exps.weight":"ffn_gate_exps.weight");
    for(const auto& pair:manifest.tensors)
        if(pair.name==other&&pair.placement.kind==PlacementKind::local&&pair.ggml_type==12&&
            pair.dimensions==t.dimensions&&pair.byte_size==t.byte_size)return spec;
    return {};
}

ModelContentIdentity asset_identity(Spec spec, const Source& source) {
    const auto layout = describe(spec);
    if (!source.model_hash.valid()) throw std::invalid_argument("prepared MoE requires model content hash");
    // Explicit little-endian fields, independent of ABI padding and endianness.
    std::vector<std::byte> key;
    constexpr std::string_view domain = "STRIX-QWEN-MOE-COMPRESSED-PLANES";
    for (char c : domain) key.push_back(static_cast<std::byte>(c));
    key.push_back(std::byte{0});
    const auto u32 = [&](std::uint32_t n) {
        for (unsigned i = 0; i < 4; ++i) key.push_back(std::byte((n >> (8 * i)) & 255U));
    };
    const auto u64 = [&](std::uint64_t n) {
        for (unsigned i = 0; i < 8; ++i) key.push_back(std::byte((n >> (8 * i)) & 255U));
    };
    u32(layout.version);
    for (auto b : source.model_hash.bytes) key.push_back(std::byte(b));
    u32(source.tensor_index); u64(source.file_offset);
    u32(static_cast<std::uint32_t>(spec.format)); u32(static_cast<std::uint32_t>(spec.role));
    u32(spec.experts); u32(spec.outputs); u32(spec.input); u64(layout.tensor_bytes);
    return sha256_bytes(key);
}

void repack_unit(Format f, std::span<const std::byte> src, std::span<std::byte> dst) { permute(f, src, dst, false); }
void inverse_unit(Format f, std::span<const std::byte> src, std::span<std::byte> dst) { permute(f, src, dst, true); }

PreparedTensor prepare(Spec spec, const Source& source, std::uint64_t file_bytes,
                       const Read& read, const Allocate& allocate) {
    const auto layout = describe(spec);
    const auto identity = asset_identity(spec, source);
    if (!read || !allocate) throw std::invalid_argument("prepared MoE requires reader and allocator");
    if (source.file_offset > file_bytes || layout.tensor_bytes > file_bytes - source.file_offset)
        throw std::invalid_argument("prepared MoE source tensor outside file");
    auto allocation = allocate(layout.tensor_bytes);
    if (!allocation || !allocation->address() || allocation->bytes() != layout.tensor_bytes ||
        allocation->address() % 16 || layout.tensor_bytes > std::numeric_limits<std::uintptr_t>::max() - allocation->address())
        throw std::invalid_argument("prepared MoE allocator returned invalid final allocation");
    const std::uint64_t blocks = spec.input / 256;
    const auto chunk_units = std::min<std::uint64_t>(blocks, staging_units);
    const auto slab_bytes = static_cast<std::size_t>(chunk_units * layout.unit_bytes);
    std::vector<std::byte> raw(slab_bytes), packed(slab_bytes), gathered(layout.unit_bytes);
    const std::uint64_t row_stride = blocks * layout.row_bytes;
    std::uint64_t unit = 0;
    for (std::uint64_t expert = 0; expert < spec.experts; ++expert)
        for (std::uint64_t output = 0; output < spec.outputs; output += 16)
            for (std::uint64_t block = 0; block < blocks; block += chunk_units) {
                const auto count = std::min(chunk_units, blocks - block);
                const auto row_bytes = static_cast<std::size_t>(count * layout.row_bytes);
                for (std::uint64_t row = 0; row < 16; ++row) {
                    const auto offset = source.file_offset + (expert * spec.outputs + output + row) * row_stride + block * layout.row_bytes;
                    read(offset, std::span<std::byte>(raw).subspan(static_cast<std::size_t>(row) * row_bytes, row_bytes));
                }
                for (std::uint64_t k = 0; k < count; ++k) {
                    for (std::size_t row = 0; row < 16; ++row)
                        std::memcpy(gathered.data() + row * layout.row_bytes,
                                    raw.data() + row * row_bytes + k * layout.row_bytes, layout.row_bytes);
                    repack_unit(spec.format, gathered, std::span<std::byte>(packed).subspan(static_cast<std::size_t>(k * layout.unit_bytes), layout.unit_bytes));
                }
                allocation->upload(unit * layout.unit_bytes, std::span<const std::byte>(packed).first(static_cast<std::size_t>(count * layout.unit_bytes)));
                unit += count;
            }
    return {spec, layout, identity, std::move(allocation)};
}

PreparedTensor prepare_file(Spec spec, const Source& source, const std::filesystem::path& path, const Allocate& allocate) {
    // Validate before opening or seeking. Source snapshot/content verification is
    // owned by the model loader, as for the ordinary resident tensor bindings.
    (void)asset_identity(spec, source);
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open prepared MoE source file");
    const auto end = input.tellg();
    if (end < 0) throw std::runtime_error("cannot size prepared MoE source file");
    const Read read = [&](std::uint64_t offset, std::span<std::byte> dst) {
        input.seekg(static_cast<std::streamoff>(offset));
        input.read(reinterpret_cast<char*>(dst.data()), static_cast<std::streamsize>(dst.size()));
        if (!input) throw std::runtime_error("short prepared MoE source read");
    };
    return prepare(spec, source, static_cast<std::uint64_t>(end), read, allocate);
}

} // namespace strix::qwen_moe_prepared
