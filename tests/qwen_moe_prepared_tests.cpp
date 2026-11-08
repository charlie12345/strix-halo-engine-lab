#include "qwen_moe_prepared_fixture.hpp"
#include "strix/model_content_hash.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

namespace {
using namespace qwen_moe_prepared_fixture;
template<class F> void rejects(F&& f) {
    bool caught = false;
    try { f(); } catch (const std::exception&) { caught = true; }
    require(caught, "invalid preparation accepted");
}
struct State {
    unsigned allocations = 0, live = 0, uploads = 0, reads = 0, fail_upload = 0;
    std::size_t max_read = 0, max_upload = 0;
    std::uint64_t uploaded_bytes = 0;
};
class HostAllocation final : public qp::Allocation {
    State& state_;
    std::vector<std::byte> storage_;
    std::byte* aligned_ = nullptr;
    std::uint64_t size_ = 0;
public:
    HostAllocation(std::uint64_t bytes, State& state) : state_(state), storage_(static_cast<std::size_t>(bytes) + 15), size_(bytes) {
        const auto base = reinterpret_cast<std::uintptr_t>(storage_.data());
        aligned_ = reinterpret_cast<std::byte*>((base + 15) & ~std::uintptr_t{15});
        ++state_.allocations; ++state_.live;
    }
    ~HostAllocation() override { --state_.live; }
    std::uintptr_t address() const noexcept override { return reinterpret_cast<std::uintptr_t>(aligned_); }
    std::uint64_t bytes() const noexcept override { return size_; }
    void upload(std::uint64_t offset, std::span<const std::byte> src) override {
        require(offset == state_.uploaded_bytes, "noncontiguous/duplicate destination write");
        require(offset <= size_ && src.size() <= size_ - offset, "upload outside final allocation");
        if (++state_.uploads == state_.fail_upload) throw std::runtime_error("injected upload failure");
        state_.max_upload = std::max(state_.max_upload, src.size());
        std::memcpy(aligned_ + offset, src.data(), src.size());
        state_.uploaded_bytes += src.size();
    }
};
qp::Allocate allocator(State& state) {
    return [&](std::uint64_t bytes) { return std::make_unique<HostAllocation>(bytes, state); };
}
std::span<const std::byte> contents(const qp::PreparedTensor& tensor) {
    return {reinterpret_cast<const std::byte*>(tensor.allocation->address()), static_cast<std::size_t>(tensor.layout.tensor_bytes)};
}
void test_units(qp::Format f) {
    const auto layout = qp::describe(spec(f));
    require(layout.version == 1 && layout.unit_bytes == (f == qp::Format::q4_k ? 2304U : f == qp::Format::q5_k ? 2816U : 3072U), "versioned byte map changed");
    std::vector<std::byte> packed(layout.unit_bytes), inverse(layout.unit_bytes);
    // Every byte value, every header/code/high-plane position, and all FP16 bit
    // patterns below survive without floating-point evaluation or canonicalizing.
    for (unsigned value = 0; value < 256; ++value) {
        std::vector<std::byte> raw(layout.unit_bytes, std::byte(value));
        qp::repack_unit(f, raw, packed); unit_oracle(f, raw, packed);
        if(f==qp::Format::q4_k)direct_b_word_oracle(raw,packed);
    }
    for (unsigned seed = 0; seed < 32; ++seed) {
        auto raw = input(layout.unit_bytes, seed);
        qp::repack_unit(f, raw, packed); unit_oracle(f, raw, packed);
        if(f==qp::Format::q4_k)direct_b_word_oracle(raw,packed);
        // Inverse must be onto even for arbitrary prepared bytes.
        qp::inverse_unit(f, raw, inverse); qp::repack_unit(f, inverse, packed);
        require(raw == packed, "arbitrary prepared bytes not invertible");
    }
    for (unsigned bits : {0x0000U, 0x8000U, 0x0001U, 0x8001U, 0x03ffU, 0x0400U, 0x3555U, 0x3c01U, 0x7bffU, 0x7c00U, 0x7e01U, 0xffffU}) {
        auto raw = input(layout.unit_bytes);
        for (unsigned row = 0; row < 16; ++row) {
            raw[row * layout.row_bytes] = std::byte(bits & 255U);
            raw[row * layout.row_bytes + 1] = std::byte(bits >> 8);
        }
        qp::repack_unit(f, raw, packed); unit_oracle(f, raw, packed);
        if(f==qp::Format::q4_k)direct_b_word_oracle(raw,packed);
    }
    auto raw = input(layout.unit_bytes + 1);
    rejects([&] { qp::repack_unit(f, std::span(raw).first(layout.unit_bytes), std::span(raw).subspan(1)); });
    rejects([&] { qp::inverse_unit(f, packed, packed); });
    rejects([&] { qp::repack_unit(f, raw, packed); });
}
void test_tensor(qp::Format f, unsigned blocks) {
    auto s = spec(f); s.input = 256 * blocks;
    const auto src = source(); const auto layout = qp::describe(s);
    const auto chunks = static_cast<unsigned>(s.experts * (s.outputs / 16) * ((blocks + qp::staging_units - 1) / qp::staging_units));
    const auto chunk_units = std::min(blocks, qp::staging_units);
    auto file = input(static_cast<std::size_t>(src.file_offset + layout.tensor_bytes + 19));
    const auto unchanged = file;
    State state;
    const qp::Read read = [&](std::uint64_t offset, std::span<std::byte> dst) {
        require(offset >= src.file_offset && offset + dst.size() <= src.file_offset + layout.tensor_bytes, "read outside source tensor");
        ++state.reads; state.max_read = std::max(state.max_read, dst.size());
        std::copy_n(file.begin() + static_cast<std::ptrdiff_t>(offset), dst.size(), dst.begin());
    };
    {
        auto tensor = qp::prepare(s, src, file.size(), read, allocator(state));
        require(state.allocations == 1 && state.live == 1, "not exactly one resident allocation");
        require(state.uploads == chunks && state.reads == chunks * 16, "incorrect bounded staging work");
        require(state.max_read == chunk_units * layout.row_bytes && state.max_upload == chunk_units * layout.unit_bytes, "staging bound changed");
        require(state.uploaded_bytes == layout.tensor_bytes, "incomplete final upload");
        require(tensor.layout.tensor_bytes == layout.units * 16 * layout.row_bytes, "resident expansion");
        require(tensor.identity == qp::asset_identity(s, src), "published identity mismatch");
        tensor_oracle(s, std::span(file).subspan(static_cast<std::size_t>(src.file_offset), static_cast<std::size_t>(layout.tensor_bytes)), contents(tensor));
    }
    require(!state.live && file == unchanged, "owner/source lifetime failure");
    for (unsigned fail_at : {1U, 2U, chunks}) {
        State failure; failure.fail_upload = fail_at;
        rejects([&] { (void)qp::prepare(s, src, file.size(), read, allocator(failure)); });
        require(failure.allocations == 1 && !failure.live, "failed upload leaked/published final allocation");
    }
    State failure;
    unsigned reads = 0;
    const qp::Read short_read = [&](std::uint64_t offset, std::span<std::byte> dst) {
        if (++reads == 18) throw std::runtime_error("injected short read");
        read(offset, dst);
    };
    rejects([&] { (void)qp::prepare(s, src, file.size(), short_read, allocator(failure)); });
    require(!failure.live && failure.uploads == 1, "failed read leaked/published allocation");
    State rejected;
    rejects([&] { (void)qp::prepare(s, src, src.file_offset + layout.tensor_bytes - 1, read, allocator(rejected)); });
    auto invalid_source = src; invalid_source.file_offset = std::numeric_limits<std::uint64_t>::max();
    rejects([&] { (void)qp::prepare(s, invalid_source, file.size(), read, allocator(rejected)); });
    require(!rejected.allocations, "bad file range allocated");

    // Exercise the real seek/read adapter with a prefix, suffix and multiple
    // experts/output tiles/K units. Test paths never enter the asset identity.
    const auto path = std::filesystem::temp_directory_path() / ("strix-moe-prepared-" + strix::model_content_identity_hex(qp::asset_identity(s, src)) + ".bin");
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::error_code ec; std::filesystem::remove(path, ec); } } cleanup{path};
    { std::ofstream out(path, std::ios::binary); out.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size())); require(out.good(), "fixture write failed"); }
    State from_file;
    auto tensor = qp::prepare_file(s, src, path, allocator(from_file));
    tensor_oracle(s, std::span(file).subspan(static_cast<std::size_t>(src.file_offset), static_cast<std::size_t>(layout.tensor_bytes)), contents(tensor));
}
void test_identity_and_selector() {
    const auto s = spec(qp::Format::q4_k); const auto src = source();
    const auto key = qp::asset_identity(s, src);
    require(key.valid() && key == qp::asset_identity(s, src), "unstable prepared identity");
    auto changed = src; changed.model_hash.bytes[31] ^= 1;
    require(key != qp::asset_identity(s, changed), "model content hash omitted");
    changed = src; ++changed.tensor_index; require(key != qp::asset_identity(s, changed), "tensor index omitted");
    changed = src; ++changed.file_offset; require(key != qp::asset_identity(s, changed), "source offset omitted");
    auto geometry = s;
    for (unsigned which = 0; which < 5; ++which) {
        geometry = s;
        if (which == 0) geometry.role = qp::Role::up;
        if (which == 1) geometry.format = qp::Format::q5_k;
        if (which == 2) ++geometry.experts;
        if (which == 3) geometry.outputs += 16;
        if (which == 4) geometry.input += 256;
        require(key != qp::asset_identity(geometry, src), "layout descriptor field omitted");
    }
    // Pin the canonical encoding, including its version field, independently.
    std::string serialized = "STRIX-QWEN-MOE-COMPRESSED-PLANES"; serialized.push_back('\0');
    const auto append = [&](std::uint64_t n, unsigned bytes) { for (unsigned i = 0; i < bytes; ++i) serialized.push_back(static_cast<char>((n >> (8 * i)) & 255U)); };
    append(qp::layout_version, 4);
    for (auto b : src.model_hash.bytes) serialized.push_back(static_cast<char>(b));
    append(src.tensor_index, 4); append(src.file_offset, 8); append(12, 4); append(1, 4);
    append(s.experts, 4); append(s.outputs, 4); append(s.input, 4); append(qp::describe(s).tensor_bytes, 8);
    const auto digest = [&] { return strix::sha256_bytes(std::as_bytes(std::span(serialized))); };
    require(key == digest(), "asset identity encoding/version differs");
    serialized[std::string_view("STRIX-QWEN-MOE-COMPRESSED-PLANES").size() + 1] ^= 1;
    require(key != digest(), "layout version omitted");
    changed = src; changed.model_hash = {};
    rejects([&] { (void)qp::asset_identity(s, changed); });
    for (const char* off : {static_cast<const char*>(nullptr), "0"}) {
        const auto selection = qp::select(off, s);
        require(!selection.prepare && !selection.execute && selection.reason == "off", "OFF selector changed");
    }
    for (const char* invalid : {"", "01", "true", "2", "-1"}) {
        const auto selection = qp::select(invalid, s);
        require(!selection.prepare && !selection.execute && selection.reason == "invalid-switch", "invalid switch accepted");
    }
    const qp::Spec production{qp::Format::q4_k,qp::Role::gate,512,512,2560};
    const auto selected = qp::select("1", production);
    require(selected.prepare && selected.execute && selected.reason == "candidate", "Q4_K consumer not admitted");
    auto unsupported=production;unsupported.format=qp::Format::q5_k;
    require(!qp::select("1",unsupported).prepare && qp::select("1",unsupported).reason=="format-consumer-pending", "Q5_K consumer admitted");
    unsupported=production;unsupported.outputs=16;
    require(!qp::select("1",unsupported).execute && qp::select("1",unsupported).reason=="consumer-geometry", "consumer tail admitted");
    strix::ModelManifest manifest;
    strix::ManifestTensor g;g.name="blk.0.ffn_gate_exps.weight";g.ggml_type=12;
    g.dimensions={2560,512,512};g.byte_size=qp::describe(production).tensor_bytes;
    auto u=g;u.name="blk.0.ffn_up_exps.weight";
    manifest.tensors={g,u};
    require(qp::admitted_spec(manifest,0)->role==qp::Role::gate && qp::admitted_spec(manifest,1)->role==qp::Role::up,"paired admission");
    for(unsigned mutation=0;mutation<5;++mutation) {
        auto bad=manifest;
        if(mutation==0)bad.tensors.pop_back();
        if(mutation==1)bad.tensors[1].ggml_type=13;
        if(mutation==2)bad.tensors[1].dimensions[0]=512;
        if(mutation==3)bad.tensors[1].placement.kind=strix::PlacementKind::remote;
        if(mutation==4)--bad.tensors[1].byte_size;
        require(!qp::admitted_spec(bad,0),"incomplete or mixed pair admitted");
    }
    for (unsigned which = 0; which < 9; ++which) {
        geometry = s;
        if (which == 0) geometry.experts = 0;
        if (which == 1) geometry.outputs = 17;
        if (which == 2) geometry.input = 257;
        if (which == 3) geometry.format = static_cast<qp::Format>(99);
        if (which == 4) geometry.role = qp::Role::down;
        if (which == 5) geometry.role = static_cast<qp::Role>(99);
        if (which == 6) geometry = {qp::Format::q5_1, qp::Role::gate, 2, 32, 256};
        if (which == 7) geometry.outputs = 0;
        if (which == 8) geometry.input = 0;
        rejects([&] { (void)qp::describe(geometry); });
        require(!qp::select("1", geometry).prepare, "unsupported geometry selected");
        State state;
        rejects([&] { (void)qp::prepare(geometry, src, 0, {}, allocator(state)); });
        require(!state.allocations, "unsupported geometry allocated");
    }
    geometry = {qp::Format::q4_k, qp::Role::gate, 0xffffffffU, 0xfffffff0U, 0xffffff00U};
    require(qp::select("1", geometry).reason == "size-overflow", "overflow selector");

    const auto size = qp::describe(s).tensor_bytes;
    const qp::Read unused = [](std::uint64_t, std::span<std::byte>) { throw std::runtime_error("unexpected read"); };
    rejects([&] { (void)qp::prepare(s, src, src.file_offset + size, unused, {}); });
    State state;
    rejects([&] { (void)qp::prepare(s, src, src.file_offset + size, {}, allocator(state)); });
    require(!state.allocations, "missing reader allocated");
    const qp::Allocate empty = [](std::uint64_t) { return std::unique_ptr<qp::Allocation>{}; };
    rejects([&] { (void)qp::prepare(s, src, src.file_offset + size, unused, empty); });
    class InvalidAllocation final : public qp::Allocation {
        std::uintptr_t address_;
        std::uint64_t bytes_;
        bool& live_;
    public:
        InvalidAllocation(std::uintptr_t address, std::uint64_t bytes, bool& live) : address_(address), bytes_(bytes), live_(live) { live_ = true; }
        ~InvalidAllocation() override { live_ = false; }
        std::uintptr_t address() const noexcept override { return address_; }
        std::uint64_t bytes() const noexcept override { return bytes_; }
        void upload(std::uint64_t, std::span<const std::byte>) override { throw std::runtime_error("unexpected upload"); }
    };
    for (unsigned which = 0; which < 4; ++which) {
        bool live = false;
        const qp::Allocate bad = [&](std::uint64_t bytes) {
            const auto address = which == 0 ? std::uintptr_t{0} : which == 1 ? std::uintptr_t{0x1001}
                : which == 2 ? std::numeric_limits<std::uintptr_t>::max() & ~std::uintptr_t{15} : std::uintptr_t{0x1000};
            return std::make_unique<InvalidAllocation>(address, which == 3 ? bytes - 1 : bytes, live);
        };
        bool invalid_binding = false;
        try { (void)qp::prepare(s, src, src.file_offset + size, unused, bad); }
        catch (const std::invalid_argument&) { invalid_binding = true; }
        require(invalid_binding && !live, "invalid final allocation not rejected/released before read");
    }
}
}
int main() {
    try {
        test_identity_and_selector();
        for (auto f : {qp::Format::q4_k, qp::Format::q5_k, qp::Format::q5_1}) {
            test_units(f);
            for (unsigned blocks : {1U, 3U, 65U}) test_tensor(f, blocks);
        }
        std::cout << "prepared MoE layout/identity/selector/upload tests passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
