#include "glm/engine/glm_kvshard_layout.hpp"
#include "glm/engine/glm_kvshard_dump.hpp"
#include <chrono>
#include <cstdlib>
#include "xport/kvshard_oracle.hpp"
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>

namespace K = strix::glm::kvshard;
namespace O = strix::xport::kvshard;
namespace {
unsigned checks = 0;
void require(bool good, const char* message) {
    ++checks;
    if (!good) throw std::runtime_error(message);
}
template<class Function> void rejects(Function function) {
    bool threw = false;
    try { function(); } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "missing layout rejection");
}

void mapping(std::uint64_t capacity) {
    const auto storage = K::storage_tokens(capacity);
    std::array<std::vector<std::uint64_t>, 2> compact;
    for (auto& rank : compact) rank.resize(static_cast<std::size_t>(storage), capacity);
    for (std::uint64_t token = 0; token < capacity; ++token) {
        const unsigned rank = K::owner(token);
        const auto local = K::local_token(token);
        require(local < storage, "compact cache address exceeds allocation");
        require(K::global_token(local, rank) == token, "global position changed");
        compact[rank][static_cast<std::size_t>(local)] = token;
        require(K::owned_prefix(token + 1, 0) + K::owned_prefix(token + 1, 1) == token + 1, "prefix token lost");
    }
    for (std::uint64_t token = 0; token < capacity; ++token)
        require(compact[K::owner(token)][static_cast<std::size_t>(K::local_token(token))] == token, "prefill scatter/gather changed bytes");
    require(storage % 4 == 0, "pool split across allocation boundary");
    for (unsigned rank = 0; rank < 2; ++rank) {
        const auto visible = K::owned_prefix(capacity, rank);
        for (std::uint64_t item = visible; item < storage; ++item)
            require(compact[rank][static_cast<std::size_t>(item)] == capacity, "rounded tail touched");
    }
}

void selection(std::uint64_t tokens) {
    const auto pools = static_cast<std::size_t>(tokens / 4);
    std::vector<std::uint32_t> full(pools), left(pools, 0xffffffffU), right(pools, 0xffffffffU);
    std::vector<O::Candidate> candidates;
    for (std::size_t pool = 0; pool < pools; ++pool) {
        // Heavy ties spanning both owners; poisoned unowned entries must never enter selection.
        full[pool] = static_cast<std::uint32_t>((pool * 31) % 17);
        (pool % 2 == 0 ? left : right)[pool] = full[pool];
        candidates.push_back({static_cast<std::uint64_t>(pool), static_cast<std::uint64_t>(pool * 4 + 3), static_cast<float>(full[pool])});
    }
    const auto expected = O::select_global(candidates, tokens - 1, 512);
    for (unsigned rank = 0; rank < 2; ++rank) {
        const auto merged = K::merge_score_keys(rank == 0 ? left : right, rank == 0 ? right : left, rank);
        require(merged == full, "owned score exchange changed global keys");
        std::vector<O::Candidate> global;
        for (std::size_t pool = 0; pool < pools; ++pool)
            global.push_back({static_cast<std::uint64_t>(pool), static_cast<std::uint64_t>(pool * 4 + 3), static_cast<float>(merged[pool])});
        require(O::select_global(global, tokens - 1, 512) == expected, "global top-k/ties changed");
    }
}

void snapshots() {
    const auto key = K::snapshot_key(65536, 1, 8195, 32);
    static_assert(K::layout_id == 0x474c4d4b56533033ULL);
    require(std::string(K::numerical_identity) ==
            "glm-dsa-kvshard-rank01-fp32-v3-declaredexp-owned-tree256-fma-rankmerge-FAST", "v3 arithmetic identity");
    std::vector<unsigned char> bytes(sizeof(key) + 32, 0);
    std::memcpy(bytes.data(), key.data(), sizeof(key));
    K::validate_snapshot(bytes.data(), bytes.size(), true, key);
    for (const std::uint64_t legacy_layout : {0x474c4d4b56533031ULL, 0x474c4d4b56533032ULL}) {
        auto legacy_key = key;
        legacy_key[1] = legacy_layout;
        auto legacy_bytes = bytes;
        std::memcpy(legacy_bytes.data(), legacy_key.data(), sizeof(legacy_key));
        rejects([&] { K::validate_snapshot(legacy_bytes.data(), legacy_bytes.size(), true, key); });
    }
    rejects([&] { K::validate_snapshot(bytes.data(), bytes.size(), false); });
    for (std::size_t word = 0; word < key.size(); ++word) {
        auto bad = key;
        bad[word] ^= 1;
        rejects([&] { K::validate_snapshot(bytes.data(), bytes.size(), true, bad); });
    }
    rejects([&] { K::validate_snapshot(bytes.data(), bytes.size() - 1, true, key); });
    rejects([&] { K::validate_snapshot(bytes.data(), sizeof(key) - 1, true, key); });
    std::array<unsigned char, 64> old{};
    K::validate_snapshot(old.data(), old.size(), false);
    rejects([&] { K::validate_snapshot(old.data(), old.size(), true, key); });
    require(K::owned_prefix(8195, 0) == 4099 && K::owned_prefix(8195, 1) == 4096, "snapshot tail frontier");
}

void partials() {
    // D512 absorbed values; rank ordering is tested independently of value projection.
    std::vector<O::Row> rank0, rank1;
    for (std::uint64_t token = 0; token < 19; ++token) {
        O::Row row{};
        row.id = token;
        row.score = static_cast<float>(static_cast<int>(token % 7) - 3) * 0.25F;
        row.value.resize(512);
        for (std::size_t dim = 0; dim < row.value.size(); ++dim)
            row.value[dim] = static_cast<float>(token + dim % 3) * 0.125F;
        (K::owner(token) == 0 ? rank0 : rank1).push_back(std::move(row));
    }
    const auto a = O::accumulate(rank0, 512), b = O::accumulate(rank1, 512);
    const auto merged = O::merge_rank0_rank1(a, b);
    const auto normalized = O::normalize(merged);
    const float maximum = std::max(a.m, b.m), w0 = std::exp(a.m - maximum), w1 = std::exp(b.m - maximum);
    const float denominator = a.l * w0 + b.l * w1;
    for (std::size_t dim = 0; dim < 512; ++dim)
        require(normalized[dim] == (a.a[dim] * w0 + b.a[dim] * w1) / denominator, "rank01 FP32 formula");
    const auto empty = O::accumulate({}, 512);
    require(O::normalize(O::merge_rank0_rank1(empty, a)) == O::normalize(a), "empty rank");
    const auto zero = O::normalize(O::merge_rank0_rank1(empty, empty));
    require(std::all_of(zero.begin(), zero.end(), [](float value) { return value == 0; }), "both-empty NaN");
}
void dump_format() {
    const auto root = std::filesystem::temp_directory_path() /
        ("strix-glm-dump-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const std::array<std::uint32_t, 16> bits{{0U, 0x80000000U, 0x3f800000U, 0xc0000000U,
        0x7f800000U, 0xff800000U, 0x7fc01234U, 0x00000001U,
        0x3f800001U, 0x3f800002U, 0x3f800003U, 0x3f800004U,
        0x3f800005U, 0x3f800006U, 0x3f800007U, 0x3f800008U}};
    std::array<float, 16> values{};
    std::memcpy(values.data(), bits.data(), sizeof(values));
    const std::vector<std::string> expected_hex{"00000000", "80000000", "3f800000", "c0000000",
        "7f800000", "ff800000", "7fc01234", "00000001", "3f800001", "3f800002", "3f800003",
        "3f800004", "3f800005", "3f800006", "3f800007", "3f800008"};
    for (unsigned rank = 0; rank < 2; ++rank) {
        K::write_decode_dump(root.string(), rank, 2, 7, 8192, {0, 4, 9}, {0xbf800000U, 0x3fffffffU});
        K::write_attention_dump(root.string(), rank, 2, 7, 8192, values);
        const auto dir = root / ("rank-" + std::to_string(rank));
        const std::string stem = "epoch-2-layer-7-pos-8192-";
        const auto read = [&](const char* suffix) {
            std::ifstream file(dir / (stem + suffix));
            require(file.is_open(), "dump filename");
            std::vector<std::string> lines;
            for (std::string line; std::getline(file, line);) lines.push_back(line);
            require(file.eof(), "dump read error");
            return lines;
        };
        const std::string identity = std::string("# ") + K::numerical_identity;
        require(read("selected.csv") == std::vector<std::string>{identity,
            "order,global_id,owner,local_token", "0,0,0,0", "1,4,1,0", "2,9,0,5"}, "selected format/order");
        const auto pools = read("pools.csv");
        require(pools.size() == 4 && pools[0] == identity &&
            pools[1] == "global_pool,owner,local_pool,key_hex,score_hexfloat", "pool header/count");
        require(pools[2].find("0,0,0,bf800000,") == 0 && pools[3].find("1,1,0,3fffffff,") == 0,
                "pool key/ownership format");
        for (std::size_t i = 2; i < pools.size(); ++i) {
            const auto score = pools[i].substr(pools[i].rfind(',') + 1);
            char* end = nullptr;
            const float value = std::strtof(score.c_str(), &end);
            require(end != score.c_str() && *end == '\0' && value == (i == 2 ? 1.0F : -2.0F), "pool score decode");
        }
        const auto output = read("attention.csv");
        require(output.size() == 18 && output[0] == identity && output[1] == "index,fp32_hex", "output header/count");
        for (std::size_t i = 0; i < expected_hex.size(); ++i)
            require(output[i + 2] == std::to_string(i) + ',' + expected_hex[i], "output FP32 bits/order");
        for (const auto* suffix : {"selected.csv", "pools.csv", "attention.csv"})
            std::filesystem::remove(dir / (stem + suffix));
        std::filesystem::remove(dir);
    }
    std::filesystem::remove(root);
}
} // namespace
int main() {
    try {
        for (const auto capacity : {4ULL, 8ULL, 12ULL, 8192ULL, 8444ULL, 8448ULL, 65536ULL, 65540ULL}) mapping(capacity);
        selection(8192); selection(65536); selection(65539);
        snapshots(); partials(); dump_format();
        rejects([] { (void)K::storage_tokens(0); });
        rejects([] { (void)K::storage_tokens(7); });
        rejects([] { (void)K::storage_tokens(std::uint64_t{1} << 32); });
        rejects([] { (void)K::merge_score_keys({1}, {}, 0); });
        rejects([] { (void)K::merge_score_keys({1}, {1}, 2); });
        require(K::storage_tokens(65536) * 1664 * 11 == 65536ULL * 9152, "eleven-layer cache slope");
        std::printf("GLM KV shard host contracts: %u checks passed (device gates NOT RUN)\n", checks);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "GLM KV shard host contracts FAILED: %s\n", error.what());
        return 1;
    }
}
