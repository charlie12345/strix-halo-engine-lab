#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strix::minimax::pfx {
inline constexpr std::uint64_t kHeaderMagic = 0x3148584650584d4dULL;   // "MMXPFXH1"
inline constexpr std::uint64_t kTrailerMagic = 0x3154584650584d4dULL;  // "MMXPFXT1"

// Identity of one exchange; the header and the trailer both carry it and both ranks must agree on every field
// except `rank`.
struct Header {
    std::uint64_t magic = kHeaderMagic;
    std::uint64_t epoch = 0;
    std::uint64_t plan = 0;
    std::uint64_t sequence = 0;
    std::uint64_t request = 0;   // the engine's call counter (prefill chunk group)
    std::uint64_t bytes = 0;
    std::uint32_t kind = 0;      // 1 = attention sum, 2 = FFN sum
    std::uint32_t layer = 0;
    std::uint32_t chunk = 0;     // wire chunk size | bit 31 = CRC trailer on (both ranks must agree)
    std::uint32_t rank = 0;
};
static_assert(sizeof(Header) == 64);

struct Tag {
    std::uint32_t kind = 0;
    std::uint32_t layer = 0;
    std::uint64_t request = 0;
};

// MiniMax prefill exactness is row-dependent: preserve complete FP32 rows,
// the caller's row batch, and rank-0-then-rank-1 reduction at the consumer.
struct PrefillTraits {
    static constexpr std::size_t payload_bytes(std::size_t rows, std::size_t width) {
        return rows * width * sizeof(float);
    }
    template<class OptionsType>
    static Header header(const OptionsType& options, std::uint64_t sequence,
                         std::size_t bytes, const Tag& tag) {
        Header result;
        result.epoch = options.epoch;
        result.plan = options.plan;
        result.sequence = sequence;
        result.request = tag.request;
        result.bytes = bytes;
        result.kind = tag.kind;
        result.layer = tag.layer;
        result.chunk = static_cast<std::uint32_t>(options.chunk) | (options.crc ? 0x80000000U : 0U);
        result.rank = options.rank;
        return result;
    }
    static void trailer(std::byte* destination, Header identity,
                        const std::uint32_t* crcs, std::size_t count) {
        identity.magic = kTrailerMagic;
        std::memcpy(destination, &identity, sizeof(identity));
        std::memcpy(destination + sizeof(Header), crcs, count * 4);
    }
};
} // namespace strix::minimax::pfx
