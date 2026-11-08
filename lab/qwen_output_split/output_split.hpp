#pragma once
// Complete output rows only. No K split, floating-point reduction or model-state
// mutation is implemented here. This contract is shared by the native fixture
// and the release-mode host negatives.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::qwen_split {
inline void require(bool ok, const char *why) {
    if (!ok) {
        throw std::invalid_argument(why);
    }
}
inline std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    require(b <= UINT64_MAX - a, "split byte addition overflow");
    return a + b;
}
inline std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    require(!a || b <= UINT64_MAX / a, "split byte multiplication overflow");
    return a * b;
}
inline std::size_t extent(std::uint64_t n) {
    require(n <= SIZE_MAX, "split extent exceeds address space");
    return static_cast<std::size_t>(n);
}
inline void append_framed(std::string &out, std::string_view name, std::string_view value) {
    out += std::to_string(name.size()) + ":";
    out.append(name);
    out += std::to_string(value.size()) + ":";
    out.append(value);
}
using Digest = std::array<std::uint8_t, 32>;
inline bool present(const Digest &d) {
    return std::any_of(d.begin(), d.end(), [](auto x) { return x != 0; });
}
inline Digest hex_digest(std::string_view s) {
    require(s.size() == 64, "split digest must be 64 lowercase hex digits");
    Digest d{};
    auto nib = [](char c) -> unsigned {
        require((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'),
                "split digest is not lowercase hex");
        return c <= '9' ? static_cast<unsigned>(c - '0') : static_cast<unsigned>(c - 'a' + 10);
    };
    for (std::size_t i = 0; i < d.size(); ++i) {
        d[i] = static_cast<std::uint8_t>((nib(s[2 * i]) << 4U) | nib(s[2 * i + 1]));
    }
    require(present(d), "split digest must be nonzero");
    return d;
}
struct Shape {
    std::uint64_t width = 2560, outputs = 248320, inputs = 1, capacity = 1, position = 0, world = 2;
    friend bool operator==(const Shape &, const Shape &) = default;
};
struct Plan {
    Shape shape;
    std::uint32_t rank = 0;
    std::uint64_t first = 0, count = 0, row_bytes = 0, weights = 0, full_weights = 0;
    std::uint64_t input_bytes = 0, wire_bytes = 0, full_row_bytes = 0, all_logits_bytes = 0,
                  declared_peak = 0;
    friend bool operator==(const Plan &, const Plan &) = default;
};
inline Plan plan(Shape s, std::uint32_t rank) {
    require(s.world == 2 && rank < 2, "split requires world2 and rank0/1");
    require(s.width && s.width <= 10240 && s.width % 32 == 0, "unsupported packed Q8 input width");
    // Keep both shards on complete wave8 row groups, with the same global row
    // alignment as the full production launch. No implicit padding/tail policy.
    require(s.outputs && s.outputs <= UINT32_MAX && s.outputs % 16 == 0,
            "invalid complete output-row shards");
    require(s.inputs && s.inputs <= 32 && s.capacity && s.position < s.capacity &&
                s.inputs <= s.capacity - s.position,
            "split input sequence exceeds fixed capacity");
    Plan p;
    p.shape = s;
    p.rank = rank;
    p.count = s.outputs / 2;
    p.first = mul(rank, p.count);
    p.row_bytes = mul(s.width / 32, 34);
    p.full_weights = mul(s.outputs, p.row_bytes);
    p.weights = mul(p.count, p.row_bytes);
    p.input_bytes = mul(mul(s.inputs, s.width), 2);
    p.wire_bytes = mul(p.count, 4);
    p.full_row_bytes = mul(s.outputs, 4);
    p.all_logits_bytes = mul(s.inputs, p.full_row_bytes);
    // Conservative declared storage: full host+device oracle weights, input
    // host+device, reference/candidate/scratch rows, eight communication buffers.
    // Allocator/driver/host-object overhead is covered by a separate reserve;
    // this is neither RSS accounting nor a general UMA admission oracle.
    p.declared_peak = add(add(mul(2, p.full_weights), mul(2, p.input_bytes)),
                          add(mul(4, p.all_logits_bytes), mul(8, add(p.wire_bytes, 128))));
    (void)extent(p.declared_peak);
    return p;
}
inline void disjoint(std::uintptr_t a, std::uint64_t an, std::uintptr_t b, std::uint64_t bn) {
    require(a && b && an && bn && an <= UINTPTR_MAX - a && bn <= UINTPTR_MAX - b,
            "split null/overflowing live view");
    require(!(a < b + bn && b < a + an), "split live views alias");
}
inline void gather(const Plan &p, std::span<const float> local, std::span<const float> peer,
                   std::span<float> full) {
    require(p == plan(p.shape, p.rank), "split gather requires a canonical shard layout");
    require(local.size() == p.count && peer.size() == p.count && full.size() == p.shape.outputs,
            "split output extent mismatch/truncation");
    disjoint(reinterpret_cast<std::uintptr_t>(local.data()), p.wire_bytes,
             reinterpret_cast<std::uintptr_t>(peer.data()), p.wire_bytes);
    disjoint(reinterpret_cast<std::uintptr_t>(local.data()), p.wire_bytes,
             reinterpret_cast<std::uintptr_t>(full.data()), p.full_row_bytes);
    disjoint(reinterpret_cast<std::uintptr_t>(peer.data()), p.wire_bytes,
             reinterpret_cast<std::uintptr_t>(full.data()), p.full_row_bytes);
    std::memcpy(full.data() + extent(p.first), local.data(), extent(p.wire_bytes));
    std::memcpy(full.data() + extent((1U - p.rank) * p.count), peer.data(), extent(p.wire_bytes));
}
inline std::uint32_t firstmax(std::span<const float> row) {
    require(!row.empty() && row.size() <= UINT32_MAX, "split empty/oversized logit row");
    std::uint32_t best = 0;
    for (std::size_t i = 0; i < row.size(); ++i) {
        require(std::isfinite(row[i]), "split nonfinite raw logit");
        if (row[i] > row[best]) {
            best = static_cast<std::uint32_t>(i);
        }
    }
    return best;
}
struct Binding {
    Digest source{}, target{}, asset{}, tensor{}, input{}, reference{}, environment{}, binary{};
    std::uint64_t epoch = 0, reserve = 0, asset_bytes = 0, tensor_offset = 0, scalar_leaf = 1;
    friend bool operator==(const Binding &, const Binding &) = default;
};
inline void validate_binding(const Binding &b) {
    require(b.epoch && b.reserve && b.asset_bytes && b.tensor_offset < b.asset_bytes,
            "split incomplete epoch/reserve/asset binding");
    require(b.scalar_leaf == 1 || b.scalar_leaf == 2,
            "split scalar leaf must be declared FMA or separate");
    for (const auto *d : {&b.source, &b.target, &b.asset, &b.tensor, &b.input, &b.reference,
                          &b.environment, &b.binary}) {
        require(present(*d), "split missing source/target/asset/environment binding");
    }
}
enum class Phase : std::uint64_t { offer = 1, reference = 2, row = 3, commit = 4 };
struct Frame {
    Plan layout;
    Binding binding;
    Phase phase = Phase::offer;
    std::uint64_t sequence = 0, available = 0, ready = 0, reason = 0;
    Digest outcome{};
};
inline constexpr std::size_t frame_bytes = 512;
struct Census {
    std::uint64_t controls = 0, payloads = 0, total_calls = 0, payload_bytes = 0, trailer_bytes = 0;
};
inline Census census(const Plan &p) {
    require(p == plan(p.shape, p.rank), "split census requires canonical layout");
    Census c;
    c.controls = add(p.shape.inputs, 3);
    c.payloads = p.shape.inputs;
    c.total_calls = add(c.controls, c.payloads);
    c.payload_bytes = add(mul(p.wire_bytes, c.payloads), mul(frame_bytes, c.controls));
    c.trailer_bytes = mul(128, c.total_calls);
    return c;
}
using Wire = std::array<std::byte, frame_bytes>;
inline Wire encode(const Frame &f) {
    Wire out{};
    std::size_t at = 0;
    auto put = [&](std::uint64_t n) {
        for (unsigned i = 0; i < 8; ++i) {
            out[at++] = std::byte((n >> (8U * i)) & 255U);
        }
    };
    put(0x315450554f5751ULL);
    put(1);
    put(static_cast<std::uint64_t>(f.phase));
    const auto &p = f.layout;
    const auto &s = p.shape;
    const auto &b = f.binding;
    for (auto n : {std::uint64_t{p.rank},
                   s.world,
                   b.epoch,
                   f.sequence,
                   s.width,
                   s.outputs,
                   s.inputs,
                   s.capacity,
                   s.position,
                   p.first,
                   p.count,
                   p.row_bytes,
                   p.wire_bytes,
                   p.declared_peak,
                   b.reserve,
                   f.available,
                   f.ready,
                   f.reason,
                   b.asset_bytes,
                   b.tensor_offset,
                   b.scalar_leaf}) {
        put(n);
    }
    for (const auto *d : {&b.source, &b.target, &b.asset, &b.tensor, &b.input, &b.reference,
                          &b.environment, &b.binary, &f.outcome}) {
        for (auto n : *d) {
            out[at++] = std::byte(n);
        }
    }
    require(at == 480 && at <= out.size(), "split internal wire layout changed");
    return out;
}
inline Frame decode(std::span<const std::byte> in) {
    require(in.size() == frame_bytes, "split control frame truncated/oversized");
    std::size_t at = 0;
    auto get = [&] {
        std::uint64_t n = 0;
        for (unsigned i = 0; i < 8; ++i) {
            n |= std::uint64_t(std::to_integer<unsigned>(in[at++])) << (8U * i);
        }
        return n;
    };
    require(get() == 0x315450554f5751ULL && get() == 1, "split control magic/version mismatch");
    const auto phase = get();
    require(phase >= 1 && phase <= 4, "split control phase invalid");
    Frame f;
    f.phase = static_cast<Phase>(phase);
    const auto rank = get();
    require(rank < 2, "split wire rank invalid");
    Shape s;
    s.world = get();
    f.binding.epoch = get();
    f.sequence = get();
    s.width = get();
    s.outputs = get();
    s.inputs = get();
    s.capacity = get();
    s.position = get();
    f.layout = plan(s, static_cast<std::uint32_t>(rank));
    for (auto expected : {f.layout.first, f.layout.count, f.layout.row_bytes, f.layout.wire_bytes,
                          f.layout.declared_peak}) {
        require(get() == expected, "split wire shard/byte layout mismatch");
    }
    auto &b = f.binding;
    b.reserve = get();
    f.available = get();
    f.ready = get();
    f.reason = get();
    b.asset_bytes = get();
    b.tensor_offset = get();
    b.scalar_leaf = get();
    require(f.ready <= 1 && (f.ready ? f.reason == 0 : f.reason != 0),
            "split control readiness/reason invalid");
    for (auto *d : {&b.source, &b.target, &b.asset, &b.tensor, &b.input, &b.reference,
                    &b.environment, &b.binary, &f.outcome}) {
        for (auto &n : *d) {
            n = static_cast<std::uint8_t>(std::to_integer<unsigned>(in[at++]));
        }
    }
    for (; at < in.size(); ++at) {
        require(in[at] == std::byte{}, "split control reserved bytes changed");
    }
    validate_binding(b);
    return f;
}
// Serialized, one-use admission/sequence gate. Completion here is provisional
// until both final control frames validate. A transport exception must call fail;
// neither a local payload completion nor an ACK alone permits publication.
class Gate {
  public:
    Gate(Plan p, Binding b) : plan_(p), binding_(b) {
        validate_binding(b);
        require(p == plan(p.shape, p.rank), "split noncanonical local plan");
        (void)add(p.declared_peak, b.reserve);
        require(add(b.tensor_offset, p.full_weights) <= b.asset_bytes,
                "split tensor exceeds bound model asset");
    }
    Frame offer(std::uint64_t available, bool ready = true, std::uint64_t reason = 0) {
        return prepare(Phase::offer, 0, {}, ready, reason, available);
    }
    Frame reference(Digest digest, bool ready = true, std::uint64_t reason = 0) {
        return prepare(Phase::reference, 0, digest, ready, reason, 0);
    }
    void begin_payload(std::uint64_t sequence) {
        checked([&] {
            require(next_ == Phase::row && !waiting_ && !payload_started_ && sequence == row_ + 1,
                    "split payload is out of order/stale");
            payload_started_ = true;
        });
    }
    void finish_payload(std::uint64_t sequence) {
        checked([&] {
            require(payload_started_ && !payload_done_ && sequence == row_ + 1,
                    "split payload completion is out of order/stale");
            payload_done_ = true;
        });
    }
    Frame row(Digest digest, bool ready = true, std::uint64_t reason = 0) {
        checked([&] {
            require(payload_done_, "split row validation before actual payload completion");
        });
        return prepare(Phase::row, row_ + 1, digest, ready, reason, 0);
    }
    Frame commit(Digest digest) {
        return prepare(Phase::commit, plan_.shape.inputs + 1, digest, true, 0, 0);
    }
    void accept(const Frame &peer) {
        checked([&] {
            require(waiting_ && !committed_, "split unexpected/duplicate peer control");
            require(peer.phase == local_.phase && peer.sequence == local_.sequence &&
                        peer.layout.rank == 1U - plan_.rank,
                    "split peer phase/rank/stale generation mismatch");
            require(peer.layout.shape == plan_.shape && peer.binding == binding_,
                    "split peer shape/identity/capacity mismatch");
            // Reconstruct the peer's layout; callers cannot bypass decode with a
            // forged same-shape object carrying modified first/count/bytes.
            require(encode(peer) == encode(decode(encode(peer))),
                    "split peer layout is not canonical");
            require(local_.ready && peer.ready,
                    "split both-rank admission/reference/result refused");
            if (next_ == Phase::offer) {
                const auto need = add(plan_.declared_peak, binding_.reserve);
                require(local_.available >= need && peer.available >= need,
                        "split both-rank memory admission failed");
                next_ = Phase::reference;
            } else {
                require(present(local_.outcome) && peer.outcome == local_.outcome,
                        "split complete native raw-row digest mismatch");
                if (next_ == Phase::reference) {
                    next_ = Phase::row;
                } else if (next_ == Phase::row) {
                    ++row_;
                    payload_started_ = false;
                    payload_done_ = false;
                    if (row_ == plan_.shape.inputs) {
                        next_ = Phase::commit;
                    }
                } else {
                    committed_ = true;
                }
            }
            waiting_ = false;
        });
    }
    void accept(std::span<const std::byte> wire) {
        checked([&] { accept(decode(wire)); });
    }
    void fail() noexcept {
        failed_ = true;
        committed_ = false;
    }
    [[nodiscard]] bool publishable() const noexcept { return committed_ && !failed_; }
    [[nodiscard]] bool failed() const noexcept { return failed_; }

  private:
    template <class F> void checked(F f) {
        require(!failed_, "split gate is poisoned");
        try {
            f();
        } catch (...) {
            fail();
            throw;
        }
    }
    Frame prepare(Phase phase, std::uint64_t sequence, Digest digest, bool ready,
                  std::uint64_t reason, std::uint64_t available) {
        checked([&] {
            require(!waiting_ && !committed_ && next_ == phase,
                    "split local phase/duplicate control invalid");
            require(ready ? reason == 0 : reason != 0, "split local readiness/reason invalid");
            require(phase == Phase::offer || present(digest),
                    "split missing complete result digest");
            local_ = {plan_,  binding_, phase, sequence, available, ready ? 1ULL : 0ULL,
                      reason, digest};
            waiting_ = true;
        });
        return local_;
    }
    Plan plan_;
    Binding binding_;
    Frame local_;
    Phase next_ = Phase::offer;
    std::uint64_t row_ = 0;
    bool waiting_ = false, payload_started_ = false, payload_done_ = false, failed_ = false,
         committed_ = false;
};
} // namespace strix::qwen_split
