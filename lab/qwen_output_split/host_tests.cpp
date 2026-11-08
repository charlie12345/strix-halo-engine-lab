#include "options.hpp"
#include "output_split.hpp"
#include "scalar_q8.hpp"
#include <bit>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>
using namespace strix::qwen_split;
namespace {
std::uint64_t checks = 0;
void check(bool ok) {
    ++checks;
    if (!ok) {
        throw std::runtime_error("host expectation failed");
    }
}
template <class F> void rejects(F f) {
    bool caught = false;
    try {
        f();
    } catch (const std::exception &) {
        caught = true;
    }
    check(caught);
}
Digest digest(unsigned seed) {
    Digest d{};
    for (std::size_t i = 0; i < d.size(); ++i) {
        d[i] = static_cast<std::uint8_t>(seed + static_cast<unsigned>(i));
    }
    return d;
}
Binding binding() {
    Binding b;
    b.source = digest(1);
    b.target = digest(2);
    b.asset = digest(3);
    b.tensor = digest(4);
    b.input = digest(5);
    b.reference = digest(6);
    b.environment = digest(7);
    b.binary = digest(8);
    b.epoch = 17;
    b.reserve = 4096;
    b.asset_bytes = 1ULL << 40U;
    b.tensor_offset = 4096;
    return b;
}
Shape shape() { return {256, 32, 4, 64, 7, 2}; }
struct Pair {
    Plan p0 = plan(shape(), 0), p1 = plan(shape(), 1);
    Binding b = binding();
    Gate a{p0, b}, z{p1, b};
    void exchange(Frame x, Frame y) {
        auto wx = encode(x), wy = encode(y);
        a.accept(wy);
        z.accept(wx);
    }
    void offer() {
        const auto n = add(p0.declared_peak, b.reserve);
        exchange(a.offer(n), z.offer(n));
    }
    void reference() { exchange(a.reference(digest(21)), z.reference(digest(21))); }
    void row(std::uint64_t n) {
        a.begin_payload(n);
        z.begin_payload(n);
        a.finish_payload(n);
        z.finish_payload(n);
        exchange(a.row(digest(static_cast<unsigned>(n + 30))),
                 z.row(digest(static_cast<unsigned>(n + 30))));
    }
};
void complete() {
    Pair x;
    check(!x.a.publishable());
    x.offer();
    x.reference();
    for (std::uint64_t i = 1; i <= x.p0.shape.inputs; ++i) {
        x.row(i);
        check(!x.a.publishable() && !x.z.publishable());
    }
    x.exchange(x.a.commit(digest(90)), x.z.commit(digest(90)));
    check(x.a.publishable() && x.z.publishable());
    rejects([&] { x.a.commit(digest(90)); });
    check(x.a.failed() && !x.a.publishable());
}
void layout() {
    auto p = plan({}, 0);
    check(p.full_weights == 675430400 && p.weights == 337715200 && p.wire_bytes == 496640);
    for (auto width : {32ULL, 256ULL, 320ULL, 2560ULL, 10240ULL}) {
        for (auto outputs : {16ULL, 32ULL, 248320ULL}) {
            for (auto inputs : {1ULL, 4ULL, 8ULL, 32ULL}) {
                for (std::uint32_t rank = 0; rank < 2; ++rank) {
                    const auto q = plan({width, outputs, inputs, 1024, 19, 2}, rank);
                    check(q.first == rank * q.count);
                    check(q.first + q.count <= outputs && q.weights * 2 == q.full_weights &&
                          q.input_bytes == width * inputs * 2);
                    check(q.declared_peak > q.full_weights &&
                          q.all_logits_bytes == outputs * inputs * 4);
                    const auto c = census(q);
                    check(c.controls == inputs + 3 && c.payloads == inputs &&
                          c.total_calls == 2 * inputs + 3);
                    check(c.payload_bytes == q.wire_bytes * inputs + 512 * (inputs + 3) &&
                          c.trailer_bytes == 128 * (2 * inputs + 3));
                }
            }
        }
    }
    for (auto mutate : std::vector<std::function<void(Shape &)>>{
             [](Shape &s) { s.width = 0; }, [](Shape &s) { s.width = 33; },
             [](Shape &s) { s.width = 10272; }, [](Shape &s) { s.outputs = 0; },
             [](Shape &s) { s.outputs = 24; }, [](Shape &s) { s.outputs = 1ULL << 32U; },
             [](Shape &s) { s.inputs = 0; }, [](Shape &s) { s.inputs = 33; },
             [](Shape &s) { s.capacity = 0; }, [](Shape &s) { s.position = s.capacity; },
             [](Shape &s) { s.capacity = s.position + 3; }, [](Shape &s) { s.world = 1; }}) {
        auto s = shape();
        mutate(s);
        rejects([&] { (void)plan(s, 0); });
    }
    rejects([] { (void)plan(shape(), 2); });
    rejects([] { (void)add(UINT64_MAX, 1); });
    rejects([] { (void)mul(UINT64_MAX, 2); });
    rejects([] {
        auto b = binding();
        b.reserve = UINT64_MAX;
        Gate g(plan(shape(), 0), b);
    });
    rejects([] {
        auto b = binding();
        b.tensor_offset = UINT64_MAX - 1;
        b.asset_bytes = UINT64_MAX;
        Gate g(plan(shape(), 0), b);
    });
    rejects([] {
        auto bad = plan(shape(), 0);
        ++bad.first;
        Gate g(bad, binding());
    });
}
void gather_cases() {
    for (std::uint32_t rank = 0; rank < 2; ++rank) {
        const auto p = plan({32, 32, 1, 1, 0, 2}, rank);
        std::vector<float> a(16), b(16), out(32);
        for (std::uint32_t i = 0; i < 16; ++i) {
            a[i] = std::bit_cast<float>((rank * 16 + i) * 7919U);
            b[i] = std::bit_cast<float>(((1U - rank) * 16 + i) * 7919U);
        }
        gather(p, a, b, out);
        for (std::uint32_t i = 0; i < 32; ++i) {
            check(std::bit_cast<std::uint32_t>(out[i]) == i * 7919U);
        }
        rejects([&] { gather(p, std::span(a).first(15), b, out); });
        rejects([&] { gather(p, a, a, out); });
        rejects([&] { gather(p, std::span(out).first(16), b, out); });
    }
    rejects([] { disjoint(0, 1, 100, 1); });
    rejects([] { disjoint(UINTPTR_MAX - 1, 4, 100, 1); });
    rejects([] { disjoint(100, 10, 109, 10); });
    disjoint(100, 10, 110, 10);
    check(true);
    const std::array<float, 4> row{-0.0F, 0.0F, 2.0F, 2.0F};
    check(firstmax(row) == 2);
    auto bad = row;
    bad[3] = std::numeric_limits<float>::infinity();
    rejects([&] { (void)firstmax(bad); });
    bad[3] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { (void)firstmax(bad); });
    rejects([] { (void)firstmax({}); });
}
void binding_cases() {
    for (auto field :
         {&Binding::source, &Binding::target, &Binding::asset, &Binding::tensor, &Binding::input,
          &Binding::reference, &Binding::environment, &Binding::binary}) {
        auto b = binding();
        b.*field = {};
        rejects([&] { Gate g(plan(shape(), 0), b); });
    }
    for (unsigned which = 0; which < 4; ++which) {
        auto b = binding();
        if (which == 0) {
            b.epoch = 0;
        }
        if (which == 1) {
            b.reserve = 0;
        }
        if (which == 2) {
            b.asset_bytes = 0;
        }
        if (which == 3) {
            b.scalar_leaf = 0;
        }
        rejects([&] { Gate g(plan(shape(), 0), b); });
    }
    {
        Pair x;
        const auto need = add(x.p0.declared_peak, x.b.reserve);
        auto a = x.a.offer(need - 1), b = x.z.offer(need);
        (void)a;
        rejects([&] { x.a.accept(encode(b)); });
        check(!x.a.publishable());
    }
    {
        Pair x;
        x.offer();
        x.reference();
        x.a.begin_payload(1);
        x.a.finish_payload(1);
        rejects([&] { x.a.finish_payload(1); });
        check(x.a.failed());
    }
    {
        Pair x;
        x.offer();
        x.reference();
        rejects([&] { x.a.commit(digest(8)); });
        check(x.a.failed());
    }
}
void wire_cases() {
    Pair x;
    auto frame = x.z.offer(add(x.p1.declared_peak, x.b.reserve));
    const auto wire = encode(frame);
    check(encode(decode(wire)) == wire);
    for (std::size_t n = 0; n < frame_bytes; ++n) {
        rejects([&] { (void)decode(std::span(wire).first(n)); });
    }
    std::vector<std::byte> oversized(frame_bytes + 1);
    rejects([&] { (void)decode(oversized); });
    for (std::size_t n = 480; n < frame_bytes; ++n) {
        auto w = wire;
        w[n] = std::byte{1};
        rejects([&] { (void)decode(w); });
    }
    for (std::size_t word :
         {0U, 1U, 2U, 3U, 4U, 7U, 8U, 9U, 11U, 12U, 13U, 14U, 15U, 16U, 19U, 20U, 23U}) {
        auto w = wire;
        for (std::size_t k = 0; k < 8; ++k) {
            w[word * 8 + k] = std::byte{255};
        }
        rejects([&] { (void)decode(w); });
    }
    for (auto bad :
         {std::string(63, 'a'), std::string(64, 'G'), std::string(64, 'A'), std::string(64, '0')}) {
        rejects([&] { (void)hex_digest(bad); });
    }
    check(present(hex_digest(std::string(64, 'a'))));
}
void mismatch_cases() {
    for (auto mutate : std::vector<std::function<void(Frame &)>>{
             [](Frame &f) { ++f.binding.epoch; },
             [](Frame &f) { ++f.binding.reserve; },
             [](Frame &f) { ++f.binding.asset_bytes; },
             [](Frame &f) { ++f.binding.tensor_offset; },
             [](Frame &f) { f.binding.scalar_leaf = 2; },
             [](Frame &f) { f.binding.source[0] ^= 1; },
             [](Frame &f) { f.binding.target[0] ^= 1; },
             [](Frame &f) { f.binding.asset[0] ^= 1; },
             [](Frame &f) { f.binding.tensor[0] ^= 1; },
             [](Frame &f) { f.binding.input[0] ^= 1; },
             [](Frame &f) { f.binding.reference[0] ^= 1; },
             [](Frame &f) { f.binding.environment[0] ^= 1; },
             [](Frame &f) { f.binding.binary[0] ^= 1; },
             [](Frame &f) {
                 auto s = f.layout.shape;
                 ++s.capacity;
                 f.layout = plan(s, 1);
             },
             [](Frame &f) {
                 auto s = f.layout.shape;
                 s.width += 32;
                 f.layout = plan(s, 1);
             },
             [](Frame &f) {
                 auto s = f.layout.shape;
                 s.outputs += 16;
                 f.layout = plan(s, 1);
             },
             [](Frame &f) {
                 auto s = f.layout.shape;
                 ++s.position;
                 f.layout = plan(s, 1);
             },
             [](Frame &f) {
                 auto s = f.layout.shape;
                 --s.inputs;
                 f.layout = plan(s, 1);
             },
             [](Frame &f) { f.sequence = 1; },
             [](Frame &f) { f.phase = Phase::reference; },
             [](Frame &f) { f.layout = plan(f.layout.shape, 0); },
             [](Frame &f) { ++f.layout.first; },
             [](Frame &f) { ++f.layout.wire_bytes; },
             [](Frame &f) {
                 f.ready = 0;
                 f.reason = 7;
             },
             [](Frame &f) { --f.available; }}) {
        Pair x;
        const auto need = add(x.p0.declared_peak, x.b.reserve);
        (void)x.a.offer(need);
        auto peer = x.z.offer(need);
        mutate(peer);
        rejects([&] { x.a.accept(encode(peer)); });
        check(x.a.failed() && !x.a.publishable());
        rejects([&] { x.a.offer(need); });
    }
    {
        Pair x;
        const auto n = add(x.p0.declared_peak, x.b.reserve);
        auto f = x.a.offer(n, false, 9);
        auto g = x.z.offer(n);
        rejects([&] { x.a.accept(encode(g)); });
        rejects([&] { x.z.accept(encode(f)); });
        check(!x.a.publishable() && !x.z.publishable());
    }
    {
        Pair x;
        rejects([&] { x.a.accept(encode(x.z.offer(UINT64_MAX))); });
    }
    {
        Pair x;
        x.offer();
        rejects([&] { x.a.row(digest(1)); });
        check(x.a.failed());
    }
    {
        Pair x;
        x.offer();
        rejects([&] { x.a.reference({}); });
        check(x.a.failed());
    }
    {
        Pair x;
        x.offer();
        auto a = x.a.reference(digest(10)), b = x.z.reference(digest(11));
        (void)a;
        rejects([&] { x.a.accept(encode(b)); });
        check(x.a.failed());
    }
    for (std::uint64_t row = 1; row <= 4; ++row) {
        Pair x;
        x.offer();
        x.reference();
        for (std::uint64_t i = 1; i < row; ++i) {
            x.row(i);
        }
        x.a.begin_payload(row);
        x.z.begin_payload(row);
        x.a.finish_payload(row);
        x.z.finish_payload(row);
        (void)x.a.row(digest(10));
        auto peer = x.z.row(digest(10));
        --peer.sequence;
        rejects([&] { x.a.accept(encode(peer)); });
        check(x.a.failed() && !x.a.publishable());
    }
    {
        Pair x;
        x.offer();
        x.reference();
        rejects([&] { x.a.begin_payload(2); });
        check(x.a.failed());
    }
    {
        Pair x;
        x.offer();
        x.reference();
        rejects([&] { x.a.finish_payload(1); });
        check(x.a.failed());
    }
    {
        Pair x;
        x.offer();
        x.reference();
        for (std::uint64_t i = 1; i <= 4; ++i) {
            x.row(i);
        }
        (void)x.a.commit(digest(9));
        auto peer = x.z.commit(digest(9));
        --peer.sequence;
        rejects([&] { x.a.accept(encode(peer)); });
        check(!x.a.publishable());
    }
    // Simulate transport failure at every control/payload boundary. No output
    // remains publishable, and no retry can reuse the same gate.
    for (unsigned fail_at = 0; fail_at < 8; ++fail_at) {
        Pair x;
        if (fail_at > 0) {
            x.offer();
        }
        if (fail_at > 1) {
            x.reference();
        }
        for (unsigned i = 1; i <= 4 && i + 1 < fail_at; ++i) {
            x.row(i);
        }
        x.a.fail();
        x.z.fail();
        check(!x.a.publishable() && !x.z.publishable());
        rejects([&] { x.a.offer(UINT64_MAX); });
    }
}
void option_cases() {
    std::string framed_one, framed_two;
    append_framed(framed_one, "STRIX_X", "foo\nSTRIX_Y=bar");
    append_framed(framed_two, "STRIX_X", "foo");
    append_framed(framed_two, "STRIX_Y", "bar");
    check(framed_one != framed_two);
    std::string c;
    append_framed(c, "STRIX_X", "foo\nSTRIX_Y=bar");
    check(framed_one == c);
    auto parse = [](std::vector<std::string> words) {
        std::vector<char *> argv;
        for (auto &w : words) {
            argv.push_back(w.data());
        }
        return Options(static_cast<int>(argv.size()), argv.data());
    };
    rejects([&] { (void)parse({"fixture"}); });
    rejects([&] {
        (void)parse({"fixture", "--enable-output-row-split", "--enable-output-row-split"});
    });
    for (const char *name : {"--mtp", "--draft", "--mtp-sidecar", "--prefix-cache",
                             "--session-cache", "--prefill-chunk", "--batch", "--http-batch"}) {
        rejects([&] { (void)parse({"fixture", "--enable-output-row-split", name, "off"}); });
    }
    rejects([&] { (void)parse({"fixture", "--enable-output-row-split", "--unknown", "1"}); });
    rejects([&] { (void)parse({"fixture", "--enable-output-row-split", "--rank"}); });
    rejects([&] {
        (void)parse({"fixture", "--enable-output-row-split", "--rank", "0", "--rank", "1"});
    });
    const auto o = parse({"fixture", "--enable-output-row-split", "--rank", "1"});
    check(o.number("--rank") == 1);
    rejects([&] { (void)o.get("--model"); });
    for (const char *number : {"", "-1", "+1", " 1", "1 ", "1x", "18446744073709551616"}) {
        rejects([&] { (void)integer(number); });
    }
    check(integer("18446744073709551615") == UINT64_MAX);
    for (std::uint32_t rank = 0; rank < 3; ++rank) {
        auto p = plan({32, 32, 1, 1, 0, 2}, 0);
        p.rank = rank;
        p.first = 1;
        std::vector<float> a(16), b(16), full(32);
        rejects([&] { gather(p, a, b, full); });
    }
}
void scalar_cases() {
    for (unsigned raw = 0; raw < 65536; ++raw) {
        const auto h = static_cast<std::uint16_t>(raw);
        const unsigned e = (raw >> 10U) & 31U, m = raw & 1023U;
        const float observed = fp16(h);
        if (e == 31) {
            check(m ? std::isnan(observed) : std::isinf(observed));
        } else {
            float expected = e ? std::ldexp(static_cast<float>(1024U + m), static_cast<int>(e) - 25)
                               : std::ldexp(static_cast<float>(m), -24);
            if (raw & 32768U) {
                expected = -expected;
            }
            check(std::bit_cast<std::uint32_t>(observed) == std::bit_cast<std::uint32_t>(expected));
        }
    }
    for (auto width : {32U, 256U, 320U, 2560U, 10240U}) {
        std::vector<std::uint16_t> input(width, 0x3f80U);
        std::vector<std::byte> weights(width / 32 * 34, std::byte{1});
        for (std::size_t b = 0; b < weights.size(); b += 34) {
            weights[b] = std::byte{0};
            weights[b + 1] = std::byte{0x3c};
        }
        for (auto mode : {1ULL, 2ULL}) {
            check(scalar_q8(input, weights, mode) == static_cast<float>(width));
        }
        rejects([&] { (void)scalar_q8(input, std::span(weights).first(weights.size() - 1), 1); });
        input[0] = 0x7f80U;
        rejects([&] { (void)scalar_q8(input, weights, 1); });
        input[0] = 0x3f80U;
        weights[1] = std::byte{0x7c};
        rejects([&] { (void)scalar_q8(input, weights, 1); });
    }
}
} // namespace
int main() {
    try {
        layout();
        gather_cases();
        binding_cases();
        wire_cases();
        complete();
        mismatch_cases();
        option_cases();
        scalar_cases();
        std::printf("PASS %llu actual planner/control/scalar checks (NDEBUG=%d)\n",
                    static_cast<unsigned long long>(checks),
#ifdef NDEBUG
                    1
#else
                    0
#endif
        );
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL after %llu checks: %s\n",
                     static_cast<unsigned long long>(checks), e.what());
        return 1;
    }
}
