// Existing executable: A=unchanged wide-rows output14, B=experimental output15.
// The producer, projection, copy policy and complete transaction boundaries agree.
using ReuseBytes = std::vector<std::vector<unsigned char>>;
ReuseBytes reuse_bytes(const RecomputeFixture& f) {
    ReuseBytes out{f.f.normalized.download(), f.f.injection.download(), f.f.output.download(),
        f.hyper.download(), f.streams.download(), f.hidden.download(), f.routed.download(),
        f.combined.download(), f.inverse.download(), f.gates.download(), f.norm.download(),
        f.f.down.download(), f.f.inject.download(), f.f.up.download()};
    std::vector<unsigned char> low16(static_cast<std::size_t>(f.f.rows) * kRank * 2U);
    check(hipMemcpy(low16.data(), strix::testing::wide_hyper_rows_test_low16(), low16.size(), hipMemcpyDeviceToHost), "reuse low16 read");
    out.push_back(std::move(low16));
    if (f.f.rows < 16384U) {
        std::vector<unsigned char> tail(64U);
        const auto* low = static_cast<const unsigned char*>(strix::testing::wide_hyper_rows_test_low16());
        check(hipMemcpy(tail.data(), low + static_cast<std::size_t>(f.f.rows) * kRank * 2U, tail.size(), hipMemcpyDeviceToHost), "reuse low16 tail");
        out.push_back(std::move(tail));
    }
    return out;
}
void reuse_canaries(const RecomputeFixture& f, const ReuseBytes& bytes, bool on) {
    const auto poison = [](const std::vector<unsigned char>& b, std::size_t first, std::size_t end) {
        if (!std::all_of(b.begin() + static_cast<std::ptrdiff_t>(first), b.begin() + static_cast<std::ptrdiff_t>(end),
                [](unsigned char v) { return v == 0xa5; })) throw std::runtime_error("reuse canary overwritten");
    };
    poison(bytes[8], static_cast<std::size_t>(f.f.rows) * 16U, bytes[8].size());
    if (f.f.rows < 16384U) poison(bytes[15], 0U, bytes[15].size());
    for (unsigned r = 0; r < f.f.rows; ++r) {
        const auto n = static_cast<std::size_t>(r) * kNormalizedStride;
        if (on && std::memcmp(bytes[0].data() + n, f.f.normalized_host.data() + n, kWidth * 4U))
            throw std::runtime_error("reuse wrote normalized image");
        poison(bytes[0], n + (kWidth + kRank) * 4U, n + kNormalizedStride);
        poison(bytes[1], static_cast<std::size_t>(r) * kInjectionStride + 16U, static_cast<std::size_t>(r + 1U) * kInjectionStride);
        poison(bytes[2], static_cast<std::size_t>(r) * f.f.output_stride + kHidden * (f.f.bf16 ? 2U : 4U),
            static_cast<std::size_t>(r + 1U) * f.f.output_stride);
        const auto same_pad = [&](unsigned b, const std::vector<std::uint16_t>& src, std::size_t stride, std::size_t width) {
            if (std::memcmp(bytes[b].data() + r * stride + width,
                    reinterpret_cast<const unsigned char*>(src.data()) + r * stride + width, stride - width))
                throw std::runtime_error("reuse stream/hidden padding overwritten");
        };
        same_pad(3U, f.hyper_host, RecomputeFixture::ss, kWidth * 2U);
        same_pad(4U, f.hyper_host, RecomputeFixture::ss, kWidth * 2U);
        same_pad(5U, f.hidden_host, RecomputeFixture::hs, kHidden * 2U);
        same_pad(6U, f.routed_host, RecomputeFixture::hs, kHidden * 2U);
        if (!(f.in_place && f.kind == 2U))
            poison(bytes[7], r * RecomputeFixture::hs + kHidden * 2U, (r + 1U) * RecomputeFixture::hs);
    }
}
void compare_inverse_reuse(RecomputeFixture& f, hipStream_t stream, bool on = true, const std::vector<float>* inverses = nullptr) {
    // Allocate the pre-existing scratch through the unchanged A transaction.
    f.reset(); f.enqueue(on, 14U, 0U, stream, inverses);
    check(hipStreamSynchronize(stream), "reuse scratch priming");
    f.reset();
    const auto immutable = std::array{f.gates.download(), f.norm.download(), f.f.down.download(), f.f.inject.download(), f.f.up.download()};
    const auto run = [&](unsigned form) {
        f.reset();
        if (f.f.rows < 16384U) {
            auto* low = const_cast<unsigned char*>(static_cast<const unsigned char*>(strix::testing::wide_hyper_rows_test_low16()));
            check(hipMemsetAsync(low + static_cast<std::size_t>(f.f.rows) * kRank * 2U, 0xa5, 64U, stream), "reuse low16 guard poison");
        }
        f.enqueue(on, form, 0U, stream, inverses);
        check(hipStreamSynchronize(stream), "reuse byte drain");
        auto b = reuse_bytes(f); reuse_canaries(f, b, on);
        for (std::size_t i = 0; i < immutable.size(); ++i)
            if (b[9U + i] != immutable[i]) throw std::runtime_error("reuse immutable weights/gates modified");
        return b;
    };
    const auto a = run(14U);
    if (a != run(15U) || a != run(14U)) throw std::runtime_error("wide-rows/reuse/restored wide-rows byte mismatch");
    std::cout << "HC inverse reuse EXACT rows " << f.f.rows << " site " << f.kind << " bf16 " << f.f.bf16
              << " in_place " << f.in_place << " wide_rows " << on << " inverse_probe " << (inverses != nullptr) << '\n';
}
void fixtures_inverse_reuse(hipStream_t stream) {
    for (bool bf16 : {true, false})
        for (unsigned rows : {1U, 33U, 65U, 264U})
            for (unsigned kind : {0U, 1U, 2U})
                for (unsigned flavor : {0U, 1U, 2U}) {
                    RecomputeFixture f(rows, kind, flavor, true, flavor == 1U ? 0.0F : 1e-6F, bf16, true);
                    compare_inverse_reuse(f, stream);
                }
    for (bool bf16 : {true, false}) {
        RecomputeFixture f(65U, 1U, 2U, false, 1e-6F, bf16, true);
        std::vector<float> inverse(f.f.rows * 4U);
        constexpr std::uint32_t bits[]{0U, 0x80000000U, 1U, 0x80000001U, 0x007fffffU, 0x00800000U,
            0x7f800000U, 0xff800000U, 0x7fc12345U, 0x7f812345U, 0xffc54321U, 0x3f800001U};
        for (std::size_t i = 0; i < inverse.size(); ++i) inverse[i] = std::bit_cast<float>(bits[i % std::size(bits)]);
        compare_inverse_reuse(f, stream, true, &inverse);
        // Wide rows OFF + requested candidate must restore unchanged image output14.
        compare_inverse_reuse(f, stream, false);
        RecomputeFixture production(16384U, 1U, 0U, false, 1e-6F, bf16, true);
        compare_inverse_reuse(production, stream);
    }
    RecomputeFixture alias(65U, 1U, 0U, false, 1e-6F);
    auto v = alias.f.views();
    v.recompute_input = alias.stream_data(); v.recompute_stride = RecomputeFixture::ss;
    v.recompute_inverse = static_cast<const float*>(alias.inverse.p);
    v.recompute_norm = static_cast<const float*>(alias.norm.p);
    v.output = alias.stream_data(); v.output_stride = RecomputeFixture::ss;
    for (unsigned form : {14U, 15U}) {
        bool refused = false;
        try { strix::testing::enqueue_wide_hyper_rows_test(alias.c, v, 4U, form, stream); }
        catch (const std::invalid_argument&) { refused = true; }
        if (!refused) throw std::runtime_error("inverse reuse accepted activation/stream aliasing");
    }
}
double reuse_median(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("empty median");
    std::sort(values.begin(), values.end());
    const auto n = values.size();
    return n % 2U ? values[n / 2U] : (values[n / 2U - 1U] + values[n / 2U]) / 2.0;
}
void time_inverse_reuse(unsigned kind, hipStream_t stream) {
    RecomputeFixture f(16384U, kind, 0U, false, 1e-6F, true, true);
    compare_inverse_reuse(f, stream);
    hipEvent_t begin{}, end{};
    check(hipEventCreate(&begin), "reuse begin"); check(hipEventCreate(&end), "reuse end");
    const auto run = [&](bool b) {
        f.reset(); // outside the original complete-transaction event boundaries
        check(hipEventRecord(begin, stream), "reuse start");
        f.enqueue(true, b ? 15U : 14U, 0U, stream);
        check(hipEventRecord(end, stream), "reuse stop"); check(hipEventSynchronize(end), "reuse event drain");
        float ms = 0.0F; check(hipEventElapsedTime(&ms, begin, end), "reuse elapsed"); return static_cast<double>(ms);
    };
    for (unsigned i = 0; i < 2U; ++i) { (void)run(false); (void)run(true); }
    const auto campaign = [&](bool abba) {
        std::vector<double> a, b;
        for (unsigned q = 0; q < 4U; ++q) {
            std::vector<double> qa, qb;
            for (unsigned pos = 0; pos < 4U; ++pos) {
                const bool second = pos == 1U || pos == 2U;
                const double ms = run(abba && second);
                (second ? b : a).push_back(ms); (second ? qb : qa).push_back(ms);
                std::cout << "HC_REUSE_SAMPLE phase " << (abba ? "ABBA" : "AA") << " quartet " << q << " pos " << pos
                          << " arm " << (abba && second ? 'B' : 'A') << " ms " << ms << '\n';
            }
            std::cout << "HC_REUSE_QUARTET phase " << (abba ? "ABBA" : "AA") << " quartet " << q
                      << " ratio " << reuse_median(qb) / reuse_median(qa) << '\n';
        }
        std::cout << "HC_REUSE_SUMMARY phase " << (abba ? "ABBA" : "AA") << " site " << kind
                  << " median_A_ms " << reuse_median(a) << " median_second_ms " << reuse_median(b) << '\n';
    };
    campaign(false); campaign(true);
    compare_inverse_reuse(f, stream);
    check(hipEventDestroy(begin), "reuse destroy begin"); check(hipEventDestroy(end), "reuse destroy end");
}
