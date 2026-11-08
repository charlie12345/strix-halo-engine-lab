// Included in the existing launcher test executable; no additional CMake target.
struct PleTileBuffer {
    static constexpr std::size_t guard = 256;
    static constexpr std::uint32_t poison = 0xa5a5a5a5U;
    std::uint32_t* allocation = nullptr;
    std::size_t words;
    explicit PleTileBuffer(std::size_t n) : words(n) {
        check(hipMalloc(reinterpret_cast<void**>(&allocation), (n + 2U * guard) * 4U), "PLE tile allocate");
        reset();
    }
    ~PleTileBuffer() { if (allocation) (void)hipFree(allocation); }
    PleTileBuffer(const PleTileBuffer&) = delete;
    float* data() const { return reinterpret_cast<float*>(allocation + guard); }
    void reset() const { check(hipMemset(allocation, 0xa5, (words + 2U * guard) * 4U), "PLE tile poison"); }
    std::vector<std::uint32_t> read() const {
        std::vector<std::uint32_t> result(words + 2U * guard);
        check(hipMemcpy(result.data(), allocation, result.size() * 4U, hipMemcpyDeviceToHost), "PLE tile read");
        for (std::size_t i = 0; i < guard; ++i)
            ple_tile_host::require(result[i] == poison && result[guard + words + i] == poison,
                "PLE tile prefix history/suffix canary changed");
        return result;
    }
};

struct PleTileWeights {
    void* device = nullptr;
    PleTileWeights(unsigned width, unsigned outputs, unsigned seed, bool special) {
        const auto blocks = std::size_t{width / 32U} * outputs;
        std::vector<unsigned char> bytes(blocks * 34U);
        constexpr std::uint16_t scales[] = {0x0001, 0x0400, 0x3400, 0xb400, 0, 0x8000, 0x7c00, 0xfc00, 0x7e12};
        for (std::size_t b = 0; b < blocks; ++b) {
            const auto scale = scales[(b + seed) % (special ? 9U : 4U)];
            std::memcpy(bytes.data() + b * 34U, &scale, 2U);
            for (unsigned j = 0; j < 32U; ++j)
                bytes[b * 34U + 2U + j] = static_cast<unsigned char>((b * 37U + j * 13U + seed) & 255U);
        }
        check(hipMalloc(&device, bytes.size()), "PLE tile weights allocate");
        check(hipMemcpy(device, bytes.data(), bytes.size(), hipMemcpyHostToDevice), "PLE tile weights upload");
    }
    ~PleTileWeights() { if (device) (void)hipFree(device); }
};

void ple_tile_compare(const PleTileBuffer& reference, const PleTileBuffer& candidate,
    unsigned rows, unsigned outputs, unsigned stride) {
    // Compare integer words, including NaN payloads, signed zeros, all row padding
    // and a preserved suffix row. Never use floating tolerance or NaN equivalence.
    const auto a = reference.read(), b = candidate.read();
    ple_tile_host::require(a == b, "PLE tile complete projection bytes differ");
    for (std::size_t t = 0; t < reference.words; ++t)
        if (t / stride >= rows || t % stride >= outputs)
            ple_tile_host::require(a[PleTileBuffer::guard + t] == PleTileBuffer::poison,
                "PLE tile preserved output/padding overwritten");
}

void ple_tile_tail_transaction(unsigned rows, unsigned expanded, unsigned hidden, unsigned key_stride, unsigned value_stride,
    const PleTileBuffer& ka, const PleTileBuffer& kb, const PleTileBuffer& va, const PleTileBuffer& vb, hipStream_t stream) {
    const unsigned stride = expanded + 8U, history = 9U;
    TailBuffer<float> ak(std::size_t{rows} * expanded), bk(ak.n), av(std::size_t{rows} * hidden), bv(av.n);
    for (unsigned row = 0; row < rows; ++row) {
        check(hipMemcpyAsync(ak.p + std::size_t{row} * expanded, ka.data() + std::size_t{row} * key_stride,
            expanded * 4U, hipMemcpyDeviceToDevice, stream), "PLE tail key A");
        check(hipMemcpyAsync(bk.p + std::size_t{row} * expanded, kb.data() + std::size_t{row} * key_stride,
            expanded * 4U, hipMemcpyDeviceToDevice, stream), "PLE tail key B");
        check(hipMemcpyAsync(av.p + std::size_t{row} * hidden, va.data() + std::size_t{row} * value_stride,
            hidden * 4U, hipMemcpyDeviceToDevice, stream), "PLE tail value A");
        check(hipMemcpyAsync(bv.p + std::size_t{row} * hidden, vb.data() + std::size_t{row} * value_stride,
            hidden * 4U, hipMemcpyDeviceToDevice, stream), "PLE tail value B");
    }
    TailBuffer<std::uint16_t> a(std::size_t{rows + 1U} * stride), b(a.n);
    TailBuffer<std::uint16_t> sa(std::size_t{expanded} * history), sb(sa.n), norm(expanded), conv(std::size_t{expanded} * 4U);
    std::vector<std::uint16_t> input(a.n, 0xa5a5), state(sa.n, 0x3d80);
    for (unsigned row = 0; row < rows; ++row)
        for (unsigned f = 0; f < expanded; ++f) input[std::size_t{row} * stride + f] = ple_tile_host::input_bits(row, f);
    a.put(input); b.put(input); sa.put(state); sb.put(state);
    norm.put(std::vector<std::uint16_t>(expanded, 0x3f80)); conv.put(std::vector<std::uint16_t>(conv.n, 0x3c00));
    TailBuffer<float> scratch_a(std::size_t{rows} * 4U + 2U * expanded), scratch_b(scratch_a.n);
    strix::testing::enqueue_ple_tail(true, ak.p, av.p, a.p, stride, a.p, stride, sa.p,
        norm.p, norm.p, norm.p, conv.p, true, rows, hidden, 4U, 4U, 3U, 1e-6F, scratch_a.p, stream);
    strix::testing::enqueue_ple_tail(true, bk.p, bv.p, b.p, stride, b.p, stride, sb.p,
        norm.p, norm.p, norm.p, conv.p, true, rows, hidden, 4U, 4U, 3U, 1e-6F, scratch_b.p, stream);
    check(hipStreamSynchronize(stream), "PLE tile composed tail sync");
    const auto output_a = a.read(), output_b = b.read();
    ple_tile_host::require(output_a == output_b && sa.read() == sb.read(), "PLE tile BF16 output/history differs");
    for (std::size_t t = 0; t < output_a.size(); ++t)
        if (t / stride >= rows || t % stride >= expanded)
            ple_tile_host::require(output_a[t] == input[t], "PLE tile composed tail changed preserved output/padding");
    std::cout << "EXACT composed PLE BF16 in-place output and convolution history rows=" << rows << '\n';
}

void ple_tile_case(unsigned rows, unsigned width, unsigned key_outputs, unsigned value_outputs,
    unsigned extra_pitch, int mode, bool expected, bool special, bool bench) {
    const unsigned input_stride = width + extra_pitch, key_stride = key_outputs + 3U, value_stride = value_outputs + 5U;
    std::vector<std::uint16_t> input(std::size_t{rows + 1U} * input_stride, 0xa5a5);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned f = 0; f < width; ++f)
            input[std::size_t{r} * input_stride + f] = bench ? static_cast<std::uint16_t>(0x3c00U + (r + f) % 1024U) :
                ple_tile_host::input_bits(r, f);
    PleTileBuffer din((input.size() + 1U) / 2U);
    check(hipMemcpy(din.data(), input.data(), input.size() * 2U, hipMemcpyHostToDevice), "PLE tile input upload");
    const auto before = din.read();
    PleTileWeights kw(width, key_outputs, 1U, special), vw(width, value_outputs, 7U, special);
    strix::testing::invalidate_ple_projection_weights();
    PleTileBuffer ka(std::size_t{rows + 1U} * key_stride), kb(ka.words);
    PleTileBuffer va(std::size_t{rows + 1U} * value_stride), vb(va.words);
    hipStream_t stream = nullptr;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "PLE tile stream");
    const auto transaction = [&](int setting, bool b) {
        return strix::testing::enqueue_ple_projection_pair(setting,
            reinterpret_cast<const std::uint16_t*>(din.data()), kw.device, vw.device,
            width, key_outputs, value_outputs, rows, input_stride,
            b ? kb.data() : ka.data(), key_stride, b ? vb.data() : va.data(), value_stride, stream);
    };
    ple_tile_host::require(!transaction(0, false), "OFF fixture unexpectedly reused panel");
    ple_tile_host::require(transaction(mode, true) == expected, "PLE tile route witness differs");
    check(hipStreamSynchronize(stream), "PLE tile equality sync");
    ple_tile_compare(ka, kb, rows, key_outputs, key_stride);
    ple_tile_compare(va, vb, rows, value_outputs, value_stride);
    ple_tile_host::require(din.read() == before, "PLE tile input/history/pitch/suffix bytes changed");
    if (rows == 65U && key_outputs == 10240U && value_outputs == 2560U)
        ple_tile_tail_transaction(rows, key_outputs, value_outputs, key_stride, value_stride, ka, kb, va, vb, stream);
    std::cout << "EXACT PLE pair rows=" << rows << " width=" << width << " key=" << key_outputs
              << " value=" << value_outputs << " pitch=" << input_stride << " reused=" << expected << '\n';
    if (bench) {
        // Producer + both consumers, including cold weight pack when invalidated.
        // Synchronize before reusing event handles; no projection-only timing.
        hipEvent_t start = nullptr, stop = nullptr;
        check(hipEventCreate(&start), "PLE tile start event"); check(hipEventCreate(&stop), "PLE tile stop event");
        const auto measure = [&](bool b, bool cold) {
            if (cold) strix::testing::invalidate_ple_projection_weights();
            const auto wall_start = std::chrono::steady_clock::now();
            check(hipEventRecord(start, stream), "PLE tile start");
            transaction(b ? 1 : 0, b);
            check(hipEventRecord(stop, stream), "PLE tile stop"); check(hipEventSynchronize(stop), "PLE tile wait");
            const auto wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start).count();
            float ms = 0; check(hipEventElapsedTime(&ms, start, stop), "PLE tile elapsed");
            const double scale = 32768.0 / static_cast<double>(rows) / 1000.0;
            std::cout << "PLE sample rows=" << rows << " cold=" << cold << " arm=" << b
                      << " event_ms=" << ms << " wall_ms=" << wall_ms
                      << " modeled32K-stage-floor-ratio=" << ms * scale / 0.10 << ',' << wall_ms * scale / 0.10
                      << " modeled11.9s-floor-fraction=" << ms * scale / 11.9 << ',' << wall_ms * scale / 11.9 << '\n';
            return ms;
        };
        const auto med = [](const std::vector<float>& v) { return ple_tile_host::median(v); };
        for (bool cold : {false, true}) {
            std::vector<float> aa0, aa1, a, b;
            for (unsigned rep = 0; rep < 3U; ++rep) { aa0.push_back(measure(false, cold)); aa1.push_back(measure(false, cold)); }
            for (unsigned rep = 0; rep < 3U; ++rep) {
                a.push_back(measure(false, cold)); b.push_back(measure(true, cold));
                b.push_back(measure(true, cold)); a.push_back(measure(false, cold));
            }
            const auto scale32k = 32768.0 / static_cast<double>(rows) / 1000.0;
            const double a_seconds = med(a) * scale32k, b_seconds = med(b) * scale32k;
            std::cout << "PLE " << (cold ? "cold" : "warm") << " A/A ms=" << med(aa0) << ',' << med(aa1)
                      << " ABBA ms=" << med(a) << ',' << med(b)
                      << " AA-modeled32K-stage-floor-ratio=" << med(aa0) * scale32k / 0.10 << ',' << med(aa1) * scale32k / 0.10
                      << " modeled32K-stage-floor-ratio=" << a_seconds / 0.10 << ',' << b_seconds / 0.10
                      << " modeled11.9s-floor-fraction=" << a_seconds / 11.9 << ',' << b_seconds / 11.9 << '\n';
        }
        check(hipEventDestroy(start), "PLE tile destroy start"); check(hipEventDestroy(stop), "PLE tile destroy stop");
        // Timing cannot hide stores or non-repeatability from the byte gate.
        ple_tile_compare(ka, kb, rows, key_outputs, key_stride);
        ple_tile_compare(va, vb, rows, value_outputs, value_stride);
        ple_tile_host::require(din.read() == before, "PLE timing mutated input/history");
    }
    check(hipStreamDestroy(stream), "PLE tile destroy stream");
}

void ple_tile_exact_cases(bool inactive) {
    // Production T values, Qwen dimensions, padded input and independent output pitches.
    for (unsigned rows : {8192U, 16384U})
        ple_tile_case(rows, 2560U, 10240U, 2560U, 8U, -1, !inactive, false, false);
    // Partial output tiles, row remainders, nonfinite scales, zero signs and odd pitch fallback.
    for (unsigned rows : {63U, 64U, 65U, 127U})
        ple_tile_case(rows, 512U, 2051U, 2113U, 8U, -1, !inactive && rows >= 64U, true, false);
    ple_tile_case(64U, 512U, 2051U, 2113U, 1U, -1, false, false, false);
    ple_tile_case(65U, 2560U, 10240U, 2560U, 8U, -1, !inactive, false, false);
}
