// Included inside glm_pf_exact_moe.hip's anonymous namespace (Args, Tile, E, H, I, P, ok, activation_kernel,
// combine_kernel, launch_linear_w): the FAST_I4 stored routed experts on the strix-trellis grouped
// IU4 (W4A4, g64) core. Lossy, a new identity on top of glm-pf-wmma-x; shared expert and every other linear unchanged.
//   weights: EXL3 tiles decoded with the engine's own mul1_pair (the FP16 values the WMMA-X cores multiply), then per
//            output column and K64 group s = FP16(max|w| / 7), q = RNE(w / s) in [-7, 7], nibbles in the trellis K16 x N16
//            tile order (strix::core::iu4_weight_byte), all in the had128-rotated basis (the EXL3 suh / svh stay).
//   activations: the transformed FP16 rows (input_kernel: suh + had128; activation_kernel: R15 + down suh + had128),
//            per row and K64 group (library quantiser), gathered through the plan's pair map.
//   epilogue: the library's fused output transform, had128 over each 128-column block times the expert's svh.
#if STRIX_GLM_PF_IU4_LIB
// input_kernel with routed rows quantised (row pair * 2 + m of aq/as); the shared expert's rows stay FP16 in a.input.
__global__ void iu4_input_kernel(Args a, std::uint8_t* aq, float* as) {
    if (a.ntiles && blockIdx.y >= *a.ntiles) return;
    const Tile tile = a.tiles[blockIdx.y];
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned m = blockIdx.z;
    const moe::Exl3 w = matrix(a, tile.expert, m);
    for (unsigned r = 0; r < tile.count; ++r) {
        const unsigned pair = a.pairs[tile.first + r];
        const unsigned begin = (blockIdx.x * 8U + wave) * 128U;
        if (tile.expert == E) {
            d::transform_block_at(a.x + std::size_t{pair / P} * H, w.suh, a.input + (std::size_t{pair} * 2U + m) * H + begin, begin, 0, lane);
            continue;
        }
        float v[4];
        d::transform_values(a.x + std::size_t{pair / P} * H, w.suh, begin, lane, v);
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) v[j] = static_cast<float>(static_cast<_Float16>(v[j]));
        const std::size_t row = std::size_t{pair} * 2U + m;
        strix::core::iu4_quant128(v, aq + row * (H / 2U), as + row * (H / 64U), begin, lane);
    }
}
// activation_kernel with routed rows quantised (row pair of aq/as); shared rows stay FP16 in a.at.
__global__ void iu4_activation_kernel(Args a, std::uint8_t* aq, float* as) {
    if (a.ntiles && blockIdx.y >= *a.ntiles) return;
    const Tile tile = a.tiles[blockIdx.y];
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const moe::Exl3 down = matrix(a, tile.expert, 2);
    for (unsigned r = wave; r < tile.count; r += 8) {
        const unsigned pair = a.pairs[tile.first + r];
        float v[4];
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const unsigned i = blockIdx.x * 128 + lane + 32 * j;
            const float g = a.gu[(std::size_t{pair} * 2) * I + i];
            const float u = a.gu[(std::size_t{pair} * 2 + 1) * I + i];
            const float gg = g > kClamp ? kClamp : g;
            const float uu = u > kClamp ? kClamp : (u < -kClamp ? -kClamp : u);
            const float act = silu(gg) * uu;
            const float scaled = d::materialize(act * d::half_bits(down.suh[i]));
            v[j] = static_cast<float>(static_cast<_Float16>(scaled));
        }
        d::hadamard128(v, lane);
        if (tile.expert == E) {
#pragma unroll
            for (unsigned j = 0; j < 4; ++j)
                a.at[std::size_t{pair} * I + blockIdx.x * 128 + lane + 32 * j] = static_cast<_Float16>(v[j]);
            continue;
        }
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) v[j] = static_cast<float>(static_cast<_Float16>(v[j]));
        strix::core::iu4_quant128(v, aq + std::size_t{pair} * (I / 2U), as + std::size_t{pair} * (I / 64U), blockIdx.x * 128U, lane);
    }
}
// Routed gate/up/down on IU4; shared expert, activation and combine as the WMMA-X path (aux_act off). Runs the whole
// call from the input transform on (launch_grouped returns before its own input_kernel).
void launch_iu4(Args a, unsigned grid_tiles, const moe::ExpertSlices& shared, const iu4::Layer& q, const iu4::Scratch& s,
                hipStream_t stream) {
    namespace st = strix::core;
    wmma_x::iu4_moe_calls.fetch_add(1, std::memory_order_relaxed);
    a.aux_act = false;
    a.aux_combine = aux::enabled_default_on("STRIX_GLM_PF_AUX_COMBINE");
    const unsigned pairs = a.rows * P;
    const int plan = iu4_plan(pairs);
    const auto max_tiles = static_cast<std::uint32_t>(st::iu4_grouped_max_tiles(std::size_t{a.rows} * 8U, static_cast<int>(E), plan));
    if (pairs > s.max_rows * P || max_tiles > s.max_tiles) throw std::logic_error("glm pf iu4: scratch smaller than the call");
    auto* tiles = static_cast<st::IU4GroupTile*>(s.tiles);
    static_assert(sizeof(Tile) == sizeof(st::IU4Group) && offsetof(Tile, first) == offsetof(st::IU4Group, first) &&
                  offsetof(Tile, count) == offsetof(st::IU4Group, count), "plan groups are the library's IU4Group");
    const bool fuse = iu4_fuse_quant();
    if (fuse) {
        iu4_input_kernel<<<dim3(H / 1024, grid_tiles, 2), 256, 0, stream>>>(a, s.aq_gu, s.as_gu);
        ok(hipGetLastError(), "iu4 gate/up input transform + quantiser");
    } else {
        input_kernel<<<dim3(H / 1024, grid_tiles, 2), 256, 0, stream>>>(a);
        ok(hipGetLastError(), "iu4 gate/up input transform");
        iu4_ok(st::quantise_wmma_iu4_activations(a.input, s.aq_gu, s.as_gu, static_cast<int>(H), static_cast<int>(pairs * 2U), stream, 64),
               "iu4 gate/up activation quantiser");
    }
    iu4_ok(st::build_iu4_grouped_tiles(reinterpret_cast<const st::IU4Group*>(a.groups), static_cast<int>(E), plan, tiles, s.ntiles,
                                       max_tiles, stream), "iu4 tile list");
    for (unsigned m = 0; m < 2U; ++m) {
        st::IU4GroupedOperands o;
        o.a_q = s.aq_gu; o.a_scale = s.as_gu; o.w_q = q.wq[m]; o.w_scale = reinterpret_cast<const _Float16*>(q.ws[m]);
        o.w_expert_bytes = std::size_t{H} * I / 2U; o.w_scale_expert = std::size_t{I} * (H / 64U);
        o.col_scale = q.svh[m]; o.map = a.pairs; o.a_mul = 2U; o.a_add = m; o.o_mul = 2U; o.o_add = m;
        iu4_ok(st::launch_wmma_tiled_iu4_grouped(o, tiles, s.ntiles, max_tiles, a.gu, static_cast<int>(H), static_cast<int>(I), stream, plan),
               "iu4 routed gate/up");
    }
    launch_linear_w<false, false>(a, shared.gate.bits, stream);  // shared expert only
    if (fuse) {
        iu4_activation_kernel<<<dim3(I / 128, grid_tiles), 256, 0, stream>>>(a, s.aq_down, s.as_down);
        ok(hipGetLastError(), "iu4 R15/down transform + quantiser");
    } else {
        activation_kernel<<<dim3(I / 128, grid_tiles), 256, 0, stream>>>(a);
        ok(hipGetLastError(), "iu4 R15/down transform");
        iu4_ok(st::quantise_wmma_iu4_activations(a.at, s.aq_down, s.as_down, static_cast<int>(I), static_cast<int>(pairs), stream, 64),
               "iu4 down activation quantiser");
    }
    st::IU4GroupedOperands o;
    o.a_q = s.aq_down; o.a_scale = s.as_down; o.w_q = q.wq[2]; o.w_scale = reinterpret_cast<const _Float16*>(q.ws[2]);
    o.w_expert_bytes = std::size_t{I} * H / 2U; o.w_scale_expert = std::size_t{H} * (I / 64U);
    o.col_scale = q.svh[2]; o.map = a.pairs;
    iu4_ok(st::launch_wmma_tiled_iu4_grouped(o, tiles, s.ntiles, max_tiles, a.output, static_cast<int>(I), static_cast<int>(H), stream, plan),
           "iu4 routed down");
    launch_linear_w<true, false>(a, shared.down.bits, stream);  // shared down (its aux combine reads the routed outputs)
    if (!a.aux_combine) {
        combine_kernel<<<dim3(H / 256, a.rows), 256, 0, stream>>>(a);
        ok(hipGetLastError(), "iu4 R16");
    }
}
#endif
