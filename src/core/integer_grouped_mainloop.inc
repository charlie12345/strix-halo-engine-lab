// Chunk-staged IU8 plan. Recovered from the project's split-v2 implementation;
// routing, GGUF decoding and epilogues remain operand/adapter policies.
static_assert(AD && !Fused && Probe == 0 && BM == 16 && BN == 64);
static_assert(!Operand::fast_i4 && !Contract::relaxed);
using Plan = typename Operand::Plan;
static_assert(Plan::class_waves == 8 && Plan::chunk_groups == 32 && Plan::half_exchange);
__shared__ unsigned long long grid[!Operand::affine && !Operand::direct_weights && Plan::weight_lds ? 256 : 1];
__shared__ unsigned sources[16], slots[Adapter::kFinish ? 16 : 1];
__shared__ float values[Adapter::kFinish ? BM * BN : 1];
if constexpr (!Operand::affine && !Operand::direct_weights && Plan::weight_lds)
    for (unsigned i = unsigned(thread); i < 256; i += G::threads) grid[i] = ad.grid()[i];
if (thread < rows) {
    sources[thread] = ad.source(at, thread);
    if constexpr (Adapter::kFinish) slots[thread] = ad.slot(at, thread);
}
ad.bind_rows(sources, Adapter::kFinish ? slots : sources);
using Schedule = IntegerGroupedSchedule<Operand>;
const unsigned pair = unsigned(wave) & 1, q = unsigned(wave) >> 1, lo = unsigned(lane) & 15, h = unsigned(lane) >> 4;
const typename Schedule::WeightRows weight_rows{{ad.weight_row(at, first_col), ad.weight_row(at, first_col + 32)}, ad.row_bytes()};
float leaf[4][2][8];
Schedule::run(lds, q, unsigned(inputs), sources, unsigned(rows), ad.activation_codes(), ad.activation_deltas(), grid, leaf, weight_rows, pair * 32U + unsigned(lane));
float v[2][8];
#pragma unroll
for (unsigned t = 0; t < 2; ++t)
    #pragma unroll
    for (unsigned e = 0; e < 8; ++e)
        v[t][e] = (leaf[0][t][e] + leaf[2][t][e]) + (leaf[1][t][e] + leaf[3][t][e]);
auto* exchange = reinterpret_cast<float*>(lds);
#pragma unroll
for (unsigned s = 4; s; s /= 2) {
    Schedule::sync();
    if (q >= s && q < 2 * s) {
        auto* slot = exchange + (4 * pair + q - s) * 512;
        #pragma unroll
        for (unsigned t = 0; t < 2; ++t)
            #pragma unroll
            for (unsigned e = 0; e < 8; ++e) slot[(8 * t + e) * 32 + unsigned(lane)] = v[t][e];
    }
    Schedule::sync();
    if (q < s) {
        const auto* slot = exchange + (4 * pair + q) * 512;
        #pragma unroll
        for (unsigned t = 0; t < 2; ++t)
            #pragma unroll
            for (unsigned e = 0; e < 8; ++e) v[t][e] = v[t][e] + slot[(8 * t + e) * 32 + unsigned(lane)];
    }
}
if (q == 0) {
    #pragma unroll
    for (unsigned t = 0; t < 2; ++t)
        #pragma unroll
        for (unsigned e = 0; e < 8; ++e) {
            const unsigned r = 2 * e + h, c = pair * 32 + t * 16 + lo;
            if (r < unsigned(rows)) {
                if constexpr (Adapter::kFinish) values[integer_pair_value_offset(pair, r, t * 16U + lo)] = v[t][e];
                else ad.store_raw(at, int(r), first_col + int(c), v[t][e]);
            }
        }
}
if constexpr (Adapter::kFinish) {
    __syncthreads();
    ad.template finish_tile<true>(at, values, BM, BN);
}
