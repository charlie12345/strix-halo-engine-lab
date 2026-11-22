// The planned entry has a compact group for each selection slot. The legacy
// entry still computes that rank in the kernel, including for repeated IDs.
    if (args.failure && *args.failure != 0U) return;
    __shared__ float terms[STRIX_R1_DOWN_COLUMNS][TopK];
    const unsigned col = blockIdx.x * STRIX_R1_DOWN_COLUMNS;
    const unsigned live = args.hidden - col;
    const unsigned lane = threadIdx.x % kWarp;
    for (unsigned slot = threadIdx.x / kWarp; slot < TopK; slot += kWarps) {
        const unsigned expert = args.ids[slot];
        if (expert >= args.experts) continue;
        const auto* input = args.activation + std::size_t{slot} * args.intermediate;
        float projected[STRIX_R1_DOWN_COLUMNS];
        dot_rows<Format, STRIX_R1_DOWN_COLUMNS>(args.matrices[expert].down, col, 0U, live,
                                       input, args.intermediate, lane, projected);
        if (lane == 0U) {
#if defined(STRIX_R1_PLANNED_DOWN)
            const unsigned rank = group_by_slot[slot];
#else
            unsigned rank = 0;
#pragma unroll
            for (unsigned i = 0; i < TopK; ++i) {
                const auto other = args.ids[i];
                rank += other < expert || (other == expert && i < slot);
            }
#endif
#pragma unroll
            for (unsigned i = 0; i < STRIX_R1_DOWN_COLUMNS; ++i) {
                terms[i][rank] = combine::weighted_bf16<false>(args.weights[slot], projected[i]);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0U) {
#pragma unroll
        for (unsigned slot = 0; slot < TopK; ++slot) {
            if (args.ids[slot] >= args.experts) return;
        }
#pragma unroll
        for (unsigned i = 0; i < STRIX_R1_DOWN_COLUMNS; ++i) {
            if (i >= live) break;
            float result = 0.0F;
#pragma unroll
            for (unsigned slot = 0; slot < TopK; ++slot) result = __fadd_rn(result, terms[i][slot]);
            args.output[col + i] = combine::rounded_bf16<false>(result);
        }
    }
