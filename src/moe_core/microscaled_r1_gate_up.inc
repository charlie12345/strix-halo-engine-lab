// Included in the R1 kernel so column specialization preserves the legacy code.
    if (args.failure && *args.failure != 0U) return;
    const unsigned slot = blockIdx.y;
    const unsigned expert = args.ids[slot];
    const unsigned col = gate_first_column(blockIdx.x, threadIdx.x / kWarp, STRIX_R1_GATE_COLUMNS);
    if (expert >= args.experts || col >= args.intermediate) return;
    const unsigned lane = threadIdx.x % kWarp;
    const unsigned live = args.intermediate - col;
    float pair[2U * STRIX_R1_GATE_COLUMNS];
    dot_rows<Format, STRIX_R1_GATE_COLUMNS>(args.matrices[expert].gate_up, col, args.intermediate, live,
                                   args.input, args.hidden, lane, pair);
    if (lane == 0U) {
        const moe::SoftcappedSwiglu<activation::DeviceMath> swiglu{args.gate_cap, args.up_cap};
#pragma unroll
        for (unsigned i = 0; i < STRIX_R1_GATE_COLUMNS; ++i) {
            if (i >= live) break;
            args.activation[std::size_t{slot} * args.intermediate + col + i] =
                activation::activate_bf16(swiglu, pair[i], pair[STRIX_R1_GATE_COLUMNS + i]);
        }
    }
