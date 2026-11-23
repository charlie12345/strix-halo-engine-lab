// Included in the R1 kernel so column specialization preserves the legacy code. blockIdx.y = assignment
// (row * TopK + slot); a one-row launch is the legacy body.
    const unsigned assignment = blockIdx.y;
    if (args.failure && args.failure[assignment / TopK] != 0U) return;
    const unsigned expert = args.ids[assignment];
    const unsigned col = gate_first_column(blockIdx.x, threadIdx.x / kWarp, STRIX_R1_GATE_COLUMNS);
    if (expert >= args.experts || col >= args.intermediate) return;
    const unsigned lane = threadIdx.x % kWarp;
    const unsigned live = args.intermediate - col;
    const auto at = row_offsets<TopK>(assignment / TopK, args.hidden, args.intermediate);
    const float* input = args.input + at.input;
    auto* activation_row = args.activation + at.activation + std::size_t{assignment % TopK} * args.intermediate;
    float pair[2U * STRIX_R1_GATE_COLUMNS];
    dot_rows<Format, STRIX_R1_GATE_COLUMNS>(args.matrices[expert].gate_up, col, args.intermediate, live,
                                            input, args.hidden, lane, pair);
    if (lane == 0U) {
        const moe::SoftcappedSwiglu<activation::DeviceMath> swiglu{args.gate_cap, args.up_cap};
#pragma unroll
        for (unsigned i = 0; i < STRIX_R1_GATE_COLUMNS; ++i) {
            if (i >= live) break;
            activation_row[col + i] =
                activation::activate_bf16(swiglu, pair[i], pair[STRIX_R1_GATE_COLUMNS + i]);
        }
    }
