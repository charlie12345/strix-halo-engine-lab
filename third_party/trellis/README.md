# Shared tiled core source snapshot

Project-owned strix-trellis headers were copied from branch `core/wmmax-glm`,
commit `c5c50e364ab02372d14cde502083367e73f0ebb4`. The original Apache-2.0
LICENSE and NOTICE are included. EXL3 format attribution remains in NOTICE.

The header-only policy-v1 core is instantiated by DS41 through
`STRIX_DS41_TRELLIS_DIR` (default: this directory). No second archive, resident
GGUF-to-IU4 pack or converter is linked. Local additions are `IntegerOperand`,
integer fragment operations, the class-ordered integer mainloop, a no-rotation
raw epilogue and shared A8 quantization. Dense uses the same kernel with one group.

Exact integer execution preserves K32 product boundaries and the high-bit-first
32-class reduction contract. FAST_I4 is an explicit draft numerical policy:
A8 codes are requantized to signed A4 per K32; IQ2_XXS grid magnitudes 8,25,43
map to 1,4,7 with FP32 scale correction 43/7; Q2_K keeps raw 0..3 codes,
independent half scales and a minimum correction from A-row sums. FP32 weight
scales are decoded directly from GGUF. No Hadamard rotation is introduced.

Host tests do not qualify native execution, fidelity or throughput. See the
root device requests for the remaining gates. Do not enable FAST_I4 by default.
