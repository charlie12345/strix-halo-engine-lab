# Experimental exact prefill schedules

Both experiments are off unless their environment switch is exactly `1`.

`STRIX_DS41_PREFILL_SLABS=1` permits attention calls of up to 256 rows. The
default call limit stays 64. A prefill without an explicit nondefault window
selects 64, 128 or 256 rows from its row bound and projection chunk size; the
selected window divides the chunk and is reduced further to bound total
attention-lane scratch to 2 GiB. This bound includes scores, keys, query prep,
partials and tickets; row buffers and candidate masks are separate.
`STRIX_DS41_PREFILL_WINDOW` overrides the
selection. The selected window also defines row ownership in distributed
prefill and is fixed before scratch allocation. The decode window remains eight
rows. Compression, key visibility, key tiles, chunk boundaries, softmax and
chunk combination retain their existing per-row arithmetic.

Dense and routed prefill use the split-K kernels. The unsplit-tile switch measured
slower on the current kernels and has been removed. The resource-based tile candidates
in `ds41_prefill_tuning.hpp` (dense outputs, routed rows, down stage) are
provisional and not wired into dispatch. They are not a measured tuning table.

The integer group products, 32 class chains, separate floating-point scale
products/additions, reduction tree, clamped SwiGLU and A8 rounding are unchanged.
An output tile changes which block owns an output, not its arithmetic sequence.
Neither switch selects a lower precision numerical contract.

Before enabling either switch by default, compare complete workload results and
timings on the target device. Slab scratch is proportional to the selected rows
times the context bound and attention lane count; increasing the slab is not a
free increase in batching. Export/import and continuation gates must retain the
same live attention state.
