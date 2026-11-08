# MiniMax descriptive graph and plan

The MiniMax adapter validates `minimax-m3.1-flash.toml` with the compiled
model-description contract before constructing a common `ModelGraph`. This is
description and diagnostic tooling; the existing engine executes its own schedule.
No numerical, cache, saved-state or compiled execution identity is replaced.

The graph records the full 60-layer MSA trunk, per-head Gemma RMSNorm and rotary
semantics, block indexer, three sequence caches, dense prefix, routed and unscaled
shared experts, residual edges, final norm and vocabulary head. Head and MLP
intermediate partitionability are model semantics; concrete TP2 rank ranges and
vocabulary padding belong to the plan. Both ranks own intermediate slices of all
128 routed expert IDs. They do not own disjoint expert-ID ranges.

Configured draft assets appear as an opaque draft attachment. Their internal
blocks, taps, verification and acceptance schedule remain engine-local. The source
description declares no MTP block, so the adapter does not infer an MTP attachment.
Attachment presence describes configuration, not successful asset loading.

Place the repository model description beside the model assets, then inspect with
the normal launch shape on either rank:

```sh
strix-minimax-rank MODEL_DIR 0 usb4-auto OUTPUT_DIR --print-model-graph
strix-minimax-rank MODEL_DIR 1 usb4-auto OUTPUT_DIR --print-plan
strix-minimax-serve MODEL_DIR --rank 0 --transport usb4-auto --print-plan
```

`STRIX_PRINT_MODEL_GRAPH=1` and `STRIX_PRINT_EXEC_PLAN=1` are equivalent opt-ins.
Printing exits before transport, HIP queries, weight loading or server startup.
If both are requested, graph JSON precedes plan JSON, one record per line.
`STRIX_MINIMAX_MODEL_GRAPH_CHECK=1` checks the graph against the validated sidecar
and validates its plan before transport, then continues normal execution. A missing
or mismatching sidecar is an error only for these opt-ins; ordinary execution keeps
its existing model-description admission behavior.

The common plan projects one logical decode row, then the greedy head boundary:
input norm; QKV/index projections; per-head norms and RoPE; KV/index append;
block scores/selection and attention; output projection; X1 ordered residual;
post-attention norm; dense MLP or router/routed/shared/combine; X2 ordered residual;
final norm, vocabulary shard logits and pair argmax. A topological listing does
not specify actual stream serialization. Norms fused into an exchange boundary are
listed separately by mathematical role; no launch count is implied.

The plan records the pure `mm_defaults.cpp` resolver values as declared policy,
not evidence that a device selected those kernels. It lists unprojected kernel
selection, fusion, stream/event overlap, physical scratch and cache allocation,
transport waits/publication, prefill scheduling, serving admission and drafting.
Dynamic index scratch has symbolic extents and zero unprojected bytes. Other
activation sizes describe a logical row, and sequence bytes are per retained
position, not a memory-admission estimate. No node is qualified for graph capture
by this projection. Host tests cannot establish device execution or bit equality.

For a first shared RMSNorm module, retain MiniMax's `(x*r)*(1+w)` evaluation order,
FP32 reduction tree, epsilon rounding and exchange-boundary placement. Mathematical
equivalence alone is insufficient for replacing a qualified implementation.
