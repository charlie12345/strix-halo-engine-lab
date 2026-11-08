# BF16 target with the production MTP sidecar

`Config::diagnostic_bf16_mtp` is an explicit source diagnostic capability, default
false. It requires the BF16 target, decode, a real sidecar, eager execution and
no handoff export. The ordinary candidate MTP refusal remains. Cohort, graphs,
prefix reuse, overlap and product admission remain unqualified and guarded.

The target is `46bf863a1bc53cd4`, with the same UD-Q4_K_XL weights. The proposer
calls the existing Q8 production predictor (`cuda-qwen-mtp-q8-production-v1`),
including its real attention/cache and routed expert functions. Its math is
unchanged. BF16 target route storage does not select a new proposer FFN.
Proposal agreement with another numerical target is not required. Acceptance
uses the longest prefix verified by this BF16 target plus its correction;
the reference is BF16 plain execution.

The native `qwen_bf16_mtp_e2e` target links the production engine. Its `real` arm
calls `draft`, `verify` and `commit_cohort`. Other arms force correct, early/late
rejection, rejection, zero/partial commit, stop, budget and cancellation controls;
they never count as real proposer evidence. Every window exports complete finite
raw vocabulary rows, fed IDs, proposals/probabilities, emitted IDs, pending,
history and actual kept state before rebuilding any reference. All proposed
rows, including rejected rows, are compared with independent scalar feeding.
Committed outputs also compare with independent greedy BF16 plain execution.
Continuation restores the captured actual state and retained pool bytes.

Canonical MTP pairs cover `[0,length-1)`; the final pair and rejected draft tail
are provisional. Target KV/index cover `[0,length)`. Canonical fixed state holds
the current GDN matrices, GDN and PLE histories and ngram. The held target row,
position, engine/session generation, target, capacity, sidecar path/vocab binding
and phase are explicit. Path binding is local identity, not an asset content hash;
native receipts must separately bind actual asset hashes. No portable cache or
cross-engine import is admitted.

Derived target pools are truncated to the committed extent before observation,
only in this diagnostic. Actual retained pool prefixes are captured without
recomputation and compared bytewise with scalar reference coverage. MTP validity
uses the existing teacher-cache clamp. Invalid tails are never canonical. GDN
other-buffer output, window snapshots, taps, normalized rows, draft scratch,
selection workspace and provisional cache tails are overwritten before their
next read. There is no random sampler/RNG in this greedy predictor. Depth and
probability stop are call inputs; the fixture journals pending/history/controller
decisions. Pool reference coverage failures fail closed, including at long contexts.

One engine owns target and sidecar weights, all contexts and workspaces. The
fixture reuses that owner for its scalar reference. It preserves requested
context/block capacity, pinned-host policy and expert allocation; it rejects an
observability estimate over 2 GiB before model loading. At most eight snapshots,
three full-logit journal allowances and bounded eight-row exports are reserved
in that estimate. Actual API state sizes and target/sidecar uploaded weight bytes
are exported. Native receipts still must account for draft-head conversion,
shared activation/QSA/MTP/window workspaces, tables/rope, pinned PLE, loader
transients, setup peaks, cache-off images and safety reserve. No measured peak or
capacity qualification is claimed from this source estimate.

Compiler/link/ISA, native operator writer/reader dispatch, same-target full-model
rows/state/IDs, OFF restoration/reference and all memory gates remain pending.
Quality requires the qualification owner's preregistered contract. No quality
capture, READY/default admission, performance result or expected throughput is
established. C2..4, caches and graphs remain separate software and native gates.
