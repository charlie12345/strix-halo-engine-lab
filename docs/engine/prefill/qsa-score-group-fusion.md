# QSA score group pruning: host design

Status: host oracle only. No production launch, setting, default, numerical
identity or qualification band changes. Native arithmetic, resources and timing
remain untested. The reference for this organization is the `idxscore16` score
candidate with the same active option set. It is not an equality claim against
the ordered FP32 scorer.

The token budget stays 1536: compression ratio 4 selects 384 complete blocks,
then appends the visible partial-block tail. Expert execution is unchanged.

## Contract and proof

Use the candidate's existing FP32 score expression without changing operands,
K16 panels, zero C, ascending panel/head additions, per-head ReLU or final
division. The first pass retains the four-block maxima with the existing
XOR-1/XOR-2 tree and `-inf` padding, while omitting the full block-score image.
Do not substitute native C feedback or another dot-product reduction.

Rank groups by `(select_key(maximum) descending, group ID ascending)`. Retain
at most 384 groups, in ascending group order. Recompute their four scores with
the identical candidate expression and original block IDs; the compact list
has at most 1536 entries. Select 384 blocks by the existing score-key descending,
block-ID ascending rule. Emit blocks in ascending ID order, expand each to four
tokens, then append the unchanged partial-block tail.

Proof: for any block in a discarded group, each of the 384 retained groups has
a representative whose score key is greater, or whose equal score key comes
from an earlier group. Every block in an earlier group has a smaller ID than
every block in a later group. These 384 distinct representatives precede the
discarded block. It cannot belong to the first 384 blocks. If fewer than 384
groups exist, retain them all. This also covers equal scores and signed zeros
(`select_key` canonicalizes zero). Merely retaining one block per group is
incorrect: multiple winning blocks may come from the same group.

This differs from retaining every group at a scalar maximum threshold: group
ID participates in cutoff ties, so the candidate list remains bounded even
when all scores tie. Dense rows with at most 384 complete blocks emit their
visible tokens directly and consume neither queries nor scores.

The initial admission contract declines nonfinite or negative scores to the
unchanged materialized candidate path. The host model records that decision
before publishing compact candidates. Any eventual device guard must be
uniformly resolved before selection. It must retain the full fallback
reservation; allocation failure cannot silently reduce context or capacity.

## Possible device organization and costs

There is no grid-wide synchronization inside a score epilogue. A possible
implementation needs ordered launches:

1. Score to maxima only, keeping the current candidate arithmetic and mapping.
2. Choose the ordered top 384 groups per row from those maxima.
3. Recompute retained scores, then run compact selection and token expansion.

The first pass still computes every score. A 64-key tile contains only 16
four-block groups, so truncating each existing tile to its top 384 groups
cannot remove any writes. A per-tile local top-k is not a solution by itself.

The candidate currently shares each 64-key tile across 16 rows. Retained groups
differ between rows. Rescoring their union preserves the dense tile mapping but
can repeat almost the full score calculation; per-row sparse launches instead
lose query/key reuse and need new mapping/resource evidence. Neither form is
assumed faster. A compute-only fixture is a diagnostic, not a measurement of
this multi-pass design. Require a store-cost signal and candidate-specific
score-plus-selection timings before implementing the production route.

Logical compact storage is at most 384 group IDs and 1536 FP32 scores per row,
plus the full maxima image. Keeping compact scores in LDS could avoid another
global image, but needs its own occupancy and spill audit. These are bounds,
not measured memory traffic or admitted arena savings.

Existing score tracing observes both the pre-selection image and selected
scores overwritten with `-inf`. Initially keep tracing/diagnostic consumers on
the materialized path. Do not point them at an absent image or claim its digest
was verified. Any later trace support must reproduce those exact logical bits.
Preserve score-ring leases, pinned descriptors and stream/event retirement
through the last selector reader; selection must finish before attend consumes
its indices. No shared ring/profile/default-table redesign is required here.

## Host check and future native gates

Build the standalone host contract test without changing the build registry:

```sh
c++ -O2 -std=c++20 -ffp-contract=off tests/qsa_score_group_fusion_host_tests.cpp -o qsa-score-group-host
./qsa-score-group-host
```

The independent full-image oracle uses numeric score order and ascending block
IDs. The group plan uses `select_key`. Tests compare complete token lists,
tails, retained score bits, maxima keys and the logical post-selection image.
They cover dense/cutoff rows, tile/group/ring boundaries, all-equal groups,
ULP-separated scores, clustered winners, both zero signs and explicit invalid
input fallback. This does not emulate native WMMA rounding or prove a device
implementation race-free.

Before any production admission, on each target node separately:

- Compile strict floating point and audit every specialization for scratch,
  spills, LDS and operand/output mapping. Preserve native candidate score bits.
- Compare every raw score and maximum, complete selected token lists and
  logical post-selection bits, with poison/canary, ragged rows, both key images,
  partial groups, 1536-budget seams and deterministic reruns. Mutations must fail.
- Run complete-workload token IDs, raw logits and canonical state digests against
  the same materialized `idxscore16` active set, model, capacity and schedule.
- Verify the organization OFF arm equals that same candidate, and candidate OFF
  retains the existing reference route. Restart cached policy readers.
- Time interleaved score-plus-selection and full-workload arms with route
  witnesses, available headroom and fallback/candidate counts. Record the running
  kernel, IOMMU parameter, source/compiler/binary identities and telemetry.

There is no device ticket for this host-only design. A future kernel requires a
fresh bounded ticket after the correctness and store-cost dependencies resolve.
Existing frozen numerical qualification plans are unchanged.
