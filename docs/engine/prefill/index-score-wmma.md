# FAST prefill index score candidate

`index_score_wmma` is an explicit candidate, outside READY and absent from the
FAST profile defaults. Enable `STRIX_PREFILL_INDEX_SCORE_WMMA=1` before startup.
Its identity token is `idxscore16`; `index_proj_wmma` retains `idxproj16` and its
existing arithmetic. These options can be measured independently or together.
Default and plain FP16 identity hashes remain `ee2cd64684149069` and
`a0094634843e16ee`. No native performance or model qualification is claimed.

OFF controls: `STRIX_PREFILL_INDEX_SCORE_WMMA=0`,
`--prefill-disable index_score_wmma`, or config
`prefill_disable = ["index_score_wmma"]`. The shared exact-FP16 master also
disables it. Restart after changing server-level controls. The option is reported
by the shared prefill active-set identity, effective config and capabilities.
MTP and nested admission remain unqualified for either index WMMA candidate.

## Existing projection coverage

`idxproj16` covers only the BF16 index query/key input projections, before
normalization, RoPE and pooled-key construction. It does not cover index scores
or selection. `Qpf2kIndexScope` admits ordinary prefill mode 0 with
`config.qsa_fast_prefill`, explicit projection opt-in, gfx1151 wave32 and no
index-row suppression. Selecting the FAST profile alone does not opt in.

`qpf2k_index_wmma_layout` accepts hidden width 2560 with either 512 query plus
128 key outputs, or the 128-key-only skipped-query prefix. The projection path
requires BF16 matrices, 16-byte aligned input rows, complete 64-row groups and
2-byte aligned weights. A T16384 chunk consists of 256 eligible groups; a 32K
prompt has two such chunks when the resolver admits that owner. A change of
matrix set flushes the pending descriptor batch, including skipped-query groups.
Other hidden widths, output layouts or weight kinds decline WMMA; unaligned
inputs and 8/16/32-row tails keep their existing exact projection route. There
is no partial K or N tile for the admitted 2560-by-512/128 layouts. Pool build,
norm/RoPE and selection remain separate stages.

## Arithmetic and admission

`qsa_flash::index_score_wmma_kernel` computes 16 query rows by 64 pooled keys
per CTA. Inputs are already rounded BF16 values from query norm/RoPE and the
pooled-key producer, read from FP32 containers or the packed BF16 key image.
Each head dot has eight native BF16 K16 panels with zero C, added in ascending
order in FP32. ReLU is applied to each head separately, followed by the existing
ascending head sum and division by sqrt(128). No native accumulator feedback,
split-K, atomic sum, new operand rounding, or budget adjustment is introduced.

The ordinary FAST prefill mode-0 part-2 scope admits gfx1151 wave32, BF16 KV,
four index heads of dimension 128, a valid pooled-key owner and eligible
16/32/64-row groups. Stage batches and compact score-ring slices use the same
kernel and descriptor lifetimes. Ragged key columns and causal per-row block
counts are masked; dense-prefix rows do not read skipped query projections.
Other architectures, non-FAST routes, diagnostic modes, small windows and
unsupported layouts retain the existing scorer. Index-row baseline/GSQ
suppression also declines the candidate. The existing QSA scorer mask and
stage reservation guards still apply. The active launch prints an `idxscore16`
witness; a requested identity alone is not evidence of execution.

The existing selector and its descending-score/ascending-block tie rule are
unchanged. Changed FP32 score reductions can change the selected set, including
ties at the cutoff. Group maxima are produced from the candidate scores.

## Fixture and bounded node ticket

Build this standalone fixture without linking or rebuilding the full engine:

Build the binary outside the timed GPU turn (no GPU needed):

```sh
mkdir -p build-index-score
c++ -O2 -std=c++20 -ffp-contract=off -Iinclude tests/qsa_index_score_host_tests.cpp -o build-index-score/host
build-index-score/host
timeout 180 amdclang++ -O3 -std=gnu++20 -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee --offload-arch=gfx1151 -Iinclude -x hip tests/qsa_index_score_wmma.hip -o build-index-score/device
```

The GPU ticket itself (run by the operator under `timeout -k 10 470`):

```sh
uname -r
sha256sum build-index-score/device
git rev-parse HEAD
STRIX_QPF2K_SCORE_ROWS=auto timeout -k 5 220 build-index-score/device --time
STRIX_QPF2K_SCORE_ROWS=auto timeout -k 5 200 build-index-score/device --time --packed
```

`--time` runs every correctness check and prints the `INDEX_SCORE` line before
it times anything; a nonzero exit voids that run's timing. Inner timeouts total
420 s plus kill grace, inside the 470 s outer bound and the eight-minute turn.
No model is required. The `--packed` run covers the packed BF16 pool image. Record running kernel, `amdgpu.iommu_perfopt`,
compiler, binary/source identity and clock/power telemetry with private results.

The host fixture checks OFF/master precedence, pinned identity bytes, per-head
ReLU, a FP64-panel oracle and selection/tie statistics. The device fixture uses
synthetic BF16 queries/keys for both T16384 chunks of a 32K prompt (positions 0
and 16384), compression ratio 4 and budget 1536 tokens (384 complete blocks).
It compares all scored entries to the existing FP32 scorer, checks every
score/maxima canary, samples independent CPU trees/panels and checks every
selected token/tail and guard against the host's score-descending/block-ascending
oracle. Repeated keys and zero-query rows exercise ties. The printed metrics are
maximum absolute score difference, rows with any selected-set change, and
mean/maximum symmetric difference in both blocks and tokens. Means include all
16384 query rows, including unchanged dense-prefix rows. Small 16/32-row
single-group cases additionally check ragged key columns and both pool images.

`--time` runs two warmups per arm followed by seven interleaved A/B pairs with
alternating order. It prints individual pairs and medians for scoring alone
(including maxima), then scoring plus the unchanged selector. It excludes
projection, pool construction, transfers, CPU validation and full-model work.
For a 32K projection, sum the two chunk medians and multiply by the number of
executed QSA layers; verify the layer count with a production trace. This is an
index-stage estimate, never a full-workload speed claim. Native checks remain
pending until the fixture actually runs on the target device.

## Qualification inputs

Freeze the numerical profile, 1536-token QSA budget, model assets, chunk geometry,
state and active option mask. Qualify score-only (`idxscore16`), projection-only
(`idxproj16`) and their composition separately against the same frozen reference.
Score-only changes block scores, group maxima and potentially selected token IDs,
attention outputs and downstream logits/state. It does not directly change
index projections, normalized queries, pooled keys or raw index state; projection
WMMA additionally changes those projection outputs and their consumers.

Use the fixture's per-row selected-set symmetric difference and changed-row count
as diagnostics; they are not a fidelity verdict. Collect raw head-dot ReLU
crossings and cutoff flips on real inputs, then full-vocabulary forced-history
delta NLL, KL(reference||candidate), top-1 disagreement, overlap/TV and task
outcomes on frozen evaluation sets, with simultaneous confidence bounds. Compiler/ISA, native execution, model fidelity
and complete-workload gates must all pass before promotion. OFF restoration
needs complete token/logit and committed-state comparison against the reference;
host identity pins alone do not establish native output equality.
