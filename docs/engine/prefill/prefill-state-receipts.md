# Prefill numerical-state evidence interchange

`scripts/qualification/prefill_state_receipt.py` validates a versioned evidence
interchange contract. It launches nothing and exports no engine state. There is
**no engine exporter or complete device adapter for this JSON schema**. A
`VALIDATED-RECEIPT` means that supplied adapter assertions satisfy the contract;
it does not establish native qualification or authenticate artifacts. Missing
state, an unavailable adapter, or unverified native plan preservation means
`NO-STATE-EQUIVALENCE/UNSUPPORTED`. Tokens or logits alone never satisfy it.

The two evidence classes are separate:

| `evidence` | Required requested and effective route | Meaning |
|---|---|---|
| `session-components` | `greedy-tiled-full-row-session` | Seven numerical component groups, complete frontier, and canonical target state. Session envelope metadata is validated locally, rather than compared as numerical state. |
| `target-state` | `automatic-cli` | Existing prompt/final canonical target-state stderr plus complete generated IDs, conditional on native observer plan preservation. This is not seven serialized component hashes or complete session-envelope equality. |

Session equality cannot substitute for automatic-route equality. Session save/load
admission requires greedy tiled full-row execution and cache units of at least
1024 rows. MTP sessions require additional fixed-MTP constraints; this interchange
version deliberately supports **non-MTP, single-request greedy execution only**.
HTTP, continuation/load, early-stop/short generation, predictor state, other state
storage layouts, and automatic seven-payload export are unsupported here.

## Common receipt and identity fields

State receipts have exactly these root fields: `schema` = `prefill-state-v2`,
`evidence`, `timing_eligible` = false, `adapter`, `nodes`, `configuration`, `prompts`,
and `cases`. Unknown fields are rejected. JSON duplicate object keys, non-object
roots, duplicate cases and missing/extra cases fail. Digests are lowercase SHA-256
hex strings unless specified otherwise; identities of width 16 are zero-padded
unsigned 64-bit hexadecimal strings. Booleans are never integer values.

`adapter` has exactly `id` (nonempty producer/version identifier) and `source_id`
(40- or 64-hex source revision). `nodes` maps exactly two distinct private node IDs
to objects with nonempty string `gpu_arch`, `runtime`, `driver`, `kernel`, and
`iommu_perfopt`. Record observed values; do not infer them from requested settings.
The two nodes execute independent world=1 requests; this is **not TP2**. Equality
is required separately on each node, with no cross-node bit equality assertion.

`configuration` is an exact, required common projection. Every case repeats it
and must equal it. It explicitly excludes binary/source identity, physical execution
hashes, tile size, diagnostic controls, and cache family/storage namespace hashes.
Both binaries may differ physically. All required fields follow:

| Fields | Contract and source mapping |
|---|---|
| `model_sha256`, `assets_sha256` | Full model content hash and map of stable auxiliary-asset roles to full content hashes. Empty assets only if none are used. Derived from loaded/mapped assets and checked for changes, rather than filenames. |
| `vocab_size`, `generate_tokens`, `context_tokens` | Positive uint32 values from the model contract and resolved request. Complete requested generation must fit alongside each prompt. |
| `slots`, `world`, `rank` | Integer 1, 1, 0 respectively. |
| `decode_profile`, `prefill_profile`, `profile_pair_identity` | Nonempty resolved profile names and the numerical profile-pair identity as 16 hex digits. |
| `rotary_sha256` | Hash of the canonical resolved rotary descriptor JSON. |
| `prefill_identity_hash`, `prefill_options` | Active numerical prefill identity as 16 hex digits and sorted unique active option names. |
| `execution_policy_sha256` | Hash of the adapter's canonical effective semantic policy JSON, described below. |
| `full_attention_layers` | Sorted unique nonempty uint32 layer IDs derived from the actual model contract, including every full-attention layer. |
| `tail_rows` | Ordered distinct positive tail sizes, each below 8192; the same tails must be executed by every case. Empty means no configured tiled tails. |
| `sampling`, `mtp`, `kv_format`, `recurrent_storage` | Exactly `greedy`, false, `bf16`, `key_value`. |

Adapter-owned canonical JSON uses sorted object keys, ASCII JSON escaping, no
whitespace, no non-finite numbers, and UTF-8 bytes. The policy projection includes
all effective numerical/execution CLI and environment controls, including defaults,
capacity, budgets, greedy/EOS controls and storage formats. Omit output destinations,
per-run prompt and model paths (content identities bind those separately), build/source
IDs, physical compiled/cache identities, and the separately declared tile size.
Exclude only audited observer and cache-storage namespace settings from this semantic
projection; retain their original records in the private adapter evidence. Do not
filter unknown numerical/execution controls. Version this normalization in `adapter.id`;
an adapter must refuse unmapped controls rather than silently omit them. This is an
interchange normalization, **not** the engine's `session_numerical_config` or a
claim that its storage identity can be reused across binaries.

`prompts` contains exactly keys `8192` and `32768`, each the SHA-256 of the complete
actual input IDs encoded as unsigned little-endian uint32 words. It identifies the
input actually consumed, rather than a requested prompt path.

## Exact case set and native plans

For **each** node and **each** arm (`ref`, `candidate`), require exactly these
`(length, tile_rows)` variants: `(8192,8192)`, `(32768,8192)`, `(32768,16384)`.
There are exactly 12 cases. A tile label alone does not establish admission: the
adapter must verify the admitted native plan; a fallback is not that declared case.
If a resolver cannot admit a requested combination on the automatic route, report
unsupported rather than forcing a session route or relabeling an explicit route.

Each case has exactly `node`, `length`, `tile_rows`, `arm`, `build`, `rc`, `declined`,
`rejected`, `requested_route`, `effective_route`, `configuration`, `prompt_sha256`,
`plan`, `token_ids`, `stderr`, `stderr_sha256`, `diagnostics`, and `evidence`.

`build` contains `binary_sha256` and `source_id` (40 or 64 hex). Require the same
build/source pair for a given arm across both nodes and every schedule. `rc` is
integer zero, `declined` and `rejected` are false. `prompt_sha256` matches the
length's root prompt identity. `token_ids` contains exactly `generate_tokens`
unsigned integer vocabulary IDs in emission order, including the unconsumed final
pending selection. No truncated output is accepted. `stderr` is the complete original
UTF-8 stderr, whose byte hash must equal `stderr_sha256`.

`plan` has exactly `tile_rows`, `tail_rows`, `context_tokens`, `slots`, and these
nonzero 16-hex native identities: `decode_execution_identity`,
`decode_kernel_plan_identity`, `prefill_execution_identity`,
`prefill_kernel_plan_identity`, `tiled_execution_identity`,
`tiled_kernel_plan_identity`. Use admitted values, not inspection's pending plans.
Its tile, tails and capacity must match the declaration. Physical identities are
recorded and compared **only within a binary/schedule's observer-on/off control**;
they are never numerical equality requirements across T8192/T16384 or binaries.

`diagnostics` explicitly reports all keys in the validator's `DIAGNOSTICS` tuple.
For untimed state cases, `STRIX_PROMPT_STATE_DIGEST` is string `1` and all other
values are null (environment variable unset). The adapter must audit the complete
environment and CLI for other diagnostics too; it must refuse unrepresented active
diagnostics. Null does not mean string `0`. This inventory is an adapter requirement,
not something the receipt validator can discover on another process.

## Canonical target-state lines

Both classes require exactly one existing `qwen-target-state phase=prompt
position=N sha256=HEX` line followed by exactly one `phase=final` line, with no
errors, duplicates, extra phases or malformed state lines. Other stderr lines are
retained. Prompt position must equal `length`; final position must equal
`length + generate_tokens - 1`. Complete token IDs and both canonical state records
must agree across both binaries and both schedules at each length on each node.

`QwenTargetStateDigestFrame` includes component IDs, occupied byte sizes and hashes,
ordered consumed token history and pending selection. The frozen target registry
covers attention KV and sparse-index rows in every full-attention layer, recurrent
matrix, GDN and PLE convolution, ngram history and position history. It excludes
executor/window lineage and predictor-only extensions. It is therefore distinct
from `session final_digest`, whole-envelope hashes, cache identities and logits
digests. Do not put those values into the target-state lines.

For `target-state`, per-case `evidence` has exactly `observer` =
`STRIX_PROMPT_STATE_DIGEST=1`, `native_plan_verified` = true,
`native_evidence_sha256`, and `observer_off`. The latter has exactly `node`, `build`,
`prompt_sha256`, `configuration`, `plan`, `requested_route`, `effective_route`, and
`diagnostics`; both routes are `automatic-cli`, all diagnostics are null, and
node/build/prompt/configuration/plan must equal the
observer-on case. The private native evidence artifact must establish actual
observer-off/on execution plan and effective-config preservation on that binary,
node and variant. A hand-written true or hash is not native evidence.

The existing plain CLI observer can create a null-store registry without session
save/load. It is absent from `explicit_execution_control`. Existing tests distinguish
execution routing from cache-family/storage admission: observer namespace changes
can decline automatic cache reuse without proving that the execution plan changed.
Native plan preservation remains a required, unverified gate. A missing registry,
missing state line, route change or unavailable native plan evidence is unsupported.
Automatic seven-payload export remains unavailable; this alternative never claims it.

## Session component evidence

For `session-components`, per-case `evidence` has exactly `payloads`,
`snapshot_sha256`, `snapshot_validation`, and `frontier`. `snapshot_sha256` identifies
the original complete envelope, but is not compared between arms. The adapter must
parse and validate the actual envelope with each arm's own executable, model/assets,
numerical config, observed device, compiled plan and state layout. Its
`snapshot_validation` must explicitly assert `envelope`, `compatibility`,
`range_census`, and `metadata` are all true. CPU preflight alone does not certify
native/model/compiled-plan compatibility.

`payloads` has exactly the seven keys enumerated by `SessionStateComponent`:
`attention_kv`, `recurrent_matrix`, `gdn_convolution`, `ple_convolution`,
`sparse_index`, `ngram_history`, `position_history`. Each value is an ordered list
of objects with exactly `name`, `bytes` (positive integer), and `sha256`.
Attention groups contain `kv.L<layer>` for every declared full-attention layer;
sparse groups contain `index.L<layer>` in that same order. Each other group contains
one section with its group's name. Hash all occupied row bytes and each complete
fixed allocation, preserving raw storage bits; exclude inactive capacity/padding.
The adapter must reconstruct and validate exact source-derived sizes and every
section against the native registry/layout, rather than trusting this JSON census.
Extra/missing/duplicate layers and arbitrary sets of seven names fail.

`frontier` has exactly integer `consumed`, `emitted`, `accepted`, `predictor`,
`pending`, string `sampling`, and `history_sha256`. Consumed and accepted equal the
final state position, emitted equals the full generation count, predictor is zero,
pending equals the last generated ID, sampling is `greedy`, and history hash covers
the complete consumed uint32-LE token history. All payload section names, sizes,
hashes and the full frontier must match across the comparison group.
`session.metadata` is required and validated by the adapter, but its physical
layout/executor/lineage bytes are not compared across binaries or tile schedules.
This establishes a numerical-payload comparison, not byte equality of the envelope.

## Separate timing structure

`--timed-abba` accepts only `prefill-abba-v2`, `evidence` = `timing-only`,
`timing_eligible` = true, the same required header identity fields, and `arms`
instead of `cases`. Each node must have one or more complete ordered ABBA quartets,
checked over its subsequence in the receipt. No trailing junk arms are allowed.

Every timed arm has exactly `node`, `arm` (`A`/`B`, corresponding to ref/candidate),
`build`, `rc`, `declined`, `rejected`, `configuration`, `length` = 32768,
`prompt_sha256`, `requested_route`, `effective_route`, `plan`, `diagnostics`, and
`elapsed_ms`. Both routes are `automatic-cli`; all diagnostics must be explicitly
null and all other diagnostic CLI/environment controls unset. Plans use T16384 and
the common tails/capacity, with stable native identities for each binary on each
node across repeats. Configuration and workload identities are mandatory for
every arm. `elapsed_ms` is finite, positive and non-boolean. This checks structure
only, produces no performance verdict and never supplies state equality evidence.

## Host verification and source contracts

Run `python tests/prefill_state_receipt_tests.py`. Synthetic fixtures exercise
both evidence classes, parsing and CLI rejection paths; they are not live receipts.
CLI exit status is 0 for contract validation, 1 for unsupported/invalid evidence,
and 2 for input/JSON decoding failure. Failures print
`NO-STATE-EQUIVALENCE/UNSUPPORTED`; timing success still states no state equality.

Source contracts:

- [Session components and physical layout](../../../include/strix/session_state_layout.hpp),
  [snapshot policy and ranges](../../../include/strix/session_snapshot.hpp),
  [envelope compatibility and frontier](../../../include/strix/session_state_cache.hpp).
- [Registry coverage, target digest and session capture/restore](../../../apps/hip_prefix_state_cache.hpp),
  [canonical numerical frame](../../../include/strix/qwen_target_state_digest.hpp).
- [Startup route admission and CLI prompt/final emission](../../../apps/hip_decode_token_main.cpp),
  [observer ownership predicate](../../../apps/hip_decode_options.hpp),
  [resolver, inspection fields and execution-control classification](../../../apps/qwen_effective_config.hpp).
- [Snapshot range/identity tests](../../../tests/session_snapshot_tests.cpp),
  [occupied ranges and transfers](../../../tests/session_snapshot_transfer_tests.cpp),
  [observer/cache identity tests](../../../tests/qwen_effective_config_tests.cpp),
  [target-frame tests](../../../tests/mtp_step_logits_digest_tests.cpp).
