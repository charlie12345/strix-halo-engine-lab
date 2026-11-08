# CUDA Qwen BF16 route storage candidate

This explicit experimental candidate changes the storage of routed expert down
outputs. It is OFF by default and unqualified. Compiler, native operator,
same-target execution, model fidelity and complete-workload gates are required
before any production/default or quality claim.

The weights remain the existing UD-Q4_K_XL pack. The CUDA A8 input quantizer,
gate/up projections, SwiGLU, down activation quantizer, Q5_1/Q8_0 down arithmetic,
route weights and ordered FP32 FMA merge are unchanged. Each FP32 down result is
rounded to BF16 with the shared integer RNE helper, stored, and widened exactly
before the existing slot-ordered merge. This rounding defines a new numerical
target. It is not an exactness claim against the original FP32 route target or
the HIP FP16 prefill target.

## Selection and scope

`STRIX_QPF2K_MOE_ROUTE_BF16=1` explicitly requests the candidate.
`Config::moe_route_bf16 = true` also requests it; `false` is the individual OFF
override. `STRIX_QPF2K_MOE_ROUTE_BF16=0` is the environment OFF control.
`STRIX_BASELINE=1`, `STRIX_QPF2K_DEFAULTS=0` and
`STRIX_PREFILL_EXACT_FP16=1` suppress it, including an explicit API request.
The last master is an inherited OFF control name; it does not describe the
CUDA A8 arithmetic as FP16. Resolution freezes at construction; restart to
change target. No profile table or READY registry entry is added.

The implemented candidate applies the same storage boundary in prefill and
target decode. It covers both the tiled MMA down kernel and the R1..R8 dp4a
kernel, including the tiled reference selector. The source supports padded
prefill/chunks and windows, but native row/chunk invariance remains a gate.

Only a single eager target session with `export_state = false` is admitted.
For decode, explicitly set `STRIX_QRTX_GRAPHS=0`. Candidate graphs, cohort,
multisegment execution, MTP/sidecar, prefix-cache reuse, untagged state import,
node handoff, prefill overlap and other expert weight packs decline by name.
Existing parent defaults are preserved. A candidate does not silently switch
off a requested feature. These compositions need candidate-plain versus
candidate-composed complete logit/ID/canonical-state proof before admission.

## Identities and storage

The composed target hashes NUL-framed, ordered prefill, decode and storage names
under `cuda-qwen-target-pair-v1`:

| Component | OFF | Candidate |
|---|---|---|
| Prefill | `cuda-qwen-a8-prefill-v1` | `cuda-qwen-a8-prefill+moeroute-bf16-rne-v1` |
| Decode | `cuda-qwen-a8-decode-v1` | `cuda-qwen-a8-decode+moeroute-bf16-rne-v1` |
| Raw route storage | `cuda-qwen-a8-rawdot-f32-v1` | `cuda-qwen-a8-rawdot-bf16-rne-v1` |

The parent names identify this backend's existing A8 operation order and model
geometry. Mixed phase pairs also have distinct identities; the engine admits
only OFF/OFF or candidate/candidate. The HIP storage name is not reused.
`numerical_identity()` and `route_storage_policy()` expose the frozen selection.
Only candidate `describe()` adds target/storage/qualification/capacity fields;
the OFF receipt format and fields remain unchanged.

Raw route storage has 10 slots x 2560 outputs per row, no new scale. A compact
row occupies 51200 bytes, versus 102400 bytes OFF; both are 16-byte aligned.
Strides are bytes and slot indices count elements of the selected storage.
Candidate entry points validate row count, alignment, actual allocation capacity,
extent and address overflow before enqueue. The original OFF allocation remains
unchanged. No context, session or buffer capacity is reduced.

One down producer and one merge consumer touch each logical raw element per
layer. Logical writer-plus-reader payload is 102400 bytes less per padded row
per layer. This is a source byte census, not measured DRAM traffic, timing,
throughput or a forecast of gain.

Cache identity fields distinguish the composed target even though candidate
reuse is refused. Existing raw export/import APIs do not carry a target and
therefore refuse candidate cache use. `capture_canonical()` /
`restore_canonical()` are a diagnostic, same-engine typed API. They bind target
and a nonreusable process-local engine generation, include complete KV/index,
current GDN matrices, convolution histories, ngram, held row and length, and
invalidate derived pooled keys on restore. They reject MTP and open windows or
chunks. They are not a portable snapshot or node wire format.

## Verification entry points

Host-only checks require no CUDA toolkit or device:

```sh
cmake -S tests/cuda_qwen_route_bf16 -B build-route-host
cmake --build build-route-host -j12
ctest --test-dir build-route-host --output-on-failure
python tests/cuda_qwen_route_bf16/source_audit.py --parent PARENT_SHA --mutants
```

The host adapter executes the actual shared store/load source for rounding and
layout checks. It is not GPU execution evidence. The execution diagnostic is
also compiled as host C++ to check its API/file-writing syntax, without linking
or executing an engine.

The real native CMake targets use production CUDA kernels and driver:

```sh
cmake -S . -B build-route-native -DSTRIX_CUDA_QWEN_ONLY=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-route-native --target qwen_route_bf16_operator \
  qwen_route_bf16_e2e -j12
build-route-native/qwen_route_bf16_operator TABLES route-bf16 PAIR_INDEX
STRIX_QRTX_GRAPHS=0 build-route-native/qwen_route_bf16_e2e \
  MODEL.gguf TABLES ROPE.bin PROMPT.ids N G MAX_POS OUTPUT_DIR candidate
```

`PAIR_INDEX` 0..3 selects Q4_K/Q5_1, Q4_K/Q8_0, Q5_K/Q5_1 and
Q5_K/Q8_0 respectively; omission runs all. Separate processes with
`STRIX_QRTX_DECODE_KERNELS=0` exercise the tiled reference seam. The operator
fixture runs actual GPU producers/consumers with packed weights and compares
whole views, padding and CSR state against an independent operator reference.
Its device rounding probe covers every BF16 high word with tie neighborhoods,
including zero, subnormal, infinity and NaN cases.

The execution diagnostic compares a serial run of its own target with widths
1..8, partial/zero-keep rollback, typed restore and chunk boundaries. It writes
complete raw logits, IDs and canonical state; failure retains mismatching
window outputs. `off` selects the parent target for a separate OFF restoration
gate. This diagnostic does not admit MTP, graph or cohort combinations and does
not establish model quality. The existing full-logit `step()` API can consume
a forced token stream; quality evaluation is outside the scope of this mode.

An optional final diagnostic arm selects `serial`, `window:1` through
`window:8`, or `chunk:R` for R in 1, 7, 16, 17, 257, 1024; default `all`
runs the complete matrix. Each selected arm still generates its own same-target
serial reference. The completion record names the arm, so a shard cannot stand
in for the complete matrix. Use bounded executor turns for configure, build and
execution; a timeout or missing shard is not a pass.

The `tests/cuda_qwen_route_bf16/native-ticket.sh` POSIX payload checks an
explicit full source SHA, `EXPECTED_SOURCE_BRANCH`, clean source and actual
architecture. Configure and `-j12` build, or one selected execution phase,
run inside 460 seconds plus at most five seconds of process-group cleanup;
the executor must cap the complete turn at 470 seconds. Windows uses the same
CMake targets under its executor-owned job timeout. Missing architecture or
executor availability does not authorize a substitute host.
