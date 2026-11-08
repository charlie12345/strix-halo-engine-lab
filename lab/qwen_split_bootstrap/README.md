# Explicit copied-host model sequence route

This component supplies a concrete host transport/retirement adapter and connects it to the existing HIP distributed model sequence and greedy executors. It is OFF by default and has no ordinary decode, HTTP, profile, kernel registry or numerical-default integration. It is a completed source subset, **not a standalone two-host model bootstrap** and not native qualification.

`make_hip_copied_host_sequence_route` owns four independently opened project `dist::Transport` instances. A caller can supply the real `usb4_host_auto` payload transport and three separately opened bounded TCP control transports. Do not open overlapping USB4 stream sets for independent channels. The factory does not discover endpoints, reconcile a complete model/admission manifest, load a model or admit total process memory. Those remain software work before a standalone model command can be provided.

The returned backend is an owning alias of the complete channel/state owner. Retaining that handle keeps its borrowed channel and connection backing alive even when the separate `actual_owner` field is released; the owner holds the backend uniquely, avoiding an ownership cycle. An explicit NONnative host test compiles the actual factory against declaration-only HIP headers and mock project transports and verifies this destruction order.

The returned sequence transport is accepted by `make_hip_distributed_stage_sequence_setup` with a genuine `HipDistributedStageResources` owner. That existing resource owner selects actual complete-stage weights and optionally typed `local_owned_ranges`; it retains full serial scratch. The full-resource alternative retains all weights and must not be described as weight sharding. The existing sequence executes the actual ordered prefix, crosses the compiled canonical boundary, and commands the actual suffix/head. The existing greedy request consumes the prompt causally one token at a time, then continues through its existing coordinator/terminal cursor and pending-token protocol. No second PP executor, head fixture, new reduction, numerical conversion or fast-prefill path is introduced.

## Lifetime and protocol

Each channel checks a 128-byte little-endian header: version, world two, ordered ranks, epoch, channel ID, monotonically increasing per-direction sequence, bounded payload length, agreed manifest SHA-256 and payload SHA-256. The caller must independently construct and reconcile the manifest from actual source/model/target/environment/cut/capacity/history/frontier/admission bindings. Passing an arbitrary digest is not a commissioning proof. This subset has no complete standalone manifest builder.

The existing copied envelope binds the exact canonical boundary descriptor, dtype, plan, epoch, request, generation, physical route and physical sequence. Producer HIP D2H staging follows actual prefix completion and fences the original stream/device before yielding an owned host source. Receiver H2D copies preserve every canonical payload byte and its order, fence before consumption acknowledgement, and retain the true stage owner. A source can retire only after local host release, peer validated receive and peer completed consumer acknowledgement. No callback return, token ACK or GPU-direct readiness flag substitutes for those events. Logical position dependencies use the existing privately minted participant proof; they do not claim data movement.

The token retirement ledger pins model/staging/channel owners, verifies complete local boundary coverage and seals once. Its privately minted move-only proof is checked by the existing participant before the next input. Host-copy retirement never advertises external DMA or GPU-pool support. Native-link execution keeps its existing separate provider path and rejects a mixed copied/native setup.

On a host I/O exception the channel aborts and destroys its actual transport inside a nested I/O error boundary, before borrowed headers or payload vectors unwind. The project transport destructor joins persistent host workers. Failure poisons the epoch and closes the copied payload and model controls; the greedy channel closes on request failure. Explicit error-only consumer cleanup may prove quiescence, but sends no success acknowledgement and cannot recover the failed transaction. Unknown consumer release is retained or fail-stop. No output is published by a failed sequence, and no reconnect or automatic retry is supplied.

## Host checks

Root CMake provides `strix-copied-host-channel-tests` and `strix-copied-model-retirement-tests`, linked to the actual host core, alongside the existing copied-stream and native-backend suites. Tests use explicitly named mock host transports/consumers; they do not emulate model kernels or qualify a physical provider. They cover checksummed framing, every header-byte mutation and truncation, payload corruption/truncation, rank/epoch/channel/manifest/capacity mismatches, replay, exception-time worker joining, owner retention, five consecutive token-bound backend transfers, stale/foreign retirement bindings, consumer failure, drain and sealed-backend rejection. The separate host CMake file can build normal and NDEBUG test executables against an explicitly supplied actual host core library.

CXX compilation with declaration-only HIP headers is NONnative syntax evidence only. No HIP/CUDA compiler, device, model or network operation is part of the host checks.

## Explicit native build graph

Use the frozen full source commit, not an abbreviated SHA:

```sh
cmake -S . -B build-qwen-copied-route \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON -DBUILD_TESTING=ON \
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=lab/qwen_split_bootstrap/register-native.cmake \
  -DSTRIX_QWEN_COPIED_ROUTE_SOURCE=FULL_SOURCE_COMMIT
cmake --build build-qwen-copied-route --parallel 12 --target \
  strix-hip-qwen-copied-model-route \
  strix-hip-distributed-sequence-model-fixture \
  strix-hip-distributed-greedy-model-fixture
```

The provider library links the real project HIP runtime and host transport library. The fixtures remain callable runners requiring genuine initialized stage and independent reference owners. No host mock is compiled into that provider. The factory reports the compiled source pin and actual transport name. The build hook rejects absent, short or nonhex source pins. A graph check with interface-only mock dependency declarations establishes configure wiring only.

## Remaining software and qualification

A standalone command still needs real model/path/ID loading, identity/config resolution, PLE-safe complete-layer planning, pre-allocation complete owner accounting on both ranks, four endpoint openers, bilateral initial-frontier/admission negotiation, canonical source/environment/model/target/capacity manifest construction, explicit composition refusals, reference/export file binding and exclusive final output publication. Existing stage inventory lists unobserved PLE/backend/driver/container owners; those bytes are not zero. Full scratch, retained mappings/full weights if present, setup peaks, snapshots, logits, histories, staging, transport and external pools must all be billed. Unknown totals must decline, preserving the full requested context and slot commitments. This adapter does not establish total UMA fit or fill these gaps with caller readiness booleans.

Independent scalar-reference full-vocabulary raw bytes/IDs, all seven canonical components and per-rank history/pending/frontier remain required for the complete native model gate. Existing independent-storage serial-kernel fixtures are useful execution checks, but are not an independent scalar oracle. The end-to-end standalone gate/export runner remains software work. Compiler, native failure/lifetime, complete-model equality and complete-workload gates have not been run. Prefix reuse, sessions, MTP, sidecars, batched/chunked prefill and option compositions remain unqualified.

Performance is UNMEASURED. C1 prefix + wire + suffix is sequential and may increase latency. Pipeline filling requires an independently proven scheduler. No tok/s, percentage floor, speedup or quality claim follows from this source subset. After complete correctness, any timing study needs distinct A/A controls and interleaved ABBA, the conventional even median, identical model/state/capacity, actual kernel/IOMMU receipts, bounded cleanup and no automatic retries. Slower or anomalous results return to implementation/harness study first.
