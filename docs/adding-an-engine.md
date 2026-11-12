# Adding an engine

*Español: [adding-an-engine.es.md](adding-an-engine.es.md)*

A model family is added by writing what is specific to the model and adapting it to
the shared layers. Serving, sessions, caches and scheduling are not written again.
This page lists what a family implements, what it receives, and how backends are
kept apart. A complete, compiling example is in `examples/skeleton_engine/`.

Background: [engine/session-layer.md](engine/session-layer.md) (contracts),
[engine/backends.md](engine/backends.md) (Backend seam),
[models/model-families.md](models/model-families.md) (family layout rules).

## The example

`examples/skeleton_engine/skeleton_engine.hpp` is a family in one file. Its "model"
is a counter, so the file contains only the adapter code that every family needs.
It is built as `strix-skeleton-serve` and tested by `skeleton_engine`
(`tests/skeleton_engine_tests.cpp`), which sends a completion and a chat request
through the production `strix::serve::run()` path with the CPU reference backend.

```
cmake --build build --target strix-skeleton-serve skeleton_engine_tests
ctest --test-dir build -R skeleton_engine
build/strix-skeleton-serve --registry-root examples/skeleton_engine \
    --registry models/registry.toml --model skeleton --port 8080
curl -s localhost:8080/v1/completions -d '{"model":"skeleton","prompt":"hi","max_tokens":4}'
```

## What a family implements

| Piece | Where | In the example |
|---|---|---|
| Model description | `models/<model>.toml` and a row in `models/registry.toml` (`name`, `family`, `description`; a production row also names its CMake option, profile tag, defaults and adapter files) | `examples/skeleton_engine/models/` |
| Model adapter | A class deriving from `strix::session::Engine` (`include/strix/session/engine.hpp`) | `skeleton::Engine` |
| State hooks | `strix::session::Hooks` given to `KvStore`: initialize, reset, release, capture, view, restore, truncate | `Engine::hooks()` |
| Tokenizer | `serve::BpeTokenizer`, or any `serve::ApiTokenizer` (`include/strix/serve/tokenizer.hpp`) | `ByteTokenizer` |
| Renderer and parser | `ApiConfig::chat_renderer`, optional `output_parser`, `stop_tokens` (`include/strix/serve/api.hpp`); shared templates live in `include/strix/serve/chat_template.hpp` | the lambda in `Service` |
| Service | A `serve::SessionService` that owns backend, engine, queue, `SessionRuntime` and `ApiService` (`include/strix/serve/run.hpp`) | `skeleton::Service` |
| Registration | `install_session_service(ServiceFactories&)` under the registry `family` value, referenced from `install_service_factories` in `src/serve/run.cpp` behind the family's CMake option | `install_session_service` |

### The Engine contract

Required:

- `prefill(tokens, state)`: consume the tokens into the slot. The call must consume
  all of its input, so that a snapshot taken afterwards sits exactly at that frontier.
- `decode(state, sampler)`: produce one or more tokens. The newest token stays
  *pending* (produced, not yet consumed as a row). The result reports the frontier
  before and after, and a finish reason. A step is terminal when it reaches a stop
  token (`stop`), uses up `sampler.max_tokens` (`length`), fills the context
  (`context`) or is cancelled; the runtime rejects a result that breaks these rules.
- `caps()`: context, slots, batch width and which optional features exist.
- `receipt()`: the execution receipt reported with every response.

Optional, each announced through `caps()`: `prefill_batch` and `decode_batch`
(several sessions in one forward), `advance` (mixed prefill and decode rows in one
cycle, with `RowHints`), `fork_lane`, `cancel_lane`, `resume_lane`,
`append_external` (lanes of one parallel job), `commit_pending`, and
`RecurrentStateOps` (`include/strix/session/recurrent_state.hpp`) for models whose
state is not only attention KV.

### Backend operations

A family asks a `session::Backend` (`include/strix/session/backend.hpp`) for:

- buffers (`allocate`), queues (`create_queue`), events (`record`, `wait`,
  `synchronize`);
- host and device copies (`upload`, `download`, `copy`) and `memory_budget()`;
- kernels through the operation registry: `supports(key)` and `query(key)` answer
  whether an operation exists for a data type and format, `dispatch` runs one, and
  an `ExecutionPlan` runs a prepared sequence. A missing operation is reported as
  `UnsupportedOperation`, never as a silent fallback.

Dispatch is per operation, on the host. There is no virtual call inside a kernel.

## What a family receives

Provided by the session and serve layers once the pieces above exist:

- **Serving**: `strix-serve --model <name>`; the OpenAI-compatible front end
  (`/v1/completions`, `/v1/chat/completions`, `/v1/messages`, `/v1/responses`,
  `/tokenize`, streaming, tools, structured output), health, status, metrics, drain
  and authentication.
- **Sessions and slots**: the request owner (`serve::SessionRuntime`) leases slots,
  schedules prefill and decode, batches sessions up to the reported width, handles
  cancellation and stop conditions, and checks every decode result.
- **Prefix cache**: longest-common-prefix restore and the end-of-turn snapshot that
  makes the next turn prefill only the new message. It is a facade over the single
  `PrefixStateCache`, so there is one byte budget and one durable writer
  (`include/strix/session/prefix_cache.hpp`, `checkpoint_io.hpp`).
- **Row budget**: `RowHints` and `EffectiveRows` carry the per-cycle row budget of a
  cohort to the engine and back; a family that implements `advance` gets mixed
  prefill and decode cohorts from the shared scheduler.
- **Parallel jobs**: fork, per-lane cancellation and batched lanes
  (`include/strix/session/parallel_job.hpp`).

Shared libraries that a family calls instead of copying (see "Shared code first" in
`AGENTS.md`):

- **Two-node pipelines and transports**: `include/strix/pair` (link, lockstep, pipe,
  balance) and `include/strix/dist` (transport selection, GPU-direct and host copy).
- **Kernels**: `src/core` (linear, integer and packed plans), `src/moe_core` (MoE
  routing and grouped GEMM), `src/attn_core` (flash attention tiles, masks, online
  softmax).
- **Drafting**: `include/strix/spec` and the speculation controller and verifier
  headers. A family that supplies a draft source reports it through `caps()` and the
  accepted and proposed counts of `StepResult`; drafting is on by default where the
  family qualifies it.

## Backends

| Backend | Files | Built when | Selected by |
|---|---|---|---|
| CPU reference | `include/strix/session/backend_cpu.hpp`, `src/session/backend_cpu.cpp` | always (part of `strix_session`) | the service constructs `session::CpuBackend` |
| HIP | `include/strix/session/backend_hip.hpp`, `src/session/backend_hip.cpp` | `STRIX_ENABLE_HIP_RUNTIME=ON` (`strix_session_hip`) | the service constructs the HIP backend |

The service is the only place that names a backend: in the example it is the
`ss::CpuBackend backend_;` member, and the engine takes a `session::Backend&`.
A family that follows this pattern needs no accelerator header in its adapter.
`backend_hip.hpp` itself exposes no HIP type; the HIP runtime is included only by
`src/session/backend_hip.cpp`.

### What still leaks

The seam covers buffers, queues, events, copies, the memory budget and operation
dispatch. Today it is the storage and scheduling boundary, not yet the kernel
boundary:

- The production kernels of every family are HIP translation units (`.hip`) that
  take HIP streams and device pointers directly. Their operations are not yet
  registered in the Backend operation registry, so a family cannot run its real
  forward pass on another backend by changing the constructed backend.
- The shared kernel cores (`src/core`, `src/moe_core`, `src/attn_core`) and the
  GPU-direct transports include the HIP runtime directly.
- The session adapters of the migrated families use `session::Backend` for session
  state, while their engines keep HIP types for model execution.
- The CUDA path is a separate set of sources (`src/cuda`, `cmake/QwenCuda.cmake`)
  that does not implement `session::Backend` yet; `engine/backends.md` maps it.
- Families that are not migrated own their request loop and caches and do not use
  the seam at all.

The CPU backend is a correctness reference for the contracts and for host tests. It
is not a production inference path.

## Checklist

1. `models/<model>.toml`, registry row, CMake option `STRIX_FAMILY_<NAME>`.
2. Model code under `src/<family>/`: tensors, numerics, kernels that are truly
   model-specific. Reuse `src/core`, `src/moe_core`, `src/attn_core`.
3. `session::Engine` adapter and `KvStore` hooks; report capabilities truthfully.
4. `SessionService` and `install_session_service`; no `main()` and no request loop.
5. Host test with the CPU backend or a scripted engine, in the style of
   `tests/skeleton_engine_tests.cpp`; device exactness gate for the real kernels.
6. Documentation in `docs/models/` and a row in the family table.
