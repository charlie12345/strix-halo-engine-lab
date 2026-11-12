# Pair prefill and rank-0 continuation

Pair handoff is explicit: set `STRIX_QWEN_PF_PIPE=1` and `STRIX_QWEN_PF_PIPE_DECODE=1`,
then launch one process per rank with `--rank 0|1`. The transport default is
`usb4gpu` with the production size router; `--transport usb4auto` (or
`usb4-auto`) restores host-only. New direct seam bitwise/TTFT gates are pending;
the measurements below used the previous host path. See the
[transport decision](../../coord/transport-routing.md).
Use identical binaries, model assets, prompt IDs, numerical options and capacity.
The default pair chunk is 4,096 rows where admitted. Session save after a
two-node prefill requires a prompt length that is a multiple of the chunk size.
Measured on two nodes over USB4 (release check of the merged code, one fresh pair per run): tokens, full
logits and final state equal the single-node run at 32K, 128K and an odd 32,771-token prompt. Prefill
32K **12.1-12.3 s (~2,660-2,710 tok/s)** vs 19.2 s on one node; 128K **46.5-46.9 s (~2,800 tok/s)** vs 79.8 s.

The resolver runs short requests entirely on rank 0, without a connection or
handoff. Otherwise it pipelines a legal prefix, restores full-model execution
on rank 0, and consumes the suffix before generation. Plain prefixes require
full tiles or a >=16-row multiple-of-eight tail. MTP also respects its actual
owner ladder; its continuation uses the existing predictor/session frontier.
Rank 1 exits after its work. A loaded layer/transfer cost bucket compares serial
and pipeline estimates including handoff. Without one, the provisional crossover
is 28,672 rows: the available 32K/128K end-to-end savings fit crosses at 24,907,
rounded up on the measured 4,096-row grid. Remeasure this threshold with 4,096-row
chunks and the new handoff; it is an estimate, not a short-context speed claim.

Seam packing fences the D2D snapshot before source arenas can be reused by
independent streams. Diagnostic files use an owned copy of authenticated staging
bytes. Final-state wire v2 uses two pinned 16 MiB slots, a separate copy stream,
CRC32C per piece and one completion ACK after all H2D installs. Both ranks must
use this wire version. Memory overhead is 32 MiB pinned host per active handoff.

## Stage balance, seam copies and the pair timeline

`STRIX_PAIR_TIMING=1` prints one `pair-timing` line per chunk and rank and a per-rank sum
(the shared `strix::dist::PipelineTimeline`): chunk span, seam acquire wait, producer fence,
seam pack/load, and the progress thread's D2H wait, CRC, wire, ACK wait, header wait and H2D
wait; the state handoff is the last row. Spans are host clocks of blocking calls; the
producer fence is where the host waits for the chunk's GPU work. Off, it adds no work.

Measured with it (4,096-row chunks, 32K, two nodes, host USB4 path): per chunk the
seam costs CRC 4-5 ms, wire ~18 ms for 105 MB and an ACK wait of ~4 ms; rank 0 never waits for
a frame slot. The seam is therefore not the limit. What is: stage A (layers before the
cut) also runs the embedding/PLE prelude, so with the even 24/24 split it is the slower stage.
The default cut is now **23** (`cut-source=measured-default`); `STRIX_QWEN_PF_PIPE_CUT=24`
restores the even split. Cut sweep (logits bitwise equal to the single-node prefill in every
arm): 32K 24: 12.44/12.51/12.50 s, 23: 12.08, 22: 12.31, 21: 12.63/12.65, 20: 13.00;
128K 24: 49.36/50.15 s, 23: 47.55/47.53.

The seam pack (rank 0) and scatter (rank 1) moved each chunk's rows with 3 x 4,096 small
device copies (~27 ms per chunk on each rank). They now use one strided copy per component
(same bytes).

The remaining distance to 2x is structural for a two-stage layer pipeline: wall time is
about (single-node time with the pair's chunk size) / 2 x (1 + chunk / prompt), plus the
state handoff (0.11 s at 32K, 0.4 s at 128K). A different first or last chunk size does not
shorten it (the largest chunk sets the fill plus drain), and smaller uniform chunks lose
more per-node efficiency than they save (single node at 32K: 4,096-row tiles 20.6-21.1 s
vs 19.0-20.0 s with 16,384-row tiles, all bitwise equal).

## Prompt-bound MTP golden and replay

Keep the qualified asset/provenance/window argv in private
`assets/mtp-qualified.argv`, exactly one argument per line. It includes `--mtp`
and all required sidecar/target pins, policy and device budgets, but excludes the
model, prompt, pair flags, golden options and the two host-file budgets below.
Preserve its numerical options for every command. No private asset values are
published here. These Bash arrays spell the complete launch assembly:

```bash
mapfile -t MTP < assets/mtp-qualified.argv
COMMON=(MODEL.gguf --prompt-ids assets/prompt-32768.ids --context 131072 --generate 128
        "${MTP[@]}" --mtp-golden-max-bytes 34359738368 --mtp-replay-host-bytes 35433480192)
# Single-node capture; the destination must not already exist.
env -u STRIX_QWEN_PF_PIPE -u STRIX_QWEN_PF_PIPE_DECODE -u STRIX_QWEN_PF_LAYER_RANGE \
  strix-hip-decode-token "${COMMON[@]}" --mtp-golden-dump assets/prompt-32768.golden
SHA=$(sha256sum assets/prompt-32768.golden); SHA=${SHA%% *}
REPLAY=("${COMMON[@]}" --mtp-golden assets/prompt-32768.golden --mtp-golden-sha256 "$SHA")
env -u STRIX_QWEN_PF_PIPE -u STRIX_QWEN_PF_PIPE_DECODE -u STRIX_QWEN_PF_LAYER_RANGE \
  strix-hip-decode-token "${REPLAY[@]}"
# Copy exactly the same golden/argv/prompt/model assets to the other rank first.
# Run once per node, setting R to 0 and 1 respectively:
R=0
STRIX_QWEN_PF_PIPE=1 STRIX_QWEN_PF_PIPE_DECODE=1 STRIX_QWEN_PF_PIPE_CHUNK=4096 \
  strix-hip-decode-token "${REPLAY[@]}" --rank "$R" --transport usb4auto
```

The 32K golden contains 32,767 adjacent prompt pairs, about 33.89 GB for the
2560x4/248320 geometry. Reserve complete replay memory in addition to device and
model memory. Change the prompt to 1,193 tokens only after capturing and pinning
its own golden; a 32K golden cannot validate that prompt. Self-generated goldens
are calibration evidence, not independent model-fidelity qualification. A pair
golden dump is rejected; capture is single-node and replay is paired.

## Shared pair modules

The seam pipe (two slots, CRC32C pieces and whole-frame CRC, consumption ACK), its HIP staging, the stage-balance cut
chooser, the `--rank/--peer/--transport` handling with the link setup, and the per-request mode selection live in
`include/strix/pair/` (`pipe.hpp`, `balance.hpp`, `link.hpp`, `regime.hpp`) and `src/xport/hip_pipe_staging.hpp`.
Qwen binds them with its seam-header validator; the Nemotron (gemma4) pair route (`STRIX_G4_PAIR=1`, see the model
README) uses the same modules with its own seam layout.
