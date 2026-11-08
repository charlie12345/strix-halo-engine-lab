# Pair prefill and rank-0 continuation

Pair handoff is explicit: set `STRIX_QWEN_PF_PIPE=1` and `STRIX_QWEN_PF_PIPE_DECODE=1`,
then launch one process per rank with `--rank 0|1 --transport usb4auto`.
Use identical binaries, model assets, prompt IDs, numerical options and capacity.
The default pair chunk is 4,096 rows where admitted. Session save after a
two-node prefill requires a prompt length that is a multiple of the chunk size.
This work remains pending native qualification; host tests do not establish
token equality or speed.

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
