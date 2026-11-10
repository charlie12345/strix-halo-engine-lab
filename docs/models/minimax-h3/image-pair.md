# H3 image pair execution candidate

The image CLI uses the shared H3 DiT, six conditional PDD Euler evaluations and
image VAE. There is no unconditional evaluation or CFG branch. Native image pair
compilation, fidelity, deterministic replay and performance qualification are pending.

## Select a regime

Run one process per node, with identical binary, model assets, prompt, references,
seed and numerical flags. Use separate output directories on shared storage.

```sh
# One image: row split for latency (run R=0 and R=1 on their respective nodes).
build-h3/strix-h3-image --model assets/h3-image --te-pack assets/h3-te-pack \
  --prompt-file assets/prompt.txt --world 2 --rank R --transport usb4gpu \
  --parallel auto --width 1024 --height 1024 --seed 42 --output output/image.png

# Several images: complete requests on each node for throughput.
build-h3/strix-h3-image --model assets/h3-image --te-pack assets/h3-te-pack \
  --prompt-file assets/prompt.txt --world 2 --rank R --parallel auto --requests 4 \
  --width 1024 --height 1024 --seed 42 --output output/image.png
```

`--parallel auto` selects sequence parallelism for one image on two nodes and
request parallelism for several. `--parallel sequence|request` overrides this
decision. Sequence requires world 2; world 1 retains the original single-node
forward. `--dist` on world 1 is the existing distributed schedule reference.
`--print-config` reports the selected regime without device work.

Request i uses seed+i and filename `image.request-i.png` when count >1. Rank r
owns i when i%world==r. The prompt and references are shared within this CLI
batch; this is not a heterogeneous request server. Text refinement, preparation
and device weights are retained for assigned requests. DiT is released before
one VAE load and local batch decoding. First-image latency within a batch includes
the local sampling batch. Request ranks do not connect or exchange per-step data.

## Sequence and transport contracts

Weights are replicated. Packed token rows are contiguous and the cut is aligned
to 64 rows. Each block exchanges BF16 K and transposed V, overlaps local attention
with transport, then computes the peer partial and merges in canonical block
order. Final target rows are gathered on both ranks. No tensor-parallel matrix
or alternate kernel implementation is introduced.

The CLI uses the shared `strix::dist::Communicator`, automatic USB4 host pairing,
`usb4_gpu_direct_streams` and `usb4_gpu_direct`, as the production pair runners.
`usb4gpu` (default) and `usb4auto` prefer negotiated direct GPU streams. Both ranks
fall back to the shared host path when either has no direct stream; direct setup
errors fail closed. No driver/configfs settings are changed. The shared default
route table, framing, CRC, exported arenas and event lifetime checks are retained.
The separate copied-stream backend's control/lease contract is not a replacement
for this production communicator interface.

H3 enables the communicator's optional first-direct-chunk overlap policy. Above
ring capacity only the first direct chunk overlaps caller work; the remaining
chunks drain at finish. Mixed routes also overlap the host portion. Host fallback
uses the existing asynchronous chunk pipeline and double-buffer event hazards.
Measure exposed communication time instead of assuming full wire hiding.

The pair agrees inputs and policy before direct setup, including
`--attn-score-2p15`. The default attention bound remains 2^14. Every step retains
mandatory device digests, finite/operand checks and bound-row admission; optional
full residual host downloads are disabled. State, velocity, final latent and RGB
hash consensus fails closed. This does not replace receipt comparison or asset
hash verification by the operator. Both ranks write complete identical outputs.

## Preparation and VAE

Block preparation happens before sampling through the existing uploader and
finite adaLN preparation. `--lazy-prepare` restores first-use preparation on the
single-node route. Timings separate residency/pairing from sampling; moving work
to startup does not establish a cold wall-time improvement.

`--vae-parallel auto|off` controls tile dispatch in sequence mode. The canonical
spatial tiles, including overlap, are split into two contiguous ranges. Each rank
evaluates its range through the same selected decoder and exchanges raw FP32 RGB
tiles once. Both ranks blend raw neighbours above then left, crop and normalize
in the original order. Video's temporal shard representation is unchanged.
`--fast-vae-fp16` reuses the existing GPU image decoder, including the released
1024-wide geometry; its numerical identity stays explicit. Pair mode does not
silently enable a different VAE precision. `--vae-parallel off` decodes the whole
image on both ranks, preserving rank equality.

## Qualification

Host gates are `strix-h3-image-pair-tests`, `strix-h3-image-pair-cli`, existing
image host/checkpoint tests and `tests/h3/image/test_tools.py`. They cover split,
transpose/padding, in-process exchange, canonical attention merging, request
ownership and independent halo/crop/blending. Native transport overlap is covered
by the extended `strix-hip-dist-gpu-direct-tests` device ticket.

Require byte equality for unchanged single-node outputs, both sequence ranks and
deterministic repeats (noise, all states/velocities, latent, RGB and PNG). Pair
versus single targets the existing exact canonical arithmetic. The existing
draft fallback envelope is latent relative L2 <=0.02 and RGB/PNG PSNR >=35 dB;
departures from byte equality need investigation and native qualification.
Compare identical quantization, VAE precision, attention policy, initial noise,
prompt and model contents. Missing reference assets are NO-REFERENCE, not a pass.
