# Draft exact PLE input tile reuse

`STRIX_QPF2K_PLE_TILE=1` requests reuse of the packed PLE input panel across
the key and value output tiles. It is **default OFF**, startup-cached, and has no
numerical profile tag. Unset, `0`, and every spelling other than exactly `1` keep
the original transaction. Restart between arms. Native compilation, device byte
equality, model fidelity, capacity and complete-workload qualification are pending;
there is no measured improvement claim.

The implementation reuses the existing grouped lane kernels. The original
transaction is pack/key projection/pack/value projection. The candidate is
pack/key projection/value projection. Both projection launches retain their
original sites, weights, grids, 512-thread blocks, 64-output tiles and 64-row
groups. All launches retain their ordering barriers. The shared input panel is
read-only until both consumers finish.

Each output therefore retains the same BF16 widening, Q8 coefficient decode,
ascending FMA chain with the same operands, 256-leaf tree, addition source order,
FP32 stores and downstream BF16 boundaries. No arithmetic kernel is changed or
duplicated. Weight caches, scratch reservation, identities and capacity are
unchanged. The physical transaction removes one pack launch; it does not change
the numerical identity. No experts, attention selection budget or other prefill
stage is modified.

Admission requires the existing wide/batched PLE route and both original lane
admissions, identical gathered input addresses/width/pitch, complete 64-row
groups, and disjoint input/key/value output extents. Pitched bounding extents
conservatively refuse aliases, including padding overlap and address overflow.
Remaining rows use the original remainder route. Partial output tiles keep the
existing store guards. Wide/batch OFF, baseline suppression, alternate lane
recipes, incompatible shapes, aliases and scratch allocation refusal preserve
the original route. The switch does not force any of those routes on. Start with
the ordinary production command and inspect the resolved configuration; do not
add legacy organization flags to enable this candidate.

The resource delta is zero: the existing packed FP32 slab is 80 MiB at T8192
and 160 MiB at T16384 for input width 2560. The existing consumer declares 64 KiB
LDS and uses the same pinned registers in both arms. Obtain native compiler
resource reports; this source-level argument does not establish occupancy or
timing. Removing a pack eliminates one BF16 input read and one FP32 panel write
per transaction. It does not reduce projection arithmetic or coefficient traffic.

## Focused host and device gates

The existing executable `strix-hip-ple-launcher-tests` has additional modes; no
CMake changes or additional targets are needed. A standalone HIP-free build of
the same source runs the host contract and independent packed-bit/tree oracles:

```sh
mkdir -p .cache
c++ -x c++ -std=c++20 -O2 -fno-fast-math -ffp-contract=off \
  -DSTRIX_PLE_TILE_HOST_ONLY -Iinclude tests/hip_ple_launcher_tests.hip \
  -o .cache/ple-tile-host
.cache/ple-tile-host
```

From a Windows x64 C++ developer shell:

```bat
if not exist .cache mkdir .cache
cl /nologo /std:c++20 /EHsc /O2 /fp:strict /W4 /WX /TP /DSTRIX_PLE_TILE_HOST_ONLY /Iinclude tests\hip_ple_launcher_tests.hip /Fo.cache\ple-tile-host.obj /Fe.cache\ple-tile-host.exe
.cache\ple-tile-host.exe
```

Host coverage includes T8192/T16384 addressing, contiguous/padded/odd pitches,
strict inactive-switch parsing, overlap and overflow refusals, an independent
packed BF16 byte oracle (including signed zero, subnormals, NaN and Inf), and
the subset stack versus an independent complete 256-leaf/FMA tree. It does not
execute or qualify HIP instructions.

On the qualified native HIP toolchain, use an existing Release build directory
configured with `STRIX_ENABLE_HIP_RUNTIME=ON`, `BUILD_TESTING=ON` and
`STRIX_HIP_ARCHITECTURES=gfx1151`. Preserve its arithmetic/toolchain flags.
Request `-Rpass-analysis=kernel-resource-usage` in the native compiler receipt
when available, and capture the verbose build command:

```sh
cmake --build build/hip --target strix-hip-ple-launcher-tests -j2 --verbose
build/hip/strix-hip-ple-launcher-tests --ple-tile-host
```

Request at most **470 seconds** for the following device gate after the build.
Run in a clean environment without alternate Q8 lane recipes. The exact modes
check every key/value output word at T8192/T16384 with production dimensions
(input 2560, key 10240, value 2560), independent output pitches, output tails,
row counts 63/64/65/127, signed zeros, nonfinite inputs/scales, input/history
canaries, preserved padding/suffix rows, and odd-pitch fallback. A 65-row
composition compares actual BF16 in-place PLE tail outputs and advanced
convolution history with identical nonzero initial state. Missing resources,
timeout or a missing reference are not passes.

```sh
timeout 470s bash -euc '
  exe=build/hip/strix-hip-ple-launcher-tests
  STRIX_QPF2K_PLE_TILE=1 "$exe" --ple-tile-exact
  env -u STRIX_QPF2K_PLE_TILE "$exe" --ple-tile-inactive
  STRIX_QPF2K_PLE_TILE=0 "$exe" --ple-tile-inactive
  "$exe" --ple-tile-bench 8192
  "$exe" --ple-tile-bench 16384
'
```

Timing follows byte equality and includes the producer and **both** projection
consumers, with original and candidate output buffers of equal size. Each shape
runs three A/A pairs and three ABBA rounds for both warm and invalidated weight
caches. Scratch allocation is warmed equally; cold weight packing remains in the
timed interval. The printed 32K equivalents scale this one PLE pair by token
count; they are fixture extrapolations, not complete-workload measurements. Each
timing is accompanied by its ratio to the modeled 0.10 s stage floor and its
fraction of the modeled 11.9 s whole-workload floor. Report individual samples
and A/A noise alongside medians. Compare host wall time as well as stream events
when diagnosing submission overhead.

## Complete-workload admission

After native byte equality, compare OFF and ON using the same real-text 32768
token prompt, model, numerical profile, T, capacity and binary. Capture complete
token IDs, raw logits and all seven state payload digests with the existing
[complete-state qualification contracts](hc-output-wmma-qualification.md#split-profile-complete-allocation-export).
Keep the state/logit exports separate from timing. A startup witness alone is
insufficient. The candidate prints `qpf2k ple tile active: shared input panel`
only when its production transaction is admitted.

For a subsequent timing-only gate, preserve the automatic production route and
change only this switch. With private assets at the relative placeholders,
request another bounded gate of at most 470 seconds:

```sh
timeout 470s bash -euc '
  n=0
  for arm in 0 0 0 1 1 0 0 1 1 0; do
    n=$((n+1))
    STRIX_QPF2K_PLE_TILE=$arm build/hip/strix-hip-decode-token \
      assets/MODEL.gguf $(cat assets/prompt32k.ids) --generate 1 \
      >.cache/ple-tile-$arm-$n.out 2>.cache/ple-tile-$arm-$n.err
  done
'
```

This is one A/A pair followed by two ABBA rounds; use more counterbalanced rounds
only within a separately agreed budget if noise prevents a conclusion. Record
the running kernel, IOMMU PerfOpt parameter, clocks/power and applied route with
private receipts. Never reduce context/slots, alter the attention budget, sum
unrelated savings, or publish private prompts, paths or receipts.

Fixture normalization uses analytical denominators of 0.10 s per PLE stage and
11.9 s per 32768-token workload. These reporting references are not a measured
hardware maximum or an established throughput estimate. This lever only removes
a duplicate panel pack. Qualification requires complete transactions that
improve beyond A/A noise and pass every equality and capacity gate.
