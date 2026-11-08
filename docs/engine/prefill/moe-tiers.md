# Exact routed MoE tiers and down LDS stores

Both controls default OFF until native byte and timing qualification passes:

| Control | Values | Default |
|---|---|---|
| `STRIX_QPF2K_MOE_TIERS` | `0`: existing dispatch; `1`: padded narrow tier; `2`: compact XOR narrow tier | `0` |
| `STRIX_QPF2K_MOE_DOWN_LDS_EPI` | `0`: scalar stores; `1`: LDS-backed vector stores | `0` |

Restart after changing either control. Tier lists must be reserved during pool
setup. Tune 82 partitions original CSR batch IDs into counts <=64 and >64;
tunes 80/88 retain their existing single gate/up launch. Full batches retain
the 128-row body. Both gate/up launches complete before either down launch.
Original batch IDs continue to index arrivals and DA16 images. K16 order,
operands, unscale, certificate checks and ordered merge remain unchanged.

## Down epilogue storage modes

The optional down epilogue reuses dead staging in 32-row passes, with a CTA
barrier before reuse and before vector reads. Each lane computes the same
`acc*unscale` product as the scalar store and writes it to LDS through the same
storage helper, so the bytes copied out with 16-byte stores are those the scalar
path would store:

| Rawdot storage | Epilogue | Bytes vs scalar stores of the same mode |
|---|---|---|
| FP32 (plain and certificate) | admitted | identical (same product, same FP32 value) |
| RB16 (certificate BF16) | admitted | identical (same BF16 rounding helper) |
| CTRL r22 route control | not admitted | n/a: scalar CTRL stores are kept |

Views lacking 16-byte base/stride alignment keep scalar stores. BF16 equality is
relative to its own scalar path, not to FP32 storage. The resolver's FAST
ownership and <=32768-token limit for BF16 remain unchanged.

CTRL rounds rawdots to r22 in its own down kernel, which neither the tier split
nor this epilogue reproduces. When `STRIX_QPF2K_MOE_ROUTE_CTRL=1` is set, both
controls stand down (tiers report 0, the epilogue reports off), so CTRL stores
are produced only by its scalar writer. Admitting CTRL requires an r22 epilogue
and its own byte gate.

## Offline compiler gate

Build the device TU from base `02956d449` and the candidate with the same
compiler, headers, device libraries and flags. Keep all generated files in an
ignored build directory, for example `build-isa/` (export base `src`, `include`
and `tools` with `git archive`). With Clang 20 and the ROCm 6.4 device bitcode:

```sh
clang++-20 -x hip --offload-arch=gfx1151 --hip-path=$ROCM -nogpulib \
  --cuda-device-only -D__HIP_PLATFORM_AMD__ -isystem $ROCM/include \
  -Iinclude -Isrc -std=c++20 -O3 -DNDEBUG -mno-wavefrontsize64 \
  -fno-fast-math -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt \
  -fdenormal-fp-math=ieee \
  $(for b in hip ockl ocml oclc_isa_version_1151 oclc_abi_version_500 \
      oclc_correctly_rounded_sqrt_on oclc_daz_opt_off oclc_finite_only_off \
      oclc_unsafe_math_off oclc_wavefrontsize64_off; do
    echo -Xclang -mlink-builtin-bitcode -Xclang $ROCM/lib/llvm/lib/clang/19/lib/amdgcn/bitcode/$b.bc; done) \
  -c src/hip_moe_q4k_a8_known_rows.hip -o build-isa/candidate.o

python3 scripts/qualification/qwen_moe_tiers_isa.py \
  --before build-isa/base.o --after build-isa/candidate.o \
  --readelf llvm-readelf-20 --cxxfilt llvm-cxxfilt-20 --out build-isa/audit
```

The audit requires all 30 incumbent DA16 bodies to retain identical instruction
bytes and kernel descriptors (entry offset masked), and all 48 tier
specializations plus the list builder to have zero scratch, SGPR/VGPR spills and
dynamic stack. It writes every specialization's LDS, registers and calculated
CTA limits to CSV. The LDS bound uses 128 KiB/WGP, rounded in 256-byte
allocations; the wave32 register bound uses four SIMDs/WGP, 1536 VGPRs per SIMD
and allocation units of 24. These are resource bounds, not measured residency.
The small ticket also prints the runtime-reported resources of the node build.

## Native tickets

Build before acquiring a timing ticket:

```sh
cmake -S . -B build-gfx1151 -DCMAKE_BUILD_TYPE=Release \
  -DSTRIX_WARNINGS_AS_ERRORS=ON -DSTRIX_ENABLE_HIP_RUNTIME=ON \
  -DSTRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE=ON \
  -DSTRIX_HIP_ARCHITECTURES=gfx1151
cmake --build build-gfx1151 --target strix-hip-moe-wmma-prefill-tests -j8
```

Supply a real layer's absolute tensor offsets and type IDs (gate/up: 12 Q4_K or
13 Q5_K; down: 7 Q5_1 or 8 Q8_0). Tensor weights always come from the model.

Routing has two sources, and output lines name the one used:

- `TIERS_ROUTES_FILE`: a capture of little-endian, tightly packed top-10 u32
  expert IDs per row. A 16384-row capture supplies prefixes for smaller shapes.
  Optional `TIERS_ROUTE_WEIGHTS_FILE` (top-10 FP32) and `TIERS_INPUT_FILE`
  (2560-element BF16 rows) match the same capture.
- Otherwise, histogram-matched routing: per-expert fills reproduce the measured
  production histogram at T8192 (per call 32/107/47/60/76/94/62/34 experts with
  0/1-16/17-32/33-64/65-128/129-256/257-512/>512 assignments), scaled by T/8192.
  It is deterministic per T, never repeats an expert within a row, and covers
  every fill bucket and partial batches.

The fixture prints the realised expert-fill histogram. Keep captures and
receipts out of Git.

```sh
BIN=./build-gfx1151/strix-hip-moe-wmma-prefill-tests
LAYER='assets/model.gguf:GATE_OFFSET:UP_OFFSET:DOWN_OFFSET:GATE_TYPE:DOWN_TYPE'
# Optional capture: export TIERS_ROUTES_FILE=assets/routes.u32

# Small run: synthetic byte suite, real-layer byte gates to T8192, timing T4096/T8192.
timeout -k 10 470 bash scripts/qualification/qwen_moe_tiers_ticket.sh small "$BIN" "$LAYER"

# Large run: byte gates and timing at T16384.
timeout -k 10 470 bash scripts/qualification/qwen_moe_tiers_ticket.sh large "$BIN" "$LAYER"
```

Each ticket prints `uname -r`, the binary's sha256, the routing source and the
IOMMU PerfOpt setting first; it never changes host settings. An inner 455-second
bound reports `FAIL timeout` before the outer 470-second limit. A timeout,
unreadable input, mismatch, inactive path, unsafe timing certificate or
incomplete sample set fails the ticket. Expected walls (estimated from per-call
times, not measured in this clone): small 3-5 min, large 1-2 min.

The small ticket executes `TIERS_MAX_ROWS=8192 "$BIN" tiers "$LAYER"`.
The suite compares all six arms (tiers 0/1/2 x epilogue 0/1) against 0/0,
including every written/padded word of rawdots, h, maxima, CSR/status, arrivals,
DA16 images, certificate and completed merge/commit output. It covers all four
format pairs, tunes 80/82/88, header reuse on/off, fills 1..256 plus 257/512/513
and zero experts, all eight expert-fill buckets, partial/tail reuse, failures,
nonfinite values and alignment fallback. Every tiersbench run byte-gates its
shape in plain FP32, certificate FP32 and certificate BF16 before timing.

Each timing command is `TIERS_STORAGE=fp32|bf16 "$BIN" tiersbench 7 "$LAYER" T`.
There are two warmups per arm and seven mirrored pairs, A B C D E F F E D C B A.
Every sample and median is reported in ms/call for gate/up, down and complete
enqueue plus ordered merge/commit/finish. Gate/up includes counter clearing and
tier-list construction. Events between gate/up and down preserve the production
launch sequence and add the same instrumentation to each arm.

## Default decision

The shipping arm is tiers=0, epilogue=0. Native equality and timing have not
been run in this clone; no speed gain is claimed.

Only an arm passing the compiler gate and every native byte gate is eligible.
Require its complete-call median to be no larger than baseline, with nonnegative
median paired savings, at every requested T (4096, 8192, 16384) and both storage
modes. Among eligible arms choose the smallest geometric mean of complete-call
ratios; ties retain 0/0. Inspect separate gate/up and down samples to attribute
the result. An incomplete ticket cannot qualify a mode. Do not sum independent
component gains or enable a regressing combination. If an arm is chosen, it
becomes the default with `0` remaining the off switch.
