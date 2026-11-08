#!/usr/bin/env bash
# Qwen prefill MoE TILE64 node ticket: byte gate (real layers) then interleaved off/on timing at T8192 and T16384.
# Supply the three layer specs through LAYER_SPEC_Q4_Q51, LAYER_SPEC_Q5_Q80,
# LAYER_SPEC_Q4_Q80; see the fixture's MODEL:GATE:UP:DOWN:GATE_TYPE:DOWN_TYPE syntax.
# Run under: timeout -k 10 470 bash scripts/qualification/qwen-moe-tile64-ticket.sh [build dir]
# Wall estimate: byte gate ~3 min (3 real layers x 3 T x 2 HDR x 3 storage x 5 runs + synthetic cases), timing ~2.5 min.
set -euo pipefail
if [[ $# -gt 1 ]]; then
    echo "usage: LAYER_SPEC_Q4_Q51=... LAYER_SPEC_Q5_Q80=... LAYER_SPEC_Q4_Q80=... $0 [build dir]" >&2
    exit 2
fi
BUILD=${1:-build}
BIN=$BUILD/strix-hip-moe-wmma-prefill-tests
L1=${LAYER_SPEC_Q4_Q51:?set the Q4_K gate/up and Q5_1 down layer spec}
L2=${LAYER_SPEC_Q5_Q80:?set the Q5_K gate/up and Q8_0 down layer spec}
L3=${LAYER_SPEC_Q4_Q80:?set the Q4_K gate/up and Q8_0 down layer spec}
echo "kernel $(uname -r)"
sha256sum "$BIN"
start=$(date +%s)
"$BIN" tile64 "$L1" "$L2" "$L3"
echo "tile64 gate rc=0 after $(( $(date +%s) - start )) s"
# 2 warmups + 7 interleaved pairs per T and storage mode; pairs drop to 5 if the gate took longer than 4 min.
pairs=7; [ $(( $(date +%s) - start )) -gt 240 ] && pairs=5
"$BIN" tile64time "$pairs" "$L3" 8192 16384
echo "tile64time g12/d8 rc=0 pairs=$pairs"
"$BIN" tile64time 3 "$L1" 8192 16384
echo "tile64time g12/d7 rc=0 pairs=3"
echo "ticket wall $(( $(date +%s) - start )) s"
