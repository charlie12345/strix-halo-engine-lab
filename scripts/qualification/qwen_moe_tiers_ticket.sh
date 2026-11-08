#!/usr/bin/env bash
# Bounded native qualification for STRIX_QPF2K_MOE_TIERS / _DOWN_LDS_EPI.
# Run only on the device owner's admitted node, inside its GPU turn.
#   small: synthetic tier suite (every fill bucket, partial batches, failure and
#          alignment cases) + real-layer six-arm byte gates up to T8192 +
#          tiersbench at T4096 and T8192 (fp32 and bf16 storage).
#   large: tiersbench at T16384 (fp32 and bf16); each run byte-gates first.
# Routing: TIERS_ROUTES_FILE (little-endian u32 top-10, at least T rows) when
# given, otherwise the built-in histogram-matched routing. Both are labelled.
set -euo pipefail

if [[ ${1:-} != --inner ]]; then
    if [[ $# != 3 || ( $1 != small && $1 != large ) ]]; then
        echo "usage: [TIERS_ROUTES_FILE=routes.u32] $0 small|large BIN LAYER_SPEC" >&2
        exit 2
    fi
    # The inner bound stays below an outer `timeout -k 10 470`, so a slow run
    # reports itself. Timeout or an incomplete gate is a failed ticket.
    rc=0
    timeout --signal=TERM --kill-after=5s 455s bash "$0" --inner "$@" || rc=$?
    if [[ $rc == 124 || $rc == 137 ]]; then
        echo "tiers ticket $1: FAIL timeout after 455 s (no qualification)"
    elif [[ $rc != 0 ]]; then
        echo "tiers ticket $1: FAIL rc=$rc (no qualification)"
    fi
    exit "$rc"
fi
shift
ticket=$1
bin=$2
layer=$3
printf 'kernel='
uname -r
if [[ ! -x $bin ]]; then
    echo "tiers ticket $ticket: FAIL binary not executable: $bin"
    exit 2
fi
printf 'binary sha256='
sha256sum "$bin"
if [[ -n ${TIERS_ROUTES_FILE:-} ]]; then
    if [[ ! -r $TIERS_ROUTES_FILE ]]; then
        echo "tiers ticket $ticket: FAIL unreadable TIERS_ROUTES_FILE=$TIERS_ROUTES_FILE"
        exit 2
    fi
    printf 'routing=captured sha256='
    sha256sum "$TIERS_ROUTES_FILE"
else
    echo 'routing=histogram-matched (built in)'
fi
for v in TIERS_ROUTE_WEIGHTS_FILE TIERS_INPUT_FILE; do
    if [[ -n ${!v:-} && ! -r ${!v} ]]; then
        echo "tiers ticket $ticket: FAIL unreadable $v=${!v}"
        exit 2
    fi
done
unset STRIX_INTPREFILL_MOE_TUNE STRIX_BASELINE STRIX_QPF2K_DEFAULTS BENCH_FULL128 BENCH_UNIFORM \
    STRIX_QPF2K_MOE_TIERS STRIX_QPF2K_MOE_DOWN_LDS_EPI STRIX_QPF2K_MOE_ROUTE_CTRL
export BENCH_TIERS=0,1,2 BENCH_EPI=0,1
printf 'amdgpu.iommu_perfopt='
if [[ -r /sys/module/amdgpu/parameters/iommu_perfopt ]]; then
    cat /sys/module/amdgpu/parameters/iommu_perfopt
else
    echo unavailable
fi
boot=$(tr ' ' '\n' </proc/cmdline | sed -n 's/^amdgpu\.iommu_perfopt=//p')
printf 'boot amdgpu.iommu_perfopt=%s\n' "${boot:-unset}"

if [[ $ticket == small ]]; then
    TIERS_MAX_ROWS=8192 STRIX_INTPREFILL_RESOURCES=1 "$bin" tiers "$layer"
    shapes=(4096 8192)
else
    shapes=(16384)
fi
for rows in "${shapes[@]}"; do
    for storage in fp32 bf16; do
        TIERS_STORAGE=$storage "$bin" tiersbench 7 "$layer" "$rows"
    done
done
printf 'tiers ticket %s: all requested gates and samples completed\n' "$ticket"
