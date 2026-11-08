#!/usr/bin/env bash
# Native executor payload, never run on an AMD-only executor. the executor coordinator owns dispatch.
# Usage: native-ticket.sh FULL_SOURCE_SHA CUDA_ARCH PHASE [PHASE_ARGUMENTS...]
# Run from the imported clean checkout; supply EXPECTED_SOURCE_BRANCH.
# POSIX process-group timeout; Windows needs the executor coordinator's equivalent owned job timeout.
set -euo pipefail

if [[ "${1-}" != --inside-deadline ]]; then
    if [[ "${OS-}" == Windows_NT ]]; then
        echo "Use the executor coordinator's Windows owned-job <=470s runner with the request command arrays." >&2
        exit 2
    fi
    # Configure, compiler work, execution, SHA collection and child cleanup
    # are all inside 460s + at most 5s TERM/KILL grace. the executor coordinator caps the whole turn
    # at 470s too. A timeout never becomes a passing receipt.
    set +e
    timeout --signal=TERM --kill-after=5s 460s bash "$0" --inside-deadline "$@"
    ticket_rc=$?
    echo "native-ticket-final rc=$ticket_rc"
    exit "$ticket_rc"
fi
shift
expected=${1:?full source SHA}
architecture=${2:?actual RTX architecture, resolved by the executor coordinator}
phase=${3:?build-operator, build-e2e, operator, or execution}
shift 3
[[ "$expected" =~ ^[0-9a-f]{40}$ ]]
case "$architecture" in 89|120) ;; *) echo "Unsupported/unresolved RTX architecture" >&2; exit 2;; esac
[[ "$(git rev-parse HEAD)" == "$expected" ]]
[[ "$(git branch --show-current)" == "${EXPECTED_SOURCE_BRANCH:?expected source branch}" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
# Source pin changes require a distinct build directory. No recursive cleanup.
build="build-qwen-route-bf16-${expected:0:12}-sm$architecture"
if [[ -e "$build/source-sha" ]]; then
    [[ "$(cat "$build/source-sha")" == "$expected" ]]
fi
case "$phase" in
build-operator|build-e2e)
    cmake -S . -B "$build" -DSTRIX_CUDA_QWEN_ONLY=ON \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        "-DCMAKE_CUDA_ARCHITECTURES=$architecture"
    if [[ "$phase" == build-operator ]]; then target=qwen_route_bf16_operator; else target=qwen_route_bf16_e2e; fi
    cmake --build "$build" --config Release --target "$target" -j12 --verbose
    sha256sum "$build/$target" >"$build/$target.sha256"
    printf '%s\n' "$expected" >"$build/source-sha"
    cat "$build/$target.sha256"
    ;;
operator)
    [[ $# == 3 ]]
    tables=$1
    pair=$2
    decode_kernels=$3
    [[ -d "$tables" && "$pair" =~ ^[0-3]$ && "$decode_kernels" =~ ^[01]$ ]]
    [[ "$(cat "$build/source-sha")" == "$expected" ]]
    sha256sum --check "$build/qwen_route_bf16_operator.sha256"
    STRIX_QRTX_DECODE_KERNELS="$decode_kernels" "$build/qwen_route_bf16_operator" "$tables" route-bf16 "$pair"
    ;;
execution)
    # MODEL TABLES ROPE PROMPT_IDS N G MAX_POS OUTPUT candidate|off [ARM].
    [[ $# == 9 || $# == 10 ]]
    [[ -f "$1" && -d "$2" && -f "$3" && -f "$4" ]]
    [[ ! -e "$8" ]]
    [[ "$(cat "$build/source-sha")" == "$expected" ]]
    sha256sum --check "$build/qwen_route_bf16_e2e.sha256"
    STRIX_QRTX_GRAPHS=0 "$build/qwen_route_bf16_e2e" "$@"
    ;;
*) echo "Unknown native phase" >&2; exit 2;;
esac
