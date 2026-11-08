#!/usr/bin/env bash
# MAIN RTX executor only. Each phase includes configure, -j12 build and cleanup.
set -euo pipefail
if [[ "${1-}" != --inside ]]; then
    timeout --signal=TERM --kill-after=5s 460s bash "$0" --inside "$@"
    exit $?
fi
shift
sha=${1:?full SHA}; arch=${2:?89 or 120}; phase=${3:?build or run}; shift 3
[[ "$sha" =~ ^[0-9a-f]{40}$ && "$(git rev-parse HEAD)" == "$sha" ]]
[[ "$(git branch --show-current)" == sol/parity-qwen-rtx-bf16-mtp ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
case "$arch" in 89|120) ;; *) exit 2;; esac
# Keep generated files outside the source tree; no recursive cleanup.
build="../qwen-bf16-mtp-${sha:0:12}-sm$arch"
cmake -S . -B "$build" -DSTRIX_CUDA_QWEN_ONLY=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "-DCMAKE_CUDA_ARCHITECTURES=$arch"
cmake --build "$build" --config Release --target qwen_bf16_mtp_e2e -j12 --verbose
sha256sum "$build/qwen_bf16_mtp_e2e"
case "$phase" in
build) [[ $# == 0 ]] ;;
run)
    # MODEL SIDECAR TABLES ROPE PROMPT_IDS N G CAP NEW_OUTPUT ARM
    [[ $# == 10 && -f "$1" && -f "$2" && -d "$3" && -f "$4" && -f "$5" && ! -e "$9" ]]
    # These are per-process execution controls, never global environment mutations.
    if [[ "${10}" == cancel ]]; then
        set +e
        STRIX_QRTX_GRAPHS=0 "$build/qwen_bf16_mtp_e2e" "$@"
        rc=$?
        set -e
        [[ $rc == 3 && ! -e "$9/COMPLETE.txt" ]]
    else
        STRIX_QRTX_GRAPHS=0 "$build/qwen_bf16_mtp_e2e" "$@"
        [[ -f "$9/COMPLETE.txt" ]]
    fi
    ;;
*) exit 2 ;;
esac
