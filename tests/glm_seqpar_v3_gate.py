#!/usr/bin/env python3
"""Offline acceptance of private, native GLMSP3 receipts; never launches devices.

See REPORT-SEQPAR.md v3 for capture requirements and the receipt schema. Missing
receipts are failures, not qualification. Times are seconds; trace times are ns.
"""
import argparse
import json
import math
from pathlib import Path
import re
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        require(math.isfinite(start) and math.isfinite(end) and 0 <= start < end,
                "invalid trace interval")
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def exposed_seconds(waits, compute):
    """Union of wait intervals minus useful compute on either half (no double count)."""
    waits, compute = union(waits), union(compute)
    total = sum(end - start for start, end in waits)
    for start, end in waits:
        for cstart, cend in compute:
            total -= max(0, min(end, cend) - max(start, cstart))
    return total * 1e-9


def same_bytes(left, right):
    require(left.resolve() != right.resolve(), "exactness capture must use two independent files")
    require(left.is_file() and right.is_file(), "missing exactness payload")
    require(left.stat().st_size > 0 and left.stat().st_size == right.stat().st_size,
            "empty or different-sized exactness payload")
    with left.open("rb") as a, right.open("rb") as b:
        while chunk := a.read(1024 * 1024):
            require(chunk == b.read(len(chunk)), "bitwise exactness mismatch")


def validate(receipt, base):
    require(receipt["schema"] == "glm-seqpar-v3-gate-1", "wrong receipt schema")
    require(receipt["identity"] in ("r1-exact", "r1-fast"), "unknown identity")
    config = receipt["configuration_sha256"]
    require(re.fullmatch(r"[0-9a-f]{64}", config) is not None, "missing configuration fingerprint")
    runs = receipt["timings"]
    require(len(runs) == 4 and [r["seqpar"] for r in runs] == [0, 1, 0, 1], "require ABAB")
    times = {rank: {0: [], 1: []} for rank in (0, 1)}
    for run in runs:
        require(run["configuration_sha256"] == config, "A/B configuration changed")
        require(run["profiled"] is False, "use unprofiled timing runs")
        require([r["rank"] for r in run["ranks"]] == [0, 1], "need both timing ranks")
        for rank in run["ranks"]:
            value = rank["seconds"]
            require(type(value) in (float, int) and math.isfinite(value) and value > 0,
                    "invalid timing")
            require(rank["kernel"] and rank["amdgpu.iommu_perfopt"] in ("0", "1"),
                    "record running kernel and PerfOpt")
            times[rank["rank"]][run["seqpar"]].append(value)
    # Fixed kernel and host parameter per rank across every arm.
    for rank in (0, 1):
        require(len({(r["ranks"][rank]["kernel"], r["ranks"][rank]["amdgpu.iommu_perfopt"])
                     for r in runs}) == 1, "host configuration changed across A/B")
        a, b = times[rank][0], times[rank][1]
        require(all(fast < plain for plain, fast in zip(a, b)), "SP1 must beat each paired SP0 on both ranks")
        require(statistics.median(b) < statistics.median(a), "SP1 median must beat SP0")

    captures = receipt["exactness"]
    names = [c["name"] for c in captures]
    require(len(names) == len(set(names)), "duplicate exactness payload")
    expected = receipt["state_payloads"]
    require(expected and len(expected) == len(set(expected)), "need complete canonical state manifest")
    require(set(names) == {"tokens", "logits-r0", "logits-r1"} | {"state:" + s for s in expected},
            "require complete tokens, both rank logits and all state payloads")
    for capture in captures:
        require(capture["configuration_sha256"] == config, "exactness configuration differs")
        same_bytes(base / capture["sp0"], base / capture["sp1"])

    profiles = receipt["profiles"]
    require([p["rank"] for p in profiles] == [0, 1], "need both rank profiles")
    exposed = []
    for profile in profiles:
        require(profile["seqpar"] == 1 and profile["configuration_sha256"] == config,
                "profile must be matching SP1")
        start, end = profile["window_ns"]
        union([(start, end)])
        waits, compute, queues = [], [], set()
        for event in profile["events"]:
            interval = (event["start_ns"], event["end_ns"])
            union([interval])
            require(start <= interval[0] < interval[1] <= end, "event outside full SP island")
            name = event["name"]
            require("seqpar_wait_kernel" not in name and "seqpar_copy_kernel" not in name,
                    "v2 polling/staging kernel in v3")
            require("reduce_kernel" not in name, "standalone SP reduction kernel")
            if event["kind"] == "streamOpsWait":
                waits.append(interval)
            elif event["kind"] == "compute":
                require(event["queue"] in ("half-a", "half-b"), "compute must run on a half queue")
                require(not any(s in name.lower() for s in ("wait", "copy", "reduce_kernel")),
                        "polling/copy/reduction cannot count as useful compute")
                compute.append(interval)
                queues.add(event["queue"])
            else:
                require(event["kind"] == "auxiliary", "unknown event classification")
                require("reduce_kernel" not in name, "standalone SP reduction kernel")
        require(waits and queues == {"half-a", "half-b"}, "incomplete wait or compute trace")
        seconds = exposed_seconds(waits, compute)
        require(seconds < 0.3, "exposed streamOpsWait must be < 0.3 s/rank")
        exposed.append(seconds)
    return exposed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, help="private native receipt JSON")
    args = parser.parse_args()
    try:
        exposed = validate(json.loads(args.receipt.read_text()), args.receipt.resolve().parent)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"GLMSP3 gate FAIL: {error}\n")
    print("GLMSP3 recorded exactness/ABAB/exposed-wait gate PASS; seconds=" +
          ",".join(f"{s:.6f}" for s in exposed))


if __name__ == "__main__":
    main()
