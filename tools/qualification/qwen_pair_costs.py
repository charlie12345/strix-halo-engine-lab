#!/usr/bin/env python3
"""Convert private pair prefill layer/transfer profile logs into a measured cost bucket.

Collect both ranks at >=2 distinct legacy/explicit chunk sizes. Profiles perturb
execution; these coefficients are scheduling estimates, not throughput evidence.
"""
import argparse
import json
import math
from pathlib import Path

PREFIX = "qwen-pair-profile "


def calibrate(records, context):
    groups = {}
    transfers = {}
    for row in records:
        kind = row["kind"]
        rank, rows, ms = row["rank"], row["rows"], row["ms"]
        if rank not in (0, 1) or not isinstance(rows, int) or not 16 <= rows <= 16384 or rows % 8:
            raise ValueError("invalid profile geometry")
        if not math.isfinite(ms) or ms < 0:
            raise ValueError("invalid measured duration")
        if kind == "layer":
            layer = row["layer"]
            if layer != "lm_head" and (not layer.isdigit() or not 0 <= int(layer) < 48):
                raise ValueError("invalid layer")
            # Weight by actually measured rows; head records occur on terminal chunks only.
            sample = groups.setdefault(layer, [0.0, 0])
            sample[0] += ms
            sample[1] += rows
        elif kind == "transfer":
            transfers.setdefault(rows, {}).setdefault(rank, []).append(ms)
        elif kind != "transfer":
            raise ValueError("unknown profile kind")
    expected = {str(layer) for layer in range(48)} | {"lm_head"}
    if set(groups) != expected:
        raise ValueError("require measured layers 0..47 and terminal head from both ranks")
    if len(transfers) < 2:
        raise ValueError("require transfer samples at two or more chunk sizes")
    # Require both measured endpoints: receiver excludes header wait and includes H2D.
    if any(set(ranks) != {0, 1} for ranks in transfers.values()):
        raise ValueError("require sender and receiver transfers at each chunk size")
    # Fit per-size means; conservatively cover the slower endpoint.
    points = [(size, max(sum(samples) / len(samples) for samples in ranks.values()))
              for size, ranks in sorted(transfers.items())]
    xmean = sum(x for x, _ in points) / len(points)
    ymean = sum(y for _, y in points) / len(points)
    slope = sum((x - xmean) * (y - ymean) for x, y in points) / sum((x - xmean)**2 for x, _ in points)
    fixed = ymean - slope * xmean
    if not math.isfinite(slope + fixed) or slope < 0 or fixed < 0:
        raise ValueError("nonphysical transfer fit; collect more interleaved samples (do not clamp)")
    return {"buckets": [{"context": context,
        "layers": {str(layer): groups[str(layer)][0] / groups[str(layer)][1] for layer in range(48)},
        "lm_head": groups["lm_head"][0] / groups["lm_head"][1],
        "transfer_ms_per_row": slope, "fixed_ms": fixed}],
        "provenance": {"source": "STRIX_QWEN_PF_PIPE_PROFILE=1", "transfer_sizes": sorted(transfers),
            "layer_records": sum(1 for row in records if row["kind"] == "layer"),
            "timing": "synchronized wall time; folded work attributed to executing layer"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("logs", type=Path, nargs="+")
    args = parser.parse_args()
    if args.context <= 0:
        parser.error("context must be positive")
    records = []
    for path in args.logs:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.startswith(PREFIX):
                    records.append(json.loads(line[len(PREFIX):]))
    result = calibrate(records, args.context)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
