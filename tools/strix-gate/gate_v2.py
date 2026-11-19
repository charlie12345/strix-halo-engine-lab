#!/usr/bin/env python3
"""FAST gate v2: one statistics implementation for every engine (README "FAST gate v2").

Limits are the FAST gate's (KL <= 0.02, top-1 >= 0.93, dNLL <= 0.01), applied pooled AND per window, measured so
that the reference's own chaos is not charged to the lever:
  R1 MoE engines: the lever arm replays the reference's expert routes (ids and weights); the route flip rate of the
     free-running lever arm, over the free-running null arm's, must be <= 1.5 (reported per window and pooled).
  R2 per window: KL <= max(0.02, 1.5 x the null arm's KL on that window); the pooled KL limit stays 0.02.
  R3 dNLL on held-out ids (labels that neither arm generated), at least 800 scored rows pooled, standard error shown.

  python gate_v2.py qual OUT --ref REF --cand FAST --null NULL [--cand-free FASTF --null-free NULLF] [--json F]
      option_qual runner output: OUT/logits/<case>.<arm>.bin, OUT/log/<case>.<arm>.stderr, OUT/routes/<case>.<arm>.rte
  fast_logit_gate.py compare REF CAND --rule v2 --null-arm NULL ...   (sealed strix-fast-dump arms)
Exit 0 = PASS, 1 = FAIL, 2 = no verdict (bookkeeping defect, too few rows, labels not held out, missing null).
"""
import argparse
import json
import math
import pathlib
import re
import sys

import numpy as np

LIMITS = {"max_kl": 0.02, "min_top1": 0.93, "max_dnll": 0.01, "null_factor": 1.5, "max_flip_ratio": 1.5, "min_rows": 800}
ROUTE_MAGIC = b"STXROUT1"
RECEIPT = re.compile(rb"^strix-route-file/v1 (\{.*\})\s*$", re.M)
MIN_LABEL_ROWS = 32  # below this the held-out check cannot tell greedy ids from text


class GateError(Exception):
    """A defect of the evidence: no verdict (exit 2)."""


def mean_se(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        raise GateError("no scored rows")
    se = float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else float("inf")
    return float(x.mean()), se


def window_stats(name, kl, agree, nll_ref, nll_cand, null_kls, hit_ref=None, hit_cand=None, limits=LIMITS):
    """One window: R2 KL limit relative to this window's null, absolute top-1 and dNLL, dNLL standard error."""
    kl, agree = np.asarray(kl, np.float64), np.asarray(agree, bool)
    d = np.asarray(nll_cand, np.float64) - np.asarray(nll_ref, np.float64)
    if not null_kls:
        raise GateError(f"{name}: no null arm (v2 needs the one-ulp / order-only null of every window)")
    null_kl = float(np.mean(null_kls))
    kl_limit = max(limits["max_kl"], limits["null_factor"] * null_kl)
    dnll, se = mean_se(d)
    w = {"name": name, "rows": int(kl.size), "dnll_rows": int(d.size), "kl": float(kl.mean()), "top1": float(agree.mean()),
         "dnll": dnll, "dnll_se": se, "null_kl": null_kl, "kl_limit": kl_limit}
    if hit_ref is not None:
        w["label_is_ref_top1"] = float(np.mean(hit_ref))
        w["label_is_cand_top1"] = float(np.mean(hit_cand))
    w["checks"] = {"kl": w["kl"] <= kl_limit, "top1": w["top1"] >= limits["min_top1"], "dnll": dnll <= limits["max_dnll"]}
    w["pass"] = all(w["checks"].values())
    return w


def check_held_out(windows):
    """R3: labels must not be an arm's own greedy ids (both arms agreeing with every label means no held-out text)."""
    rows = sum(w["dnll_rows"] for w in windows if "label_is_ref_top1" in w)
    if rows < MIN_LABEL_ROWS:
        return None
    for key in ("label_is_ref_top1", "label_is_cand_top1"):
        frac = sum(w[key] * w["dnll_rows"] for w in windows) / rows
        if frac >= 1.0:
            raise GateError(f"every label equals the {'reference' if 'ref' in key else 'candidate'}'s top-1: the labels are "
                            "that arm's greedy ids, not held-out text (R3)")
    return {k: sum(w[k] * w["dnll_rows"] for w in windows) / rows for k in ("label_is_ref_top1", "label_is_cand_top1")}


def read_routes(path):
    """strix-route-file/v1 -> list of (layer, ids[rows, k] uint32, weights[rows, k] float32)."""
    data = pathlib.Path(path).read_bytes()
    if len(data) < 16 or data[:8] != ROUTE_MAGIC or int.from_bytes(data[8:12], "little") != 1:
        raise GateError(f"{path}: not a strix-route-file/v1")
    out, at = [], 16
    while at < len(data):
        if at + 16 > len(data):
            raise GateError(f"{path}: truncated record header")
        layer, rows, k, _ = np.frombuffer(data, "<u4", 4, at).tolist()
        n = rows * k
        if k == 0 or k > 64 or at + 16 + 8 * n > len(data):
            raise GateError(f"{path}: corrupt or truncated record {len(out)}")
        ids = np.frombuffer(data, "<u4", n, at + 16).reshape(rows, k)
        w = np.frombuffer(data, "<f4", n, at + 16 + 4 * n).reshape(rows, k)
        out.append((layer, ids, w))
        at += 16 + 8 * n
    return out


def route_flips(ref_path, arm_path):
    """Ids of the reference's selected sets missing from the arm's, per layer (order-free). Same call sequence required."""
    a, b = read_routes(ref_path), read_routes(arm_path)
    if len(a) != len(b):
        raise GateError(f"{arm_path}: {len(b)} router calls, the reference has {len(a)}")
    per = {}
    for i, ((la, ia, _), (lb, ib, _)) in enumerate(zip(a, b)):
        if la != lb or ia.shape != ib.shape:
            raise GateError(f"{arm_path}: call {i} is layer {lb} shape {ib.shape}, the reference has layer {la} shape {ia.shape}")
        k = ia.shape[1]
        missing = k - np.array([np.intersect1d(x, y).size for x, y in zip(ia, ib)], dtype=np.int64)
        s = per.setdefault(int(la), [0, 0])
        s[0] += ia.size
        s[1] += int(missing.sum())
    slots, ids = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    return {"slots": slots, "ids": ids, "rate": ids / slots if slots else 0.0, "per_layer": per}


def flip_ratio(cand, null):
    if null["rate"] == 0.0:
        return 1.0 if cand["rate"] == 0.0 else float("inf")
    return cand["rate"] / null["rate"]


def route_receipt(stderr_bytes, what):
    found = RECEIPT.findall(stderr_bytes)
    if len(found) != 1:
        raise GateError(f"{what}: expected one strix-route-file/v1 receipt line, found {len(found)}")
    return json.loads(found[0])


def check_replay(ref_receipt, arm_receipt, ref_file, what, null=False):
    """R1 bookkeeping: the arm replayed the reference's whole route file, and the null really moved its weights."""
    out = ref_receipt.get("out")
    if not out or "in" in ref_receipt or "null" in ref_receipt:
        raise GateError(f"{what}: the reference must record its routes free-running (out only)")
    rin = arm_receipt.get("in")
    if not rin:
        raise GateError(f"{what}: the arm did not replay the reference's routes")
    if pathlib.PurePath(rin["path"]).name != pathlib.PurePath(ref_file).name:
        raise GateError(f"{what}: replayed {rin['path']}, not the reference file {ref_file}")
    if not rin["complete"] or rin["consumed"] != out["records"] or rin["records"] != out["records"]:
        raise GateError(f"{what}: replay consumed {rin['consumed']} of {rin['records']} records, the reference wrote {out['records']}")
    if null and not arm_receipt.get("null", {}).get("nudged"):
        raise GateError(f"{what}: the null arm moved no weight (no STRIX_MOE_ROUTE_NULL, or its layer never ran)")
    return {"records": rin["consumed"], "rows": rin["rows"], "local_flips": arm_receipt.get("local_flips"),
            "null": arm_receipt.get("null")}


def verdict(windows, flips=None, limits=LIMITS):
    """Pooled + per-window verdict. windows: window_stats dicts with their per-row arrays under '_rows'."""
    held = check_held_out(windows)
    kl = np.concatenate([w["_rows"]["kl"] for w in windows])
    agree = np.concatenate([w["_rows"]["agree"] for w in windows])
    d = np.concatenate([w["_rows"]["dnll"] for w in windows])
    dnll, se = mean_se(d)
    pooled = {"rows": int(kl.size), "dnll_rows": int(d.size), "kl": float(kl.mean()), "top1": float(agree.mean()),
              "dnll": dnll, "dnll_se": se, "held_out": held}
    if d.size < limits["min_rows"]:
        raise GateError(f"{d.size} scored dNLL rows pooled, v2 needs at least {limits['min_rows']} (se {se:.4f})")
    checks = {"kl": pooled["kl"] <= limits["max_kl"], "top1": pooled["top1"] >= limits["min_top1"],
              "dnll": dnll <= limits["max_dnll"], "windows": all(w["pass"] for w in windows)}
    if flips is not None:
        cand = {"slots": sum(f["cand"]["slots"] for f in flips), "ids": sum(f["cand"]["ids"] for f in flips)}
        null = {"slots": sum(f["null"]["slots"] for f in flips), "ids": sum(f["null"]["ids"] for f in flips)}
        for x in (cand, null):
            x["rate"] = x["ids"] / x["slots"] if x["slots"] else 0.0
        pooled["flips"] = {"cand": cand, "null": null, "ratio": flip_ratio(cand, null)}
        checks["flip_ratio"] = pooled["flips"]["ratio"] <= limits["max_flip_ratio"]
    pooled["checks"] = checks
    pooled["pass"] = all(checks.values())
    return pooled


def window_line(w, flips=None):
    fl = ""
    if flips:
        fl = (f"; flips free cand {flips['cand']['rate']:.3e} null {flips['null']['rate']:.3e} ratio "
              f"{flip_ratio(flips['cand'], flips['null']):.2f}")
    bad = [k for k, ok in w["checks"].items() if not ok]
    return (f"window {w['name']}: rows {w['rows']} KL {w['kl']:.5f} (limit {w['kl_limit']:.5f} = max(0.02, 1.5 x null "
            f"{w['null_kl']:.5f})) top-1 {w['top1']:.4f} dNLL {w['dnll']:+.5f} se {w['dnll_se']:.5f}{fl} -> "
            f"{'PASS' if w['pass'] else 'FAIL ' + ','.join(bad)}")


def pooled_line(p):
    fl = ""
    if "flips" in p:
        f = p["flips"]
        fl = f"; route flips free cand {f['cand']['rate']:.3e} / null {f['null']['rate']:.3e} = {f['ratio']:.2f} (<= 1.5)"
    bad = [k for k, ok in p["checks"].items() if not ok]
    return (f"POOLED v2 rows {p['rows']} (dNLL rows {p['dnll_rows']}, >= {LIMITS['min_rows']}): KL {p['kl']:.5f} (<= 0.02) "
            f"top-1 {p['top1']:.4f} dNLL {p['dnll']:+.5f} se {p['dnll_se']:.5f}{fl} -> "
            f"{'PASS' if p['pass'] else 'FAIL ' + ','.join(bad)}")


def _option_qual_reduce():
    root = pathlib.Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tools.qualification.option_qual.reduce import reduce_pair  # noqa: E402
    return reduce_pair


def qual(a):
    """option_qual adapter: reduce every case's REF vs CAND / NULL logits, check replay receipts, count free flips."""
    out = pathlib.Path(a.out)
    reduce_pair = _option_qual_reduce()
    cases = a.cases or sorted(p.name[:-len(f".{a.ref}.bin")] for p in (out / "logits").glob(f"*.{a.ref}.bin"))
    if not cases:
        raise GateError(f"{out}/logits holds no {a.ref} logits")
    windows, flips, replay = [], [], {}
    for case in cases:
        logit = lambda arm: out / "logits" / f"{case}.{arm}.bin"  # noqa: E731
        route = lambda arm: out / "routes" / f"{case}.{arm}.rte"  # noqa: E731
        err = lambda arm: (out / "log" / f"{case}.{arm}.stderr").read_bytes()  # noqa: E731
        ra, _ = reduce_pair(logit(a.ref), logit(a.cand), a.format)
        nulls = [reduce_pair(logit(a.ref), logit(n), a.format)[0]["kl_rc"].mean() for n in a.null]
        w = window_stats(case, ra["kl_rc"], ra["agree"].astype(bool), ra["nll_r"], ra["nll_c"], [float(x) for x in nulls],
                         ra["top1_r_is_label"], ra["top1_c_is_label"])
        w["_rows"] = {"kl": ra["kl_rc"], "agree": ra["agree"].astype(bool), "dnll": ra["nll_c"] - ra["nll_r"]}
        moe = route(a.ref).exists()
        f = None
        if moe:
            ref_r = route_receipt(err(a.ref), f"{case}.{a.ref}")
            replay[case] = {"cand": check_replay(ref_r, route_receipt(err(a.cand), f"{case}.{a.cand}"), route(a.ref), f"{case}.{a.cand}"),
                            "null": [check_replay(ref_r, route_receipt(err(n), f"{case}.{n}"), route(a.ref), f"{case}.{n}", null=True)
                                     for n in a.null]}
            w["replay"] = replay[case]
            if not (a.cand_free and a.null_free):
                raise GateError(f"{case}: MoE routes present: --cand-free and --null-free arms are required (R1 flip ratio)")
            f = {"cand": route_flips(route(a.ref), route(a.cand_free)), "null": route_flips(route(a.ref), route(a.null_free))}
            flips.append(f)
            w["flips"] = {k: {x: v[x] for x in ("slots", "ids", "rate")} for k, v in f.items()}
        windows.append(w)
        print(window_line(w, f))
        for c in ([replay[case]["cand"]] + replay[case]["null"]) if moe else []:
            lf = c["local_flips"] or {}
            print(f"  replay {case}: {c['records']} router calls, {c['rows']} rows replayed; local flips "
                  f"{lf.get('ids', 0)}/{lf.get('slots', 0)}" + (f"; null layer {c['null']['layer']} seed {c['null']['seed']} "
                                                                 f"nudged {c['null']['nudged']}" if c["null"] else ""))
    p = verdict(windows, flips if flips else None)
    print(pooled_line(p))
    if a.json:
        doc = {"schema": "strix-fast-gate-v2/v1", "limits": LIMITS, "ref": a.ref, "cand": a.cand, "null": a.null,
               "windows": [{k: v for k, v in w.items() if k != "_rows"} for w in windows], "pooled": p}
        pathlib.Path(a.json).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return 0 if p["pass"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("qual", help="gate option_qual runner output")
    q.add_argument("out")
    q.add_argument("--ref", required=True)
    q.add_argument("--cand", required=True)
    q.add_argument("--null", action="append", required=True)
    q.add_argument("--cand-free")
    q.add_argument("--null-free")
    q.add_argument("--cases", nargs="*")
    q.add_argument("--format", default="steps")
    q.add_argument("--json")
    a = ap.parse_args(argv)
    try:
        return qual(a)
    except GateError as e:
        print(f"NO VERDICT: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
