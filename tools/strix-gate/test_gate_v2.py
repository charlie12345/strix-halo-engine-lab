"""Host tests of the FAST gate v2 statistics (gate_v2.py) and its two adapters (numpy only; no engine)."""
import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gate_v2 as v2  # noqa: E402
import test_fast_logit_gate as tfg  # noqa: E402


def write_routes(path, calls):
    """calls: [(layer, ids[rows, k], weights[rows, k])] in the engine's strix-route-file/v1 layout."""
    with open(path, "wb") as f:
        f.write(b"STXROUT1" + np.array([1, 0], "<u4").tobytes())
        for layer, ids, w in calls:
            ids, w = np.asarray(ids, "<u4"), np.asarray(w, "<f4")
            f.write(np.array([layer, ids.shape[0], ids.shape[1], 0], "<u4").tobytes() + ids.tobytes() + w.tobytes())


def window(name, n, kl, top1, dnll, null_kl, rng, hit=0.5):
    agree = np.arange(n) < round(top1 * n)
    d = np.full(n, dnll) + rng.normal(scale=0.05, size=n)
    d -= d.mean() - dnll  # exact mean
    nll_ref = np.full(n, 1.0)
    hits = np.arange(n) < round(hit * n)
    w = v2.window_stats(name, np.full(n, kl), agree, nll_ref, nll_ref + d, [null_kl], hits, hits)
    w["_rows"] = {"kl": np.full(n, kl), "agree": agree, "dnll": d}
    return w


class Statistics(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)

    def test_per_window_kl_limit_is_null_relative(self):
        noisy = window("ho7-01", 256, 0.040, 0.95, 0.0, 0.030, self.rng)   # chaotic window: limit 1.5 x 0.030 = 0.045
        self.assertAlmostEqual(noisy["kl_limit"], 0.045)
        self.assertTrue(noisy["pass"])
        quiet = window("ho7-00", 256, 0.040, 0.95, 0.0, 0.004, self.rng)   # quiet window: the absolute 0.02 stays
        self.assertEqual(quiet["kl_limit"], 0.02)
        self.assertFalse(quiet["pass"])
        self.assertFalse(quiet["checks"]["kl"])
        with self.assertRaises(v2.GateError):
            v2.window_stats("x", [0.0], [True], [1.0], [1.0], [])

    def test_top1_and_dnll_stay_absolute_per_window(self):
        w = window("w", 256, 0.001, 0.92, 0.0, 0.5, self.rng)
        self.assertEqual((w["checks"]["top1"], w["pass"]), (False, False))
        w = window("w", 256, 0.001, 0.99, 0.02, 0.5, self.rng)
        self.assertEqual((w["checks"]["dnll"], w["pass"]), (False, False))

    def test_pooled_rows_se_and_absolute_kl(self):
        ws = [window(f"w{i}", 256, 0.015, 0.97, 0.002, 0.001, self.rng) for i in range(3)]
        with self.assertRaises(v2.GateError):   # 768 < 800 rows: no verdict
            v2.verdict(ws)
        ws.append(window("w3", 64, 0.015, 0.97, 0.002, 0.001, self.rng))
        p = v2.verdict(ws)
        self.assertEqual(p["dnll_rows"], 832)
        d = np.concatenate([w["_rows"]["dnll"] for w in ws])
        self.assertAlmostEqual(p["dnll_se"], d.std(ddof=1) / np.sqrt(d.size))
        self.assertAlmostEqual(p["dnll"], 0.002)
        self.assertTrue(p["pass"])
        # A window passing under its own null limit still counts in the pooled KL, whose limit stays 0.02.
        ws = [window(f"w{i}", 256, 0.03, 0.97, 0.0, 0.03, self.rng) for i in range(4)]
        p = v2.verdict(ws)
        self.assertTrue(p["checks"]["windows"])
        self.assertFalse(p["checks"]["kl"])
        self.assertFalse(p["pass"])

    def test_labels_must_be_held_out(self):
        ws = [window(f"w{i}", 256, 0.001, 1.0, 0.0, 0.001, self.rng, hit=1.0) for i in range(4)]
        with self.assertRaises(v2.GateError):   # every label is the reference's argmax: greedy ids, not text
            v2.verdict(ws)
        ws = [window(f"w{i}", 256, 0.001, 1.0, 0.0, 0.001, self.rng, hit=0.6) for i in range(4)]
        self.assertAlmostEqual(v2.verdict(ws)["held_out"]["label_is_ref_top1"], 0.6, places=2)

    def test_route_files_flips_and_ratio(self):
        with tempfile.TemporaryDirectory() as t:
            t = pathlib.Path(t)
            ref = [(0, [[1, 2], [3, 4]], [[.5, .5], [.6, .4]]), (1, [[5, 6]], [[.7, .3]])]
            cand = [(0, [[2, 1], [3, 9]], [[.5, .5], [.6, .4]]), (1, [[5, 8]], [[.7, .3]])]   # order-only + 2 flips
            null = [(0, [[1, 2], [3, 9]], [[.5, .5], [.6, .4]]), (1, [[5, 6]], [[.7, .3]])]   # 1 flip
            for name, calls in (("ref", ref), ("cand", cand), ("null", null)):
                write_routes(t / f"{name}.rte", calls)
            c = v2.route_flips(t / "ref.rte", t / "cand.rte")
            n = v2.route_flips(t / "ref.rte", t / "null.rte")
            self.assertEqual((c["slots"], c["ids"], c["per_layer"]), (6, 2, {0: [4, 1], 1: [2, 1]}))
            self.assertEqual(v2.flip_ratio(c, n), 2.0)
            self.assertEqual(v2.flip_ratio({"rate": 0.0}, {"rate": 0.0}), 1.0)
            self.assertEqual(v2.flip_ratio({"rate": 0.1}, {"rate": 0.0}), float("inf"))
            ws = [window(f"w{i}", 256, 0.001, 0.99, 0.0, 0.001, self.rng) for i in range(4)]
            p = v2.verdict(ws, [{"cand": c, "null": n}])
            self.assertEqual((p["flips"]["ratio"], p["checks"]["flip_ratio"], p["pass"]), (2.0, False, False))
            write_routes(t / "short.rte", ref[:1])
            with self.assertRaises(v2.GateError):   # another call sequence: no verdict
                v2.route_flips(t / "ref.rte", t / "short.rte")
            write_routes(t / "layer.rte", [(0, [[1, 2], [3, 4]], [[1, 1], [1, 1]]), (2, [[5, 6]], [[1, 1]])])
            with self.assertRaises(v2.GateError):
                v2.route_flips(t / "ref.rte", t / "layer.rte")
            (t / "cut.rte").write_bytes((t / "ref.rte").read_bytes()[:-3])
            with self.assertRaises(v2.GateError):
                v2.read_routes(t / "cut.rte")

    def test_replay_bookkeeping(self):
        ref = {"version": 1, "out": {"path": "/o/routes/c.REF.rte", "records": 96, "rows": 9000}}
        arm = {"version": 1, "in": {"path": "/o/routes/c.REF.rte", "records": 96, "consumed": 96, "rows": 9000, "complete": True},
               "local_flips": {"slots": 90000, "ids": 120, "rows_changed": 110, "per_layer": []}}
        got = v2.check_replay(ref, arm, "c.REF.rte", "c.FAST")
        self.assertEqual((got["records"], got["local_flips"]["ids"]), (96, 120))
        with self.assertRaises(v2.GateError):   # a null that never moved a weight is not a null
            v2.check_replay(ref, arm, "c.REF.rte", "c.NULL", null=True)
        self.assertEqual(v2.check_replay(ref, dict(arm, null={"layer": 0, "seed": 1, "nudged": 5}), "c.REF.rte", "c.NULL",
                                         null=True)["null"]["nudged"], 5)
        short = json.loads(json.dumps(arm))
        short["in"].update(consumed=95, complete=False)
        for bad_ref, bad_arm, f in ((ref, short, "c.REF.rte"), (ref, arm, "d.REF.rte"), (ref, {"version": 1}, "c.REF.rte"),
                                    (dict(ref, null={"nudged": 1}), arm, "c.REF.rte")):
            with self.assertRaises(v2.GateError):
                v2.check_replay(bad_ref, bad_arm, f, "c.X")
        line = b"noise\nstrix-route-file/v1 " + json.dumps(arm).encode() + b"\nmore\n"
        self.assertEqual(v2.route_receipt(line, "x")["in"]["consumed"], 96)
        with self.assertRaises(v2.GateError):
            v2.route_receipt(b"no receipt\n", "x")


class FastLogitGateRuleV2(unittest.TestCase):
    """fast_logit_gate.py compare --rule v2 on sealed synthetic arms (helpers of test_fast_logit_gate)."""
    setUp, tearDown, selection, arm, cmp, noisy = (tfg.Case.setUp, tfg.Case.tearDown, tfg.Case.selection, tfg.Case.arm,
                                                   tfg.Case.cmp, tfg.Case.noisy)

    def test_rule_v2(self):
        a, _ = self.arm("A", "r1")
        near, _ = self.arm("N", "r2", self.noisy(1e-3, 1), mode=3)
        null, _ = self.arm("Z", "r3", self.noisy(1e-3, 2), mode=3)
        self.assertEqual(self.cmp(a, near, "--rule", "v2"), 2)                          # no null: no verdict
        self.assertEqual(self.cmp(a, near, "--rule", "v2", "--null-arm", str(null)), 2)  # 8 rows < 800: no verdict
        with mock.patch.dict(v2.LIMITS, {"min_rows": 1}):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = tfg.g.main(["compare", str(a), str(near), "--ranks", "0", "1", "--tokens", f"p={self.tok}",
                                 "--rule", "v2", "--null-arm", str(null)])
            self.assertEqual(rc, 0, out.getvalue())
            self.assertIn("window p:", out.getvalue())
            self.assertIn("POOLED v2", out.getvalue())
            far, _ = self.arm("F", "r4", self.noisy(2.0, 5), mode=3)
            self.assertEqual(self.cmp(a, far, "--rule", "v2", "--null-arm", str(null)), 1)


if __name__ == "__main__":
    unittest.main()
