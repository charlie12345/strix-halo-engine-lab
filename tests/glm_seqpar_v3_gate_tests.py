#!/usr/bin/env python3
"""CPU tests with synthetic receipts only; these do not qualify device execution."""
import copy
from pathlib import Path
import tempfile
import unittest

from glm_seqpar_v3_gate import exposed_seconds, validate


class GateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        for name in ("tokens", "logits-r0", "logits-r1", "state:residual"):
            for arm in (0, 1):
                # Filenames remain portable on Windows (manifest names may use colons).
                (self.base / f"{name.replace(':', '-')}-{arm}").write_bytes(bytes(range(256)))
        config = "a" * 64
        ranks = [{"rank": r, "seconds": 14.2, "kernel": "test-kernel",
                  "amdgpu.iommu_perfopt": "1"} for r in (0, 1)]
        self.receipt = {
            "schema": "glm-seqpar-v3-gate-1", "identity": "r1-exact",
            "configuration_sha256": config,
            "timings": [{"seqpar": arm, "configuration_sha256": config, "profiled": False,
                         "ranks": [dict(r, seconds=14.2 - arm) for r in ranks]} for arm in (0, 1, 0, 1)],
            "state_payloads": ["residual"],
            "exactness": [{"name": name, "configuration_sha256": config,
                           "sp0": f"{name.replace(':', '-')}-0", "sp1": f"{name.replace(':', '-')}-1"}
                          for name in ("tokens", "logits-r0", "logits-r1", "state:residual")],
            "profiles": [{"rank": rank, "seqpar": 1, "configuration_sha256": config,
                          "window_ns": [0, 1_000_000_000],
                          "events": [
                              {"kind": "streamOpsWait", "queue": "transfer", "name": "streamOpsWait",
                               "start_ns": 0, "end_ns": 1_000_000_000},
                              {"kind": "compute", "queue": "half-a", "name": "column_attn",
                               "start_ns": 100_000_000, "end_ns": 600_000_000},
                              {"kind": "compute", "queue": "half-b", "name": "column_ffn",
                               "start_ns": 500_000_000, "end_ns": 900_000_000}]} for rank in (0, 1)],
        }

    def test_complete_receipt(self):
        self.assertEqual(validate(self.receipt, self.base), [0.2, 0.2])

    def test_interval_union_not_summed(self):
        # Overlapping queue waits and compute count once; uncovered edges survive.
        self.assertAlmostEqual(exposed_seconds([(0, 10), (5, 15)], [(2, 8), (4, 12)]), 5e-9)
        self.assertEqual(exposed_seconds([(2, 4)], [(0, 10)]), 0)

    def rejected(self, mutate):
        receipt = copy.deepcopy(self.receipt)
        mutate(receipt)
        with self.assertRaises(ValueError):
            validate(receipt, self.base)

    def test_threshold_strict(self):
        self.rejected(lambda r: r["profiles"][0]["events"][1].update(start_ns=200_000_000))

    def test_second_pair_must_win_on_both_ranks(self):
        self.rejected(lambda r: r["timings"][3]["ranks"][1].update(seconds=14.3))

    def test_not_abba(self):
        self.rejected(lambda r: r["timings"][2].update(seqpar=1))

    def test_missing_state_and_rank(self):
        self.rejected(lambda r: r["exactness"].pop())
        self.rejected(lambda r: r["profiles"].pop())

    def test_profiled_or_configuration_changed(self):
        self.rejected(lambda r: r["timings"][0].update(profiled=True))
        self.rejected(lambda r: r["timings"][1].update(configuration_sha256="b" * 64))
        self.rejected(lambda r: r["timings"][2]["ranks"][0].update(kernel="different-kernel"))

    def test_polling_and_reduction_fail(self):
        self.rejected(lambda r: r["profiles"][0]["events"][1].update(name="seqpar_wait_kernel"))
        self.rejected(lambda r: r["profiles"][0]["events"][1].update(name="reduce_kernel", kind="auxiliary"))

    def test_copy_cannot_hide_wait(self):
        self.rejected(lambda r: r["profiles"][0]["events"][1].update(name="copy_publish_kernel"))

    def test_mismatch_and_missing_payload_fail(self):
        (self.base / "tokens-1").write_bytes(b"mismatch")
        with self.assertRaises(ValueError):
            validate(self.receipt, self.base)
        (self.base / "tokens-1").unlink()
        with self.assertRaises(ValueError):
            validate(self.receipt, self.base)


if __name__ == "__main__":
    unittest.main()
