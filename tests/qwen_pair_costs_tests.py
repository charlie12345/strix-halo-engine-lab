#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("costs", Path(__file__).resolve().parents[1] / "tools/qualification/qwen_pair_costs.py")
costs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(costs)


class CostsTest(unittest.TestCase):
    def records(self):
        layers = [{"kind": "layer", "rank": int(layer >= 24), "rows": 2048,
                   "layer": str(layer), "ms": 2.048} for layer in range(48)]
        return layers + [{"kind": "layer", "rank": 1, "rows": 2048, "layer": "lm_head", "ms": 4.096}] + [
            {"kind": "transfer", "rank": rank, "rows": size, "ms": 0.01 * size + 3}
            for size in (2048, 4096, 8192) for rank in (0, 1)]

    def test_measured_fit_and_terminal_head(self):
        bucket = costs.calibrate(self.records(), 32768)["buckets"][0]
        self.assertAlmostEqual(bucket["transfer_ms_per_row"], 0.01)
        self.assertAlmostEqual(bucket["fixed_ms"], 3)
        self.assertAlmostEqual(bucket["layers"]["47"], 0.001)
        self.assertAlmostEqual(bucket["lm_head"], 0.002)

    def test_missing_layers_or_transfer_sizes(self):
        for rows in (self.records()[1:], self.records()[:-4]):
            with self.assertRaises(ValueError):
                costs.calibrate(rows, 32768)

    def test_receiver_staging_cost_is_included(self):
        rows = self.records()
        for row in rows:
            if row["kind"] == "transfer" and row["rank"] == 1:
                row["ms"] += 2
        self.assertAlmostEqual(costs.calibrate(rows, 32768)["buckets"][0]["fixed_ms"], 5)

    def test_nonphysical_fit_and_nan(self):
        for duration in (-1, float("nan"), 0):
            rows = self.records()
            rows[-1]["ms"] = duration
            rows[-2]["ms"] = duration
            with self.assertRaises(ValueError):
                costs.calibrate(rows, 32768)


if __name__ == "__main__":
    unittest.main()
