"""Ticket failures must survive the shell wrapper, without a GPU."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualification/qwen-moe-tile64-ticket.sh"
BASH = shutil.which("bash")


@unittest.skipUnless(BASH, "bash is required")
class TicketTests(unittest.TestCase):
    def run_ticket(self, fail="", missing=False):
        cache = ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tile64-ticket-", dir=cache) as directory:
            build = Path(directory)
            binary = build / "strix-hip-moe-wmma-prefill-tests"
            binary.write_text(
                '#!/usr/bin/env bash\n'
                'printf "%s\\n" "$*" >> "$FAKE_TRACE"\n'
                'if [[ $1 == tile64 && ${FAKE_FAIL:-} == gate ]]; then exit 7; fi\n'
                'if [[ $1 == tile64time && $3 == *:12:8 && ${FAKE_FAIL:-} == first ]]; then exit 8; fi\n'
                'if [[ $1 == tile64time && $3 == *:12:7 && ${FAKE_FAIL:-} == second ]]; then exit 9; fi\n'
                'exit 0\n', encoding="utf-8", newline="\n")
            binary.chmod(0o755)
            trace = build / "trace.txt"
            env = os.environ.copy()
            for name in ("LAYER_SPEC_Q4_Q51", "LAYER_SPEC_Q5_Q80", "LAYER_SPEC_Q4_Q80"):
                env.pop(name, None)
            env.update(FAKE_TRACE=trace.as_posix(), FAKE_FAIL=fail)
            if not missing:
                env.update(LAYER_SPEC_Q4_Q51="assets/model.gguf:1:2:3:12:7",
                           LAYER_SPEC_Q5_Q80="assets/model.gguf:4:5:6:13:8",
                           LAYER_SPEC_Q4_Q80="assets/model.gguf:7:8:9:12:8")
            result = subprocess.run([BASH, SCRIPT.as_posix(), build.as_posix()],
                                    env=env, text=True, capture_output=True, timeout=20)
            calls = trace.read_text().splitlines() if trace.exists() else []
            return result, calls

    def test_success_requires_all_three_phases(self):
        result, calls = self.run_ticket()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 3)
        self.assertTrue(calls[0].startswith("tile64 "))
        self.assertTrue(all(call.startswith("tile64time ") for call in calls[1:]))
        self.assertIn("ticket wall", result.stdout)

    def test_gate_failure_prevents_timing(self):
        result, calls = self.run_ticket("gate")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("ticket wall", result.stdout)

    def test_first_timing_failure_is_not_hidden_by_gate_success(self):
        result, calls = self.run_ticket("first")
        self.assertEqual(result.returncode, 8)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("ticket wall", result.stdout)

    def test_second_timing_failure_is_not_hidden_by_gate_success(self):
        result, calls = self.run_ticket("second")
        self.assertEqual(result.returncode, 9)
        self.assertEqual(len(calls), 3)
        self.assertNotIn("ticket wall", result.stdout)

    def test_missing_asset_specs_fail_before_execution(self):
        result, calls = self.run_ticket(missing=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
