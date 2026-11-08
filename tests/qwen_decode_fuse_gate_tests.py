"""CPU regressions: absent reference, token mismatch and each state component."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from qwen_decode_fuse_gate import compare, signature


class GateTests(unittest.TestCase):
    def test_compare_every_component_and_tokens(self):
        reference = ([3, 5, 8], {"generated_tokens": [3, 5, 8]},
                     {str(i): (4, "a" * 64) for i in range(7)})
        compare(reference, copy.deepcopy(reference))
        for component in reference[2]:
            changed = copy.deepcopy(reference)
            changed[2][component] = (4, "b" * 64)
            with self.subTest(component=component), self.assertRaisesRegex(ValueError, "MISMATCH"):
                compare(reference, changed)
        changed = copy.deepcopy(reference)
        changed[0][-1] = 9
        with self.assertRaisesRegex(ValueError, "MISMATCH"):
            compare(reference, changed)

    def test_actual_payloads_checked_and_no_reference_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({"generated_tokens": [2, 3]}))
            components = []
            for i in range(7):
                payload = bytes([i, 4, 5])
                (root / str(i)).write_bytes(payload)
                components.append({"component": str(i), "path": str(i), "bytes": len(payload),
                                   "sha256": hashlib.sha256(payload).hexdigest()})
            manifest = {"schema": "strix-final-generation-state-v1", "components": components,
                        "model_content_identity": "model", "prefix_tokens": [1], "generated_tokens": [2, 3],
                        "consumed_input_tokens": [1, 2], "context_capacity": 128,
                        "bonus_pending": True, "stop_reason": "max_new_tokens"}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            signature(receipt, root)
            (root / "6").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "payload"):
                signature(receipt, root)
            (root / "6").write_bytes(bytes([6, 4, 5]))
            receipt.write_text("{}")
            with self.assertRaisesRegex(ValueError, "NO-REFERENCE"):
                signature(receipt, root)
            receipt.write_text(json.dumps({"generated_tokens": [2, 3]}))
            manifest["components"].pop()
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "seven"):
                signature(receipt, root)


if __name__ == "__main__":
    unittest.main()
