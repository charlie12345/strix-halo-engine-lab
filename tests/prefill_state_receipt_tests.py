"""Synthetic host fixtures only: these are not device qualification receipts."""
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/qualification/prefill_state_receipt.py"
spec = importlib.util.spec_from_file_location("state_receipt", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def config():
    return {"model_sha256": sha("synthetic model"), "assets_sha256": {}, "vocab_size": 100,
            "generate_tokens": 3, "context_tokens": 65536, "slots": 1, "world": 1, "rank": 0,
            "decode_profile": "synthetic-decode", "prefill_profile": "synthetic-prefill",
            "profile_pair_identity": "1" * 16, "rotary_sha256": sha("synthetic rotary"),
            "prefill_identity_hash": "2" * 16, "prefill_options": ["hc_output_wmma"],
            "execution_policy_sha256": sha("synthetic normalized policy"),
            "full_attention_layers": [3, 7], "sampling": "greedy", "mtp": False,
            "kv_format": "bf16", "recurrent_storage": "key_value", "tail_rows": [256, 16]}


def plan(tile, arm):
    return {"tile_rows": tile, "tail_rows": [256, 16], "context_tokens": 65536, "slots": 1,
            **{name: f"{tile + i + (100 if arm == 'candidate' else 0):016x}"
               for i, name in enumerate(gate.IDENTITIES)}}


def set_stderr(case, value):
    case["stderr"] = value
    case["stderr_sha256"] = sha(value)


def receipt(evidence="session-components"):
    r = {"schema": "prefill-state-v2", "evidence": evidence, "timing_eligible": False,
         "adapter": {"id": "synthetic-fixture-only", "source_id": "1" * 40},
         "nodes": {n: {k: "synthetic" for k in ("gpu_arch", "runtime", "driver", "kernel", "iommu_perfopt")}
                   for n in ("node-1", "node-2")}, "configuration": config(),
         "prompts": {str(n): sha(f"synthetic prompt {n}") for n in (8192, 32768)}, "cases": []}
    for node in r["nodes"]:
        for length, tile in gate.VARIANTS:
            for arm in gate.ARMS:
                c = {"node": node, "length": length, "tile_rows": tile, "arm": arm,
                     "build": {"binary_sha256": sha(arm), "source_id": ("a" if arm == "ref" else "b") * 40},
                     "rc": 0, "declined": False, "rejected": False,
                     "requested_route": gate.ROUTES[evidence], "effective_route": gate.ROUTES[evidence],
                     "configuration": config(), "prompt_sha256": r["prompts"][str(length)],
                     "plan": plan(tile, arm), "token_ids": [1, 2, 3],
                     "diagnostics": dict.fromkeys(gate.DIAGNOSTICS) | {"STRIX_PROMPT_STATE_DIGEST": "1"}}
                set_stderr(c, "ordinary diagnostic\n" +
                           f"qwen-target-state phase=prompt position={length} sha256={sha('prompt')}\n" +
                           f"qwen-target-state phase=final position={length + 2} sha256={sha('final')}\n")
                if evidence == "session-components":
                    payloads = {}
                    for component in gate.COMPONENTS:
                        names = (["kv.L3", "kv.L7"] if component == "attention_kv" else
                                 ["index.L3", "index.L7"] if component == "sparse_index" else [component])
                        payloads[component] = [{"name": n, "bytes": 64, "sha256": sha(n)} for n in names]
                    c["evidence"] = {"payloads": payloads, "snapshot_sha256": sha(f"{arm}/{tile}/envelope"),
                                     "snapshot_validation": {k: True for k in ("envelope", "compatibility", "range_census", "metadata")},
                                     "frontier": {"consumed": length + 2, "accepted": length + 2,
                                                  "emitted": 3, "predictor": 0, "pending": 3,
                                                  "sampling": "greedy", "history_sha256": sha(str(length))}}
                else:
                    c["evidence"] = {"observer": "STRIX_PROMPT_STATE_DIGEST=1", "native_plan_verified": True,
                                     "native_evidence_sha256": sha("synthetic native plan comparison"),
                                     "observer_off": {"node": node, "build": copy.deepcopy(c["build"]),
                                                      "prompt_sha256": c["prompt_sha256"],
                                                      "configuration": config(), "plan": copy.deepcopy(c["plan"]),
                                                      "requested_route": "automatic-cli", "effective_route": "automatic-cli",
                                                      "diagnostics": dict.fromkeys(gate.DIAGNOSTICS)}}
                r["cases"].append(c)
    return r


def abba(rounds=1):
    r = receipt("target-state")
    r.update(schema="prefill-abba-v2", evidence="timing-only", timing_eligible=True)
    r.pop("cases")
    r["arms"] = []
    for node in r["nodes"]:
        for label in "ABBA" * rounds:
            arm = "ref" if label == "A" else "candidate"
            r["arms"].append({"node": node, "arm": label, "build": {"binary_sha256": sha(arm),
                              "source_id": ("a" if label == "A" else "b") * 40},
                              "rc": 0, "declined": False, "rejected": False, "configuration": config(),
                              "length": 32768, "prompt_sha256": r["prompts"]["32768"],
                              "requested_route": "automatic-cli", "effective_route": "automatic-cli",
                              "plan": plan(16384, arm), "diagnostics": dict.fromkeys(gate.DIAGNOSTICS),
                              "elapsed_ms": 12.5})
    return r


class StateReceiptTests(unittest.TestCase):
    def rejects(self, r, message=None, timed=False):
        ok, messages = (gate.validate_abba if timed else gate.validate)(r)
        self.assertFalse(ok, messages)
        self.assertIn("NO-STATE-EQUIVALENCE", messages[0])
        if message:
            self.assertIn(message, messages[0])

    def test_supported_contracts_and_serialized_roundtrip(self):
        for kind in gate.ROUTES:
            with self.subTest(kind=kind):
                r = gate.load_receipt(json.dumps(receipt(kind)))
                self.assertTrue(gate.validate(r)[0])
                # Physical hashes differ across binaries/tiles; numerical states agree.
                self.assertNotEqual(r["cases"][2]["plan"], r["cases"][4]["plan"])

    def test_exact_matrix_duplicate_missing_extra_and_second_node(self):
        for mutation in (lambda r: r["cases"].append(copy.deepcopy(r["cases"][0])),
                         lambda r: r["cases"].pop(),
                         lambda r: r["cases"][0].update(length=12345),
                         lambda r: r["nodes"].pop("node-2"),
                         lambda r: r.update(cases=r["cases"][:6])):
            r = receipt(); mutation(r); self.rejects(r)

    def test_payload_names_census_sizes_and_mutation(self):
        mutations = [lambda p: p.update(arbitrary=p.pop("position_history")),
                     lambda p: p["attention_kv"].pop(),
                     lambda p: p["attention_kv"][1].update(name="kv.L3"),
                     lambda p: p["recurrent_matrix"][0].update(bytes=True),
                     lambda p: p["recurrent_matrix"][0].update(sha256="f" * 64)]
        for mutation in mutations:
            r = receipt(); mutation(r["cases"][0]["evidence"]["payloads"]); self.rejects(r)
        r = receipt(); r["cases"][0]["evidence"]["snapshot_validation"]["metadata"] = False
        self.rejects(r)

    def test_canonical_common_configuration_and_per_arm_identity(self):
        for bad in ({"build": "sha", "plan": "default"}, {}, None):
            r = receipt(); r["configuration"] = bad; self.rejects(r)
        for field, value in (("model_sha256", "f" * 64), ("context_tokens", 65537),
                             ("prefill_identity_hash", "f" * 16), ("sampling", "random")):
            r = receipt(); r["cases"][0]["configuration"][field] = value; self.rejects(r)
        r = receipt(); r["configuration"]["binary_sha256"] = sha("build")
        self.rejects(r, "exact fields")

    def test_build_node_prompt_and_schedule_provenance(self):
        for change in ({"build": {}}, {"node": "unknown"}, {"prompt_sha256": sha("different")},
                       {"tile_rows": 4096}, {"requested_route": "automatic-cli"}, {"effective_route": "automatic-cli"}):
            r = receipt(); r["cases"][0].update(change); self.rejects(r)
        r = receipt(); r["cases"][0]["build"]["source_id"] = "c" * 40
        self.rejects(r, "changed within arm")
        r = receipt(); r["cases"][0]["plan"]["tile_rows"] = 16384
        self.rejects(r, "tile")

    def test_invalid_tokens_and_process_status(self):
        for tokens in ([True, 2, 3], ["1", 2, 3], [-1, 2, 3], [100, 2, 3],
                       [None, 2, 3], [1.0, 2, 3], [[1], 2, 3], [], [1, 2]):
            r = receipt(); r["cases"][0]["token_ids"] = tokens; self.rejects(r, "token IDs")
        for change in ({"rc": False}, {"rc": True}, {"rc": 1}, {"declined": True}, {"rejected": 0}):
            r = receipt(); r["cases"][0].update(change); self.rejects(r, "failed/declined")

    def test_state_and_token_equality_on_both_nodes_and_tiles(self):
        for index in (1, 4, 11):
            r = receipt(); r["cases"][index]["token_ids"][0] = 9
            self.rejects(r, "numerical-state mismatch")
            r = receipt(); c = r["cases"][index]
            set_stderr(c, c["stderr"].replace(sha("final"), sha("changed")))
            self.rejects(r, "numerical-state mismatch")
        r = receipt(); r["cases"][0]["evidence"]["frontier"]["history_sha256"] = sha("wrong history")
        self.rejects(r, "payload/frontier mismatch")

    def test_target_parser_rejects_missing_duplicate_error_and_wrong_frontier(self):
        for transform in (lambda s: "", lambda s: s.splitlines()[-1] + "\n", lambda s: s + s,
                          lambda s: s.replace("phase=final", "phase=other"),
                          lambda s: s.replace("position=8194", "position=1"),
                          lambda s: "\n".join(reversed(s.splitlines())) + "\n",
                          lambda s: s + "qwen-target-state phase=final position=8194 error=copy failed\n"):
            r = receipt("target-state"); c = r["cases"][0]
            set_stderr(c, transform(c["stderr"])); self.rejects(r)
        r = receipt("target-state"); r["cases"][0]["stderr_sha256"] = sha("wrong artifact")
        self.rejects(r, "artifact")

    def test_observer_native_preservation_is_required(self):
        for mutation in (lambda ev: ev.update(native_plan_verified=False),
                         lambda ev: ev.pop("native_evidence_sha256"),
                         lambda ev: ev["observer_off"]["plan"].update(tile_rows=16384),
                         lambda ev: ev["observer_off"]["plan"].update(decode_kernel_plan_identity="f" * 16),
                         lambda ev: ev["observer_off"]["configuration"].update(context_tokens=65537),
                         lambda ev: ev["observer_off"].update(node="node-2"),
                         lambda ev: ev["observer_off"]["build"].update(binary_sha256="f" * 64),
                         lambda ev: ev["observer_off"].update(prompt_sha256="f" * 64),
                         lambda ev: ev["observer_off"].update(diagnostics={})):
            r = receipt("target-state"); mutation(r["cases"][0]["evidence"]); self.rejects(r)

    def test_session_evidence_cannot_substitute_for_automatic(self):
        r = receipt(); r["evidence"] = "target-state"
        for c in r["cases"]:
            c.update(requested_route="automatic-cli", effective_route="automatic-cli")
        self.rejects(r, "observer evidence")
        r = receipt("target-state"); r["evidence"] = "session-components"
        for c in r["cases"]:
            c.update(requested_route=gate.ROUTES["session-components"], effective_route=gate.ROUTES["session-components"])
        self.rejects(r, "session evidence")

    def test_malformed_roots_and_nested_types_fail_without_exception(self):
        for value in (None, [], True, "text", 12):
            self.rejects(value); self.rejects(value, timed=True)
        for field in ("cases", "nodes", "adapter", "configuration", "prompts"):
            r = receipt(); r[field] = None; self.rejects(r)
        r = receipt(); r["cases"][0] = []
        self.rejects(r)

    def test_abba_complete_rounds(self):
        for rounds in (1, 2):
            self.assertTrue(gate.validate_abba(gate.load_receipt(json.dumps(abba(rounds))))[0])
        self.rejects(abba(), timed=False)

    def test_abba_missing_config_nonfinite_bool_and_status(self):
        for value in (None, {}, {"build": "anything"}):
            r = abba()
            for a in r["arms"]:
                a["configuration"] = value
            self.rejects(r, timed=True)
        for value in (True, False, float("nan"), float("inf"), -float("inf"), 0, -1, "12"):
            r = abba(); r["arms"][-1]["elapsed_ms"] = value
            self.rejects(r, "finite positive", timed=True)
        r = abba(); r["arms"][-1]["rc"] = False
        self.rejects(r, timed=True)

    def test_abba_every_quartet_and_node_and_diagnostics(self):
        r = abba(); r["arms"].append(copy.deepcopy(r["arms"][-1]))
        self.rejects(r, "complete ordered", timed=True)
        r = abba(2); r["arms"][5], r["arms"][7] = r["arms"][7], r["arms"][5]
        self.rejects(r, "complete ordered", timed=True)
        r = abba(); r["arms"][4]["arm"] = "junk"
        self.rejects(r, "label", timed=True)
        r = abba(); r["arms"] = r["arms"][:4]
        self.rejects(r, timed=True)
        for diagnostics in ({}, dict.fromkeys(gate.DIAGNOSTICS, "0")):
            r = abba(); r["arms"][0]["diagnostics"] = diagnostics
            self.rejects(r, "explicitly unset", timed=True)
        r = abba(); r["arms"][-1]["plan"]["tail_rows"] = []
        self.rejects(r, "tail plan", timed=True)
        r = abba(); r["arms"][-1]["plan"]["tiled_execution_identity"] = "f" * 16
        self.rejects(r, "plan changed", timed=True)

    def test_serialized_duplicate_keys_and_nonfinite_constants(self):
        for serialized in ('{"schema": 1, "schema": 2}', '{"a":{"x":1,"x":2}}',
                           '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'):
            with self.assertRaises(ValueError):
                gate.load_receipt(serialized)

    def test_cli_exit_codes_and_fail_closed_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            no_state = receipt("target-state")
            set_stderr(no_state["cases"][0], "ordinary diagnostic only\n")
            unverified = receipt("target-state")
            unverified["cases"][0]["evidence"]["native_plan_verified"] = False
            duplicate = receipt()
            duplicate["cases"].append(copy.deepcopy(duplicate["cases"][0]))
            junk = abba()
            junk["arms"].append(copy.deepcopy(junk["arms"][-1]))
            nonfinite = abba()
            nonfinite["arms"][-1]["elapsed_ms"] = float("inf")
            samples = [(json.dumps(receipt()), [], 0), (json.dumps(receipt("target-state")), [], 0),
                       (json.dumps(abba()), ["--timed-abba"], 0),
                       (json.dumps(no_state), [], 1), (json.dumps(unverified), [], 1),
                       (json.dumps(duplicate), [], 1), (json.dumps(junk), ["--timed-abba"], 1),
                       (json.dumps(nonfinite).replace("Infinity", "1e999"), ["--timed-abba"], 1),
                       ("null", [], 1), ("[]", ["--timed-abba"], 1),
                       ('{"x":1,"x":2}', [], 2), ("{", [], 2),
                       ('{"x":NaN}', ["--timed-abba"], 2)]
            for serialized, flags, expected in samples:
                path.write_text(serialized, encoding="utf-8")
                run = subprocess.run([sys.executable, str(SCRIPT), str(path), *flags], capture_output=True, text=True)
                self.assertEqual(run.returncode, expected, run.stderr)
                self.assertNotIn("Traceback", run.stderr)
                if expected:
                    self.assertIn("NO-STATE-EQUIVALENCE/UNSUPPORTED", run.stderr)
                    self.assertNotIn("VALIDATED", run.stdout)
            run = subprocess.run([sys.executable, str(SCRIPT), str(path.with_name("absent.json"))], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            path.write_bytes(b"\xff\xfe")
            run = subprocess.run([sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertIn("NO-STATE-EQUIVALENCE/UNSUPPORTED", run.stderr)


if __name__ == "__main__":
    unittest.main()
