#!/usr/bin/env python3
"""Validate evidence interchange receipts; no engine launcher or state exporter."""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HEX = re.compile(r"[0-9a-f]{64}\Z")
HEX16 = re.compile(r"[0-9a-f]{16}\Z")
SOURCE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
ARMS = ("ref", "candidate")
VARIANTS = ((8192, 8192), (32768, 8192), (32768, 16384))
COMPONENTS = ("attention_kv", "recurrent_matrix", "gdn_convolution",
              "ple_convolution", "sparse_index", "ngram_history", "position_history")
IDENTITIES = ("decode_execution_identity", "decode_kernel_plan_identity",
              "prefill_execution_identity", "prefill_kernel_plan_identity",
              "tiled_execution_identity", "tiled_kernel_plan_identity")
DIAGNOSTICS = ("STRIX_PROMPT_STATE_DIGEST", "STRIX_FINAL_STATE_DIGEST",
               "STRIX_PREFIX_CACHE_DIGEST", "STRIX_PREFIX_CACHE_DIGEST_COMPONENTS",
               "STRIX_PREFILL_LOGITS_DIGEST", "STRIX_RTX_EXPORT_STATE",
               "STRIX_QRTX_STEP_DUMP", "STRIX_PROMPT_BATCH_ROWS", "STRIX_MTP_STEP_LOGITS_DIGEST",
               "STRIX_CYCLE_TIMING", "STRIX_CYCLE_TIMING_SLOW", "STRIX_SERIAL_LOGITS_DIGEST")
COMMON_KEYS = {"model_sha256", "assets_sha256", "vocab_size", "generate_tokens",
               "context_tokens", "slots", "world", "rank", "decode_profile", "prefill_profile",
               "profile_pair_identity", "rotary_sha256", "prefill_identity_hash",
               "prefill_options", "execution_policy_sha256", "full_attention_layers",
               "sampling", "mtp", "kv_format", "recurrent_storage", "tail_rows"}
ROUTES = {"session-components": "greedy-tiled-full-row-session", "target-state": "automatic-cli"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(value, expected, label):
    require(type(value) is dict and set(value) == set(expected), f"{label}: exact fields required")


def text(value):
    return type(value) is str and bool(value.strip())


def uint(value, minimum=0):
    return type(value) is int and minimum <= value <= 0xFFFFFFFF


def digest(value, pattern=HEX):
    return type(value) is str and pattern.fullmatch(value) is not None


def configuration(config):
    keys(config, COMMON_KEYS, "common configuration")
    for name in ("model_sha256", "rotary_sha256", "execution_policy_sha256"):
        require(digest(config[name]), f"invalid {name}")
    for name in ("prefill_identity_hash", "profile_pair_identity"):
        require(digest(config[name], HEX16), f"invalid {name}")
    assets = config["assets_sha256"]
    require(type(assets) is dict and all(text(k) and digest(v) for k, v in assets.items()), "asset identities required (empty only when none)")
    for name in ("vocab_size", "generate_tokens", "context_tokens"):
        require(uint(config[name], 1), f"invalid {name}")
    require(config["generate_tokens"] < config["context_tokens"], "generation exceeds capacity")
    require(type(config["slots"]) is int and config["slots"] == 1 and
            type(config["world"]) is int and config["world"] == 1 and
            type(config["rank"]) is int and config["rank"] == 0, "independent world=1 rank=0 slots=1 only")
    require(config["sampling"] == "greedy" and config["mtp"] is False, "plain greedy non-MTP only")
    require(config["kv_format"] == "bf16" and config["recurrent_storage"] == "key_value", "unsupported state storage")
    require(all(text(config[k]) for k in ("decode_profile", "prefill_profile")), "numerical profiles required")
    options = config["prefill_options"]
    require(type(options) is list and all(text(x) for x in options) and options == sorted(set(options)),
            "prefill_options must be sorted unique names")
    layers = config["full_attention_layers"]
    require(type(layers) is list and bool(layers) and all(uint(x) for x in layers) and
            layers == sorted(set(layers)), "full attention layer census required")
    tails = config["tail_rows"]
    require(type(tails) is list and all(uint(x, 1) and x < 8192 for x in tails) and
            len(tails) == len(set(tails)), "declared common tail plan required")


def plan(value, config, tile):
    keys(value, {"tile_rows", "tail_rows", "context_tokens", "slots", *IDENTITIES}, "native plan")
    require(type(value["tile_rows"]) is int and value["tile_rows"] == tile, "undeclared tile size")
    tails = value["tail_rows"]
    require(type(tails) is list and all(uint(x, 1) and x < tile for x in tails) and len(tails) == len(set(tails)), "invalid tail plan")
    require(tails == config["tail_rows"], "native tail plan differs from common configuration")
    for name in ("context_tokens", "slots"):
        require(type(value[name]) is int and value[name] == config[name], f"plan {name} mismatch")
    require(all(digest(value[k], HEX16) and int(value[k], 16) != 0 for k in IDENTITIES), "nonzero native plan identities required")


def parse_target_state(stderr):
    """Parse existing CLI lines; reject duplicate, error, partial and extra phases."""
    require(type(stderr) is str, "missing original stderr")
    result = {}
    for line in stderr.splitlines():
        if "qwen-target-state" not in line:
            continue
        match = re.fullmatch(r"qwen-target-state phase=(prompt|final) position=([0-9]+) sha256=([0-9a-f]{64})", line)
        require(match is not None, "malformed/unsupported qwen-target-state line")
        phase, position, sha = match.groups()
        require(phase not in result, f"duplicate target-state phase {phase}")
        require(phase == ("prompt" if not result else "final"), "target-state phases out of order")
        result[phase] = {"position": int(position), "sha256": sha}
    require(set(result) == {"prompt", "final"}, "UNSUPPORTED: missing canonical prompt/final target state")
    return result


def payloads(value, config):
    keys(value, COMPONENTS, "seven component groups")
    for component in COMPONENTS:
        names = ([f"kv.L{x}" for x in config["full_attention_layers"]] if component == "attention_kv"
                 else [f"index.L{x}" for x in config["full_attention_layers"]] if component == "sparse_index"
                 else [component])
        sections = value[component]
        require(type(sections) is list and len(sections) == len(names), f"{component}: incomplete range census")
        for section, name in zip(sections, names):
            keys(section, {"name", "bytes", "sha256"}, "payload section")
            require(section["name"] == name and type(section["bytes"]) is int and
                    section["bytes"] > 0 and digest(section["sha256"]), f"invalid section {name}")


def status(case):
    require(type(case["rc"]) is int and case["rc"] == 0 and case["declined"] is False and case["rejected"] is False, "failed/declined/rejected arm")


def provenance(case, nodes, builds):
    require(text(case["node"]) and case["node"] in nodes, "unexpected node")
    keys(case["build"], {"binary_sha256", "source_id"}, "per-arm build")
    require(digest(case["build"]["binary_sha256"]) and digest(case["build"]["source_id"], SOURCE), "binary/source identity required")
    arm = case["arm"]
    require(type(arm) is str and arm in ARMS, "unexpected arm")
    require(arm not in builds or builds[arm] == case["build"], "binary/source changed within arm")
    builds[arm] = case["build"]


def header(receipt, timed=False):
    keys(receipt, {"schema", "evidence", "timing_eligible", "adapter", "nodes", "configuration",
                   "prompts", "arms" if timed else "cases"}, "receipt root")
    require(receipt["schema"] == ("prefill-abba-v2" if timed else "prefill-state-v2"), "unsupported schema")
    require(receipt["timing_eligible"] is timed, "wrong timing eligibility")
    keys(receipt["adapter"], {"id", "source_id"}, "adapter provenance")
    require(text(receipt["adapter"]["id"]) and digest(receipt["adapter"]["source_id"], SOURCE),
            "adapter provenance required; no built-in engine exporter exists")
    nodes = receipt["nodes"]
    require(type(nodes) is dict and len(nodes) == 2 and all(text(x) for x in nodes), "exactly two independent nodes required")
    for node in nodes.values():
        keys(node, {"gpu_arch", "runtime", "driver", "kernel", "iommu_perfopt"}, "node environment")
        require(all(text(v) for v in node.values()), "node environment values required")
    config = receipt["configuration"]
    configuration(config)
    keys(receipt["prompts"], {"8192", "32768"}, "prompt identities")
    require(all(digest(x) for x in receipt["prompts"].values()), "actual prompt uint32-LE SHA-256 required")
    return config, nodes


def _validate(receipt):
    config, nodes = header(receipt)
    evidence = receipt["evidence"]
    require(type(evidence) is str and evidence in ROUTES, "UNSUPPORTED evidence/route class")
    cases = receipt["cases"]
    require(type(cases) is list, "cases must be a list")
    expected = {(node, n, tile, arm) for node in nodes for n, tile in VARIANTS for arm in ARMS}
    indexed, builds = {}, {}
    for case in cases:
        keys(case, {"node", "length", "tile_rows", "arm", "build", "rc", "declined", "rejected",
                    "requested_route", "effective_route", "configuration", "prompt_sha256", "plan",
                    "token_ids", "stderr", "stderr_sha256", "diagnostics", "evidence"}, "case")
        provenance(case, nodes, builds)
        require(uint(case["length"], 1) and uint(case["tile_rows"], 1), "invalid length/tile")
        key = (case["node"], case["length"], case["tile_rows"], case["arm"])
        require(key in expected and key not in indexed, "unexpected or duplicate case key")
        indexed[key] = case
        status(case)
        configuration(case["configuration"])
        require(case["configuration"] == config, "per-arm common configuration mismatch")
        require(case["prompt_sha256"] == receipt["prompts"][str(case["length"])],
                "actual prompt identity mismatch")
        require(case["requested_route"] == case["effective_route"] == ROUTES[evidence], "requested/effective route mismatch")
        require(case["diagnostics"] == dict.fromkeys(DIAGNOSTICS, None) | {"STRIX_PROMPT_STATE_DIGEST": "1"},
                "untimed state observer must be the only diagnostic")
        plan(case["plan"], config, case["tile_rows"])
        tokens = case["token_ids"]
        require(type(tokens) is list and len(tokens) == config["generate_tokens"] and
                all(uint(t) and t < config["vocab_size"] for t in tokens),
                "complete generated uint32 vocabulary token IDs required")
        require(case["length"] + len(tokens) <= config["context_tokens"], "insufficient context")
        require(type(case["stderr"]) is str and digest(case["stderr_sha256"]) and
                hashlib.sha256(case["stderr"].encode("utf-8")).hexdigest() == case["stderr_sha256"],
                "stderr artifact hash mismatch")
        states = parse_target_state(case["stderr"])
        require(states["prompt"]["position"] == case["length"] and
                states["final"]["position"] == case["length"] + len(tokens) - 1,
                "target-state frontier mismatch")
        ev = case["evidence"]
        if evidence == "session-components":
            keys(ev, {"payloads", "snapshot_sha256", "snapshot_validation", "frontier"}, "session evidence")
            require(digest(ev["snapshot_sha256"]), "snapshot artifact hash required")
            keys(ev["snapshot_validation"], {"envelope", "compatibility", "range_census", "metadata"}, "snapshot validation")
            require(all(v is True for v in ev["snapshot_validation"].values()), "snapshot validation incomplete")
            payloads(ev["payloads"], config)
            frontier = ev["frontier"]
            keys(frontier, {"consumed", "emitted", "accepted", "predictor", "pending", "sampling", "history_sha256"}, "session frontier")
            p = states["final"]["position"]
            require(all(type(frontier[k]) is int for k in
                        ("consumed", "emitted", "accepted", "predictor", "pending")) and
                    frontier["consumed"] == frontier["accepted"] == p and
                    frontier["emitted"] == len(tokens) and frontier["predictor"] == 0 and
                    frontier["pending"] == tokens[-1] and frontier["sampling"] == "greedy" and
                    digest(frontier["history_sha256"]), "invalid session frontier")
        else:
            keys(ev, {"observer", "native_plan_verified", "observer_off", "native_evidence_sha256"}, "observer evidence")
            require(ev["observer"] == "STRIX_PROMPT_STATE_DIGEST=1", "unsupported observer")
            require(ev["native_plan_verified"] is True and digest(ev["native_evidence_sha256"]),
                    "UNSUPPORTED: native observer plan preservation unverified")
            off = ev["observer_off"]
            keys(off, {"node", "build", "prompt_sha256", "configuration", "plan", "requested_route",
                       "effective_route", "diagnostics"}, "observer-off control")
            configuration(off["configuration"])
            plan(off["plan"], config, case["tile_rows"])
            require(off["node"] == case["node"] and off["build"] == case["build"] and
                    off["prompt_sha256"] == case["prompt_sha256"] and off["configuration"] == config and
                    off["plan"] == case["plan"] and
                    off["requested_route"] == off["effective_route"] == "automatic-cli" and
                    off["diagnostics"] == {k: None for k in DIAGNOSTICS},
                    "observer changed native execution plan/config/route/workload")
    require(set(indexed) == expected, "missing expected cases: both nodes and every declared schedule required")
    for node in nodes:
        for length in (8192, 32768):
            group = [c for (nd, n, _, _), c in indexed.items() if nd == node and n == length]
            first = group[0]
            for case in group[1:]:
                require(case["token_ids"] == first["token_ids"] and
                        parse_target_state(case["stderr"]) == parse_target_state(first["stderr"]),
                        "token or canonical numerical-state mismatch")
                # Physical execution hashes are NOT compared across schedules/binaries.
                if evidence == "session-components":
                    require(case["evidence"]["payloads"] == first["evidence"]["payloads"] and
                            case["evidence"]["frontier"] == first["evidence"]["frontier"],
                            "session payload/frontier mismatch")
    return f"VALIDATED-RECEIPT: {evidence}; two independent world=1 nodes, 12 cases; adapter assertions only"


def _validate_abba(receipt):
    config, nodes = header(receipt, timed=True)
    require(receipt["evidence"] == "timing-only", "ABBA is never state evidence")
    arms = receipt["arms"]
    require(type(arms) is list and len(arms) >= 8, "both nodes need complete ABBA quartets")
    builds, groups, plans = {}, {node: [] for node in nodes}, {}
    for arm in arms:
        keys(arm, {"node", "arm", "build", "rc", "declined", "rejected", "configuration", "length",
                   "prompt_sha256", "requested_route", "effective_route", "plan", "diagnostics",
                   "elapsed_ms"}, "timed arm")
        require(type(arm["arm"]) is str and arm["arm"] in ("A", "B"), "unexpected timed label")
        provenance(dict(arm, arm="ref" if arm["arm"] == "A" else "candidate"), nodes, builds)
        status(arm)
        configuration(arm["configuration"])
        require(arm["configuration"] == config, "timed configuration mismatch")
        require(type(arm["length"]) is int and arm["length"] == 32768 and
                arm["prompt_sha256"] == receipt["prompts"]["32768"], "timed workload identity mismatch")
        require(32768 + config["generate_tokens"] <= config["context_tokens"], "insufficient timed context")
        require(arm["requested_route"] == arm["effective_route"] == "automatic-cli", "timed automatic route required")
        plan(arm["plan"], config, 16384)
        require(arm["diagnostics"] == {k: None for k in DIAGNOSTICS}, "timed diagnostics must all be explicitly unset")
        elapsed = arm["elapsed_ms"]
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0, "elapsed_ms must be finite positive non-bool")
        node = arm["node"]
        groups[node].append(arm["arm"])
        key = (node, arm["arm"])
        require(key not in plans or plans[key] == arm["plan"], "timed native plan changed within arm")
        plans[key] = arm["plan"]
    for labels in groups.values():
        require(bool(labels) and len(labels) % 4 == 0 and labels == list("ABBA") * (len(labels) // 4),
                "every node must contain only complete ordered ABBA quartets")
    return "VALIDATED-TIMING-RECEIPT: structure only; NO-STATE-EQUIVALENCE; no speed verdict"


def checked(function, receipt):
    try:
        return True, [function(receipt)]
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        return False, [f"NO-STATE-EQUIVALENCE/UNSUPPORTED: {exc}"]


def validate(receipt):
    return checked(_validate, receipt)


def validate_abba(receipt):
    return checked(_validate_abba, receipt)


def load_receipt(serialized):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(serialized, object_pairs_hook=unique, parse_constant=invalid_constant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--timed-abba", action="store_true")
    args = parser.parse_args()
    try:
        receipt = load_receipt(args.receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        print(f"NO-STATE-EQUIVALENCE/UNSUPPORTED: {exc}", file=sys.stderr)
        return 2
    ok, messages = validate_abba(receipt) if args.timed_abba else validate(receipt)
    for message in messages:
        print(message, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
