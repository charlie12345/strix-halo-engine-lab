#!/usr/bin/env python3
"""Native R1 fusion ON/OFF gate: complete tokens and all seven state payloads.

Requires a GPU and model assets. Every process exports validation state; these
runs cannot establish throughput. All prompts, receipts and payloads stay private.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from qwen_prefill_public_controls_gate import state_signature


def signature(receipt_path, state_path):
    receipt = json.loads(receipt_path.read_text())
    state, components = state_signature(state_path / "manifest.json")
    tokens = receipt.get("generated_tokens")
    if not isinstance(tokens, list) or not tokens or tokens != state["generated_tokens"]:
        raise ValueError("NO-REFERENCE: missing or inconsistent complete token IDs")
    return tokens, state, components


def compare(off, on):
    if off != on:
        raise ValueError("MISMATCH: token IDs, consumed frontier or complete state payloads differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-1k", type=Path, required=True, help="1024 text token IDs")
    parser.add_argument("--prompt-32k", type=Path, required=True, help="32768 text token IDs")
    parser.add_argument("--output", type=Path, required=True, help="fresh directory outside Git")
    parser.add_argument("--generate", type=int, default=256)
    parser.add_argument("--route", choices=("eager", "islands", "both"), default="both")
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(checkout):
        parser.error("qualification outputs must stay outside the checkout")
    if args.generate < 1:
        parser.error("--generate must be positive")
    for count, prompt in ((1024, args.prompt_1k), (32768, args.prompt_32k)):
        ids = [int(v) for v in prompt.read_text().split()]
        if len(ids) != count or any(v < 0 for v in ids):
            parser.error(f"prompt must contain exactly {count} nonnegative token IDs")
    # Refuse broad suppression instead of silently reporting an ON/OFF pass
    # when neither arm could execute the code under test.
    if os.environ.get("STRIX_BASELINE") == "1" or os.environ.get("STRIX_QPF2K_DEFAULTS", "").startswith("0"):
        parser.error("remove baseline/defaults-off controls before running this gate")
    args.output.mkdir(exist_ok=False)
    routes = ("eager", "islands") if args.route == "both" else (args.route,)
    for route in routes:
        for count, prompt in ((1024, args.prompt_1k), (32768, args.prompt_32k)):
            results = []
            for arm in ("0", "1"):
                root = args.output / f"{route}-{count}-fuse{arm}"
                root.mkdir()
                command = [args.binary, args.model, "--prompt-ids", str(prompt),
                           "--context", str(count + args.generate + 128), "--unqualified-context",
                           "--generate", str(args.generate)]
                if route == "islands":
                    # A validation-state export excludes automatic generation
                    # organization. Pin the complete admitted island contract
                    # for this separate qualification route, in both arms.
                    command += ["--optimization-profile", "manual", "--gdn-islands",
                                "--deferred-moe-status", "--virtual-q8-gdn-input",
                                "--prefetch-q8-gdn-input", "--virtual-q8-gdn-output"]
                env = {**os.environ, "STRIX_QWEN_DECODE_FUSE": arm}
                with (root / "inspection.json").open("wb") as stdout, (root / "inspection.log").open("wb") as stderr:
                    subprocess.run(command + ["--print-effective-config"], env=env,
                                   stdout=stdout, stderr=stderr, check=True)
                inspected = json.loads((root / "inspection.json").read_text())
                if inspected["long_context_items"]["STRIX_QWEN_DECODE_FUSE"]["enabled"] != (arm == "1"):
                    raise ValueError("NO-WITNESS: resolver suppressed the requested fusion arm")
                command += ["--validation-final-state", str(root / "state"),
                            "--validation-final-state-max-bytes", "17179869184"]
                (root / "command.json").write_text(json.dumps(command) + "\n")
                with (root / "receipt.json").open("wb") as stdout, (root / "stderr.log").open("wb") as stderr:
                    subprocess.run(command, env=env, stdout=stdout, stderr=stderr, check=True)
                result = signature(root / "receipt.json", root / "state")
                if len(result[0]) != args.generate:
                    raise ValueError("NO-REFERENCE: incomplete requested decode workload (choose a continuation prompt)")
                results.append(result)
            compare(*results)
            print(f"{route} {count}: EXACT tokens and seven complete state payloads (no timing verdict)")


if __name__ == "__main__":
    main()
