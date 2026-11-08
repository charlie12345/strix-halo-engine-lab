"""Single-node layer-split equality gate. Artifacts belong in a private directory outside Git.
Usage: python tests/qwen_prefill_pipeline_gpu_gate.py --output PRIVATE_DIR -- COMMAND MODEL --tokens-file PROMPT --generate 1
The command must select tiled prefill and enable the normal full-state registry.
Missing seam/logit/state evidence is NO-REFERENCE, never an exactness pass.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a complete native Qwen command is required")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    results = {}
    for name in ("full", "split"):
        env = os.environ.copy()
        env.pop("STRIX_QWEN_PF_LAYER_RANGE", None)
        env.pop("STRIX_QWEN_PF_SPLIT_TEST", None)
        env["STRIX_PREFILL_LOGITS_DIGEST"] = "1"
        env["STRIX_FINAL_STATE_DIGEST"] = "1"
        if name == "split":
            env["STRIX_QWEN_PF_LAYER_RANGE"] = "0:24"
            env["STRIX_QWEN_PF_SPLIT_TEST"] = "1"
        raw = root / (name + "-raw")
        run_command = command + ["--record-raw", str(raw)]
        with (root / (name + ".stdout")).open("wb") as stdout, (root / (name + ".stderr")).open("wb") as stderr:
            completed = subprocess.run(run_command, env=env, stdout=stdout, stderr=stderr, check=False)
        if completed.returncode:
            print(f"NOT RUN: {name} command failed ({completed.returncode}); inspect private logs")
            return 2
        text = (root / (name + ".stderr")).read_text(errors="replace")
        results[name] = {
            "logits": re.findall(r"prefill terminal logits fnv64 [0-9a-f]+ position [0-9]+", text),
            "state": re.findall(r"qwen-target-state phase=prompt position=[0-9]+ sha256=[0-9a-f]{64}", text),
            "raw": {item.name: item for item in raw.glob("step-*.f32le")},
            "save": re.findall(r"qwen-split seam=save rows=[0-9]+ bytes-per-row=25616 position=[0-9]+", text),
            "load": re.findall(r"qwen-split seam=load rows=[0-9]+ bytes-per-row=25616 position=[0-9]+", text),
        }
    full, split = results["full"], results["split"]
    if (not full["logits"] or not full["state"] or not full["raw"] or
            not split["logits"] or not split["state"] or not split["raw"] or
            not split["save"] or [line.replace("seam=save", "seam=load") for line in split["save"]] != split["load"]):
        print("NO-REFERENCE: require terminal logits, raw logits, prompt full-state digest and paired seam receipts")
        return 2
    match = (full["logits"] == split["logits"] and full["state"] == split["state"] and
             full["raw"].keys() == split["raw"].keys())
    if match:
        for filename, full_path in full["raw"].items():
            with full_path.open("rb") as first, split["raw"][filename].open("rb") as second:
                while True:
                    left, right = first.read(1024 * 1024), second.read(1024 * 1024)
                    if left != right:
                        match = False
                        break
                    if not left:
                        break
            if not match:
                break
    print("EXACT" if match else "MISMATCH")
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
