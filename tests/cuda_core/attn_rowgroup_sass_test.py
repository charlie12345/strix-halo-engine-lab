"""Compare the shared QSA kernel instances with the production reference offline."""

import argparse
import difflib
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench" / "cuda"))
from sass_compare import sass


def find_kernel(functions, name):
    token = f"{len(name)}{name}"
    matches = [(arch, body) for (arch, symbol), body in functions.items() if token in symbol]
    if len(matches) != 1 or matches[0][0] != "sm_120a":
        raise ValueError(f"expected one sm_120a instance of {name}")
    # cuobjdump's column widths vary with other functions in the object.
    return [re.sub(r"\s+", " ", line) for line in matches[0][1].splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qwen_object", help="production qwen_qsa CUDA object")
    parser.add_argument("fixture_object", help="shared row-group fixture CUDA object")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    args = parser.parse_args()
    try:
        qwen = sass(args.cuobjdump, args.qwen_object)
        shared = sass(args.cuobjdump, args.fixture_object)
        for old, new in (("qsa_union_kernel", "rowgroup_union_kernel"),
                         ("qsa_rowgroup_kernel", "rowgroup_flash_kernel")):
            expected, actual = find_kernel(qwen, old), find_kernel(shared, new)
            if expected != actual:
                delta = difflib.unified_diff(expected, actual, old, new, lineterm="")
                print("\n".join(line for _, line in zip(range(24), delta)), file=sys.stderr)
                raise ValueError(f"{old} differs from {new}")
            print(f"QSA sm_120a SASS PASS: {old} == {new}")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"QSA sm_120a SASS FAIL: {error}\n")


if __name__ == "__main__":
    main()
