#!/usr/bin/env python3
"""Census and ratchet of STRIX_* environment switches.

Every switch read from the environment is one of:
  governed  read through include/strix/env_flags.hpp (envcfg::flag/integer/text/choice, env_flag_default_on/off,
            env_bytes, env_choice, ...) or declared in a lever/profile table: validated, and listed by
            --print-effective-config;
  raw       read with getenv at its use site: unvalidated, unlisted, default known only to that line.

--check (the ctest) fails when
  * a switch is read raw that is not in the baseline (tools/env_switch_baseline.txt): a new switch must be read
    through env_flags.hpp or declared in a table, so the raw set can only shrink;
  * one boolean switch is read through the governed helpers with two different defaults.
Baseline names that are no longer read raw are reported and removed with --update (never added by hand).

--report prints the counts; --tsv FILE writes one row per switch (name, class, read sites, documented).
The scan is textual: a switch whose name is built at run time or passed through a wrapper is not seen.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

SOURCE_DIRS = ("src", "include", "apps")
SOURCE_EXT = {".cpp", ".hpp", ".h", ".hip", ".inc", ".inl", ".cu", ".cuh"}
DOC_PATHS = ("docs", "README.md", "AGENTS.md", "models", "recipes")
DOC_EXT = {".md", ".toml", ".txt", ".json", ".sh"}
HELPER_FILE = "include/strix/env_flags.hpp"

RAW = re.compile(r'\bgetenv\(\s*"(STRIX_[A-Z0-9_]+)"')
GOVERNED = re.compile(r'\b(envcfg::[a-z_]+|env_[a-z_0-9]+)\(\s*"(STRIX_[A-Z0-9_]+)"')
TABLE = re.compile(r'\.env\s*=\s*"(STRIX_[A-Z0-9_]+)"')
FLAG_DEFAULT = re.compile(
    r'\benvcfg::flag\(\s*"(STRIX_[A-Z0-9_]+)"\s*,\s*(true|false)\b|\benv_flag_default_(on|off)\(\s*"(STRIX_[A-Z0-9_]+)"')
NAME = re.compile(r'STRIX_[A-Z0-9_]+')


def source_files(root: Path):
    for top in SOURCE_DIRS:
        base = root / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix in SOURCE_EXT and path.is_file():
                yield path


def documented_names(root: Path) -> set[str]:
    names: set[str] = set()
    for entry in DOC_PATHS:
        path = root / entry
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.suffix in DOC_EXT) if path.is_dir() else []
        for file in files:
            try:
                names.update(NAME.findall(file.read_text(encoding="utf-8", errors="ignore")))
            except OSError:
                pass
    return names


def scan(root: Path):
    raw: dict[str, set[str]] = defaultdict(set)
    governed: dict[str, set[str]] = defaultdict(set)
    defaults: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for path in source_files(root):
        rel = path.relative_to(root).as_posix()
        if rel == HELPER_FILE:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "STRIX_" not in text:
            continue
        for name in RAW.findall(text):
            raw[name].add(rel)
        for helper, name in GOVERNED.findall(text):
            if helper not in ("env_reject",):
                governed[name].add(rel)
        for name in TABLE.findall(text):
            governed[name].add(rel)
        for a_name, a_default, b_default, b_name in FLAG_DEFAULT.findall(text):
            name = a_name or b_name
            default = "on" if (a_default == "true" or b_default == "on") else "off"
            defaults[name][default].add(rel)
    return raw, governed, defaults


def read_baseline(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}


def write_baseline(path: Path, names: set[str]) -> None:
    header = ("# Switches read with a raw getenv today (generated: tools/env_switch_census.py --update).\n"
              "# This list only shrinks. A new switch is read through include/strix/env_flags.hpp or declared in a\n"
              "# lever table (include/strix/lever.hpp); it is never added here.\n")
    path.write_text(header + "".join(name + "\n" for name in sorted(names)), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--baseline", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--update", action="store_true", help="drop baseline names that are no longer read raw")
    mode.add_argument("--seed", action="store_true", help="write the baseline from the current tree (first use only)")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--tsv", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    baseline_path = args.baseline or root / "tools" / "env_switch_baseline.txt"

    raw, governed, defaults = scan(root)
    baseline = read_baseline(baseline_path)
    conflicts = {name: sides for name, sides in defaults.items() if len(sides) > 1}
    new_raw = sorted(set(raw) - baseline)
    stale = sorted(baseline - set(raw))

    if args.seed:
        write_baseline(baseline_path, set(raw))
        print(f"env switch census: baseline seeded with {len(raw)} raw switches")
        return 0
    if args.update:
        write_baseline(baseline_path, baseline & set(raw))
        print(f"env switch census: baseline {len(baseline)} -> {len(baseline & set(raw))} (removed {len(stale)})")
        return 0

    if args.report or args.tsv:
        documented = documented_names(root)
        names = sorted(set(raw) | set(governed))
        only_governed = [n for n in names if n not in raw]
        multi = [n for n in names if len(raw.get(n, set()) | governed.get(n, set())) > 1]
        if args.report:
            print(f"switches read: {len(names)}")
            print(f"  governed only: {len(only_governed)}")
            print(f"  raw getenv: {len(raw)} (of which also governed somewhere: {len(set(raw) & set(governed))})")
            print(f"  read in more than one file: {len(multi)}")
            print(f"  documented (docs/, README, AGENTS, models/, recipes/): {len([n for n in names if n in documented])}")
            print(f"  boolean switches with two governed defaults: {len(conflicts)}")
        if args.tsv:
            rows = ["name\tclass\tfiles\tdocumented\n"]
            for n in names:
                files = sorted(raw.get(n, set()) | governed.get(n, set()))
                rows.append(f"{n}\t{'raw' if n in raw else 'governed'}\t{len(files)}\t{'yes' if n in documented else 'no'}\n")
            args.tsv.write_text("".join(rows), encoding="utf-8", newline="\n")

    if not args.check:
        return 0
    failed = False
    for name in new_raw:
        failed = True
        where = ", ".join(sorted(raw[name])[:3])
        print(f"env switch census: {name} is read with a raw getenv ({where}). Read a new switch through "
              f"include/strix/env_flags.hpp (envcfg::flag/integer/text/choice) or declare it in a lever table.",
              file=sys.stderr)
    for name, sides in sorted(conflicts.items()):
        failed = True
        on = ", ".join(sorted(sides["on"])[:2])
        off = ", ".join(sorted(sides["off"])[:2])
        print(f"env switch census: {name} is read default-on ({on}) and default-off ({off}).", file=sys.stderr)
    if stale:
        print(f"env switch census: {len(stale)} baseline switches are no longer read raw; "
              f"run tools/env_switch_census.py --update", file=sys.stderr)
    if failed:
        return 1
    print(f"env switch census: ok ({len(raw)} raw switches, none new; {len(set(governed) - set(raw))} governed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
