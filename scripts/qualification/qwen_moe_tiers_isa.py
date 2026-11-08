#!/usr/bin/env python3
"""Offline DA16 instruction-byte and tier resource audit; never opens a device.

Supply gfx1151 device-only ELF objects built with identical compiler, headers,
device libraries and flags, before and after the change. Keep output out of Git.
"""
import argparse
import csv
import hashlib
from pathlib import Path
import re
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from check_kernel_scratch import kernel_rows


def instructions(path, kind=2):
    """Bytes of every defined symbol of ELF type `kind` (2 = function, 1 = object)."""
    data = path.read_bytes()
    if data[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", data, 18)[0] != 224:
        raise ValueError("expected a little-endian ELF64 AMDGPU device object")
    offset = struct.unpack_from("<Q", data, 40)[0]
    size, count = struct.unpack_from("<HH", data, 58)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * size) for i in range(count)]
    result = {}
    for section in sections:
        if section[1] != 2:  # SHT_SYMTAB
            continue
        strings = sections[section[6]]
        strings = data[strings[4]:strings[4] + strings[5]]
        for at in range(section[4], section[4] + section[5], section[9]):
            name, info, _, index, value, length = struct.unpack_from("<IBBHQQ", data, at)
            if info & 15 != kind or index == 0 or index >= count:
                continue
            symbol = strings[name:].split(b"\0", 1)[0].decode()
            text = sections[index]
            begin = text[4] + value - text[3]
            result[symbol] = data[begin:begin + length]
    return result


def descriptors(path):
    """Kernel descriptors with kernel_code_entry_byte_offset (bytes 16..23) masked:
    that field is the distance to the entry point and moves with text layout."""
    return {name[:-3]: blob[:16] + bytes(8) + blob[24:]
            for name, blob in instructions(path, 1).items() if name.endswith(".kd") and len(blob) == 64}


def audit(args):
    for path in (args.before, args.after):
        header = subprocess.check_output([args.readelf, "--file-header", str(path)], text=True)
        if "GFX1151" not in header.upper():
            raise ValueError(f"expected gfx1151 ELF flags: {path}")
    old, new = instructions(args.before), instructions(args.after)
    rows = kernel_rows(args.after)
    names = sorted(set(old) | set(new) | {r["name"] for r in rows})
    demangled = subprocess.check_output([args.cxxfilt], input="\n".join(names) + "\n", text=True).splitlines()
    names = dict(zip(names, demangled))
    legacy = re.compile(r"::(?:project_prefill_f16x_da16(?:_mg)?|down_prefill_f16x_da16(?:_certificate(?:_bf16)?)?)<")
    existing = [s for s in old if legacy.search(names[s])]
    if len(existing) != 30:
        raise ValueError(f"expected exactly 30 incumbent DA16 bodies, got {len(existing)}")
    old_kd, new_kd = descriptors(args.before), descriptors(args.after)
    if any(s not in old_kd for s in existing):
        raise ValueError("incumbent DA16 body without a 64-byte kernel descriptor")
    changed = [names[s] for s in existing if s not in new or old[s] != new[s]]
    changed += [names[s] + " (descriptor)" for s in existing if old_kd[s] != new_kd.get(s)]
    added = [r for r in rows if "::project_prefill_da16_tier<" in names[r["name"]]
             or "::down_prefill_da16_tier<" in names[r["name"]]
             or "::build_prefill_tiers(" in names[r["name"]]]
    if len(added) != 49:
        raise ValueError(f"expected 48 tier specializations and list builder, got {len(added)}")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "resources.csv").open("w", newline="") as output:
        fields = ["name", "vgpr", "sgpr", "lds", "private", "vgpr_spill", "sgpr_spill", "dynamic_stack",
                  "threads", "ctas_wgp_lds", "ctas_wgp_vgpr", "ctas_wgp_bound"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in sorted(added, key=lambda r: names[r["name"]]):
            name = names[row["name"]]
            tune = re.search(r"_tier<(?:true|false), (80|82|88)u", name)
            threads = 512 if tune and tune[1] == "82" else 256
            waves = threads // 32
            lds = (row["lds"] + 255) // 256 * 256
            lds_ctas = 131072 // lds if lds else 16
            # gfx1151 wave32: 1536 VGPRs/SIMD, allocation granularity 24,
            # four SIMDs/WGP, at most 16 waves/SIMD and 16 CTAs/WGP.
            vgpr_waves = min(16, 1536 // max(24, (row["vgpr"] + 23) // 24 * 24))
            vgpr_ctas = 4 * vgpr_waves // waves
            writer.writerow({**{k: row[k] for k in fields if k in row}, "name": name,
                             "threads": threads, "ctas_wgp_lds": lds_ctas,
                             "ctas_wgp_vgpr": vgpr_ctas, "ctas_wgp_bound": min(16, lds_ctas, vgpr_ctas, 64 // waves)})
    with (args.out / "incumbent-bytes.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["name", "bytes", "before_sha256", "after_sha256", "equal", "descriptor_equal"])
        for symbol in sorted(existing):
            writer.writerow([names[symbol], len(old[symbol]), hashlib.sha256(old[symbol]).hexdigest(),
                             hashlib.sha256(new.get(symbol, b"")).hexdigest(), old[symbol] == new.get(symbol),
                             old_kd[symbol] == new_kd.get(symbol)])
    bad = [names[r["name"]] for r in added if r["private"] or r["vgpr_spill"] or r["sgpr_spill"] or r["dynamic_stack"]]
    print(f"incumbent DA16 instruction bytes + descriptors: {30 - len({c.split(' (')[0] for c in changed})}/30 identical")
    print(f"new tier/list bodies: {len(added)}, nonzero scratch/spills/dynamic stack: {len(bad)}")
    for name in changed + bad:
        print("FAIL:", name)
    if changed or bad:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--cxxfilt", default="llvm-cxxfilt")
    parser.add_argument("--readelf", default="llvm-readelf")
    parser.add_argument("--out", type=Path, required=True)
    audit(parser.parse_args())
