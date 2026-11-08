"""Replicated policy-erased device-body identity against a replicated parent revision.

Pass the parent revision with --base (or STRIX_GLM_DSA_IDENTITY_BASE); without it, or
when that revision is not available in the checkout, the gate reports NOT RUN.
Source evidence only by default. --device additionally compares compiled bodies
and VGPR/LDS descriptors with amdclang++; it never launches a GPU kernel.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PATH = "src/glm/attn/glm_attn_fast.hip"
KERNELS = ("mla_scores_lane_kernel", "mla_scores_staged_kernel", "mla_softmax_kernel", "mla_pv_kernel_t")


def canonical(text):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S))


def close_brace(text, start):
    depth = 1
    end = start + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return end


def body(source, name):
    match = re.search(r"\b" + name + r"\([^;]*?\)\s*\{", source)
    assert match, name
    start = match.end() - 1
    return source[start:close_brace(source, start)]


def replica(source):
    source = canonical(source)
    marker = "ifconstexpr(Policy::partial){"
    while marker in source:
        start = source.index(marker)
        end = close_brace(source, start + len(marker) - 1)
        source = source[:start] + source[end:]
    for token in ("pos", "tok[tt<nr?tt:0]", "tok[t]"):
        source = source.replace(f"Policy::address({token},context...)", token)
    source = source.replace("&&Policy::owns(tok[t],context...)", "")
    source = source.replace("Policy::weight(sr[t],m)", "declared_exp(sr[t]-m)")
    source = source.replace("Policy::probability(e[mm],sum)", "e[mm]/sum")
    source = source.replace("Policy::output_index(static_cast<std::size_t>(r)*H+hg+hh,l0+li)",
                            "(static_cast<std::size_t>(r)*H+hg+hh)*kKvLora+l0+li")
    source = source.replace("floatv=acc[hh]", "constfloatv=acc[hh]")
    source = source.replace("floatm=key_to_float", "constfloatm=key_to_float")
    source = re.sub(r"(for\(unsignedli=0;li<D;\+\+li\))\{(ob\[.*?;)\}", r"\1\2", source)
    source = source.replace("<Policy>", "").replace(",Policy>", ">")
    return source.replace(",context...", "").replace("mla_softmax_kernel<>", "mla_softmax_kernel")


def source_gate(old, new, parent_engine):
    policy = canonical((ROOT / "src/glm/attn/glm_dsa_policy.hpp").read_text())
    replica_policy = policy.split("template<>structDsaAttentionPolicy<false>{", 1)[1].split(
        "template<>structDsaAttentionPolicy<true>{", 1)[0]
    for contract in ("partial=false;", "owns(std::uint32_t){returntrue;}",
                     "address(std::uint32_ttoken){returntoken;}",
                     "weight(floatscore,floatmaximum){returndeclared_exp(score-maximum);}",
                     "probability(floatweight,floatsum){returnweight/sum;}",
                     "returnhead*kKvLora+dim;"):
        assert contract in replica_policy, contract
    for name in KERNELS:
        assert canonical(body(old, name)) == replica(body(new, name)), name + " body"
        # Empty Context pack preserves the original replicated argument ABI.
        signature = r"\b" + name + r"(\([^;]*?\))\s*\{"
        before = canonical(re.search(signature, old).group(1))
        after = canonical(re.search(signature, new).group(1)).replace(",Context...context", "")
        assert before == after, name + " arguments"
    for name in ("mla_scores", "mla_pv"):
        assert canonical(body(old, name)) == replica(body(new, name + "_policy")), name + " dispatch"
        assert "DsaAttentionPolicy<false>" in body(new, name), name + " default"
    assert canonical(body(old, "mla_softmax")) == replica(body(new, "mla_softmax"))
    engine = (ROOT / "src/glm/engine/glm_kvshard.hip").read_text()
    assert "partial_kernel" not in engine
    assert engine.count("F::mla_partial(") == 2
    for query in ("impl_->peer_query", "work.qt"):
        assert f"F::mla_partial(rows, tokens, count, {query}," in engine
    assert canonical(body(parent_engine, "merge_kernel")) == canonical(body(engine, "merge_kernel"))
    assert "const unsigned max_tokens" in body(engine, "attention")
    assert "const unsigned max_tokens" not in body(engine, "prepare")
    assert canonical(body(old, "mla_attention")) == canonical(body(new, "mla_attention"))
    print("PASS: four replicated device bodies/argument lists, shared dispatch, schedules and default attention wiring")


def assembly_kernels(source):
    found = {}
    for match in re.finditer(r"^\s*\.type\s+(\S+),\s*@function", source, re.M):
        symbol = match.group(1)
        name = next((name for name in KERNELS if name in symbol), None)
        if name is None:
            continue
        # Partial instantiations contain the policy's true bool argument.
        if "DsaAttentionPolicyILb1" in symbol:
            continue
        start = re.search(r"^" + re.escape(symbol) + r":.*$", source[match.end():], re.M)
        assert start, symbol
        offset = match.end() + start.end()
        end = re.search(r"^\s*\.Lfunc_end\d+:", source[offset:], re.M)
        assert end, symbol
        code = source[offset:offset + end.start()]
        code = re.sub(r"(?:;|//).*", "", code)
        labels = {}
        code = re.sub(r"\.L[A-Za-z0-9_]+", lambda m: labels.setdefault(m[0], f"L{len(labels)}"), code)
        descriptor = re.search(r"\.amdhsa_kernel\s+" + re.escape(symbol) + r"\s+(.*?)\.end_amdhsa_kernel", source, re.S)
        assert descriptor, symbol
        resources = {}
        for field in ("next_free_vgpr", "group_segment_fixed_size"):
            resources[field] = re.search(r"\.amdhsa_" + field + r"\s+(\d+)", descriptor[1])[1]
        key = name
        if name == "mla_pv_kernel_t":
            key += re.search(r"ILj(\d+)ELj(\d+)E", symbol)[0]
        assert key not in found, key
        found[key] = (canonical(code), resources)
    assert len(found) == 6, list(found)  # lane, staged, softmax, three PV variants
    return found


def device_gate(old, new, args):
    with tempfile.TemporaryDirectory(prefix="glm-dsa-identity-") as directory:
        directory = Path(directory)
        assemblies = []
        for label, source in (("parent", old), ("policy", new)):
            src, asm = directory / (label + ".hip"), directory / (label + ".s")
            src.write_text(source)
            cmd = [args.amdclang, "-x", "hip", "--cuda-device-only", "--offload-arch=" + args.arch,
                   "-std=c++20", "-O3", "-mno-wavefrontsize64", "-fno-fast-math", "-ffp-contract=off",
                   "-fhip-fp32-correctly-rounded-divide-sqrt", "-fdenormal-fp-math=ieee",
                   "-fno-gpu-flush-denormals-to-zero", "-Werror", "-Wshadow", "-Wsign-conversion",
                   "-Wunused-function", "-Wcuda-compat", "-I", str(ROOT / "src"), "-I", str(ROOT / "include"),
                   "-S", str(src), "-o", str(asm)]
            result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert result.returncode == 0, "\n".join(result.stdout.splitlines()[-20:])
            assemblies.append(assembly_kernels(asm.read_text()))
        assert assemblies[0] == assemblies[1], "replicated device instructions or VGPR/LDS changed"
        for name, (_, resources) in assemblies[1].items():
            print("PASS:", name, resources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", action="store_true")
    parser.add_argument("--amdclang", default="amdclang++")
    parser.add_argument("--arch", default="gfx1151")
    parser.add_argument("--base", default=os.environ.get("STRIX_GLM_DSA_IDENTITY_BASE"))
    args = parser.parse_args()
    if not args.base:
        print("NOT RUN: parent revision not given (use --base)")
        return
    try:
        old = subprocess.check_output(["git", "show", args.base + ":" + PATH], cwd=ROOT, text=True,
                                      stderr=subprocess.DEVNULL)
        parent_engine = subprocess.check_output(
            ["git", "show", args.base + ":src/glm/engine/glm_kvshard.hip"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, OSError):
        print("NOT RUN: parent revision unavailable in this checkout:", args.base)
        return
    new = (ROOT / PATH).read_text()
    source_gate(old, new, parent_engine)
    if args.device:
        device_gate(old, new, args)
    else:
        print("NOT RUN: compiled device identity / VGPR / LDS (use --device; no GPU execution)")


if __name__ == "__main__":
    main()
