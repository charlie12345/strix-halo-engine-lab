#!/usr/bin/env python3
"""MiniMax serve prefix-reuse exactness gate.

Prefix KV reuse (opt-in: STRIX_MM_SERVE_PREFIX=1 on both ranks) lets a request that shares a leading token prefix
with a free slot's resident cache skip prefilling that prefix and reuse its KV rows. Causal attention makes the reused
rows the rows a full prefill writes, so the emitted tokens must not change. This gate checks that at the serve level
and that reuse actually fired (a silently skipped reuse would "pass" a plain token comparison).

It needs the running two-rank MiniMax pair and a prompt text file, so it is not a CI test. The comparison logic is
CPU-only and self-checked with `self-test` (no server).

Procedure (the switch is read once at startup, so each arm is a fresh pair):

    # arm ON  (reuse on):  start both ranks with STRIX_MM_SERVE_PREFIX=1, rank 0 on :PORT, then
    python3 tests/minimax_prefix_cache_gpu_gate.py capture --port PORT --prompt PROMPT.txt --out mm-on.json
    # arm OFF (reuse off): start both ranks without STRIX_MM_SERVE_PREFIX, then
    python3 tests/minimax_prefix_cache_gpu_gate.py capture --port PORT --prompt PROMPT.txt --out mm-off.json
    python3 tests/minimax_prefix_cache_gpu_gate.py compare --on mm-on.json --off mm-off.json

A capture sends A = the prompt (at least one prefill chunk, 4096 tokens; use several), lets it finish (freeing its slot but keeping its resident KV), then
sends B = A + a short distinct suffix. In the ON arm B reuses the leading tokens it shares with A (cached delta >=
that shared length rounded down to whole chunks); in the OFF arm B prefills everything (cached delta 0). compare requires: token_ids(A) and
token_ids(B) identical across arms; ON reused >= the whole chunks of the shared length; OFF reused 0. Verdict EXACT, MISMATCH or NO-REFERENCE (missing diagnostics is never a pass).
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

CHUNK = 4096  # the serve's default --max-rows: reuse is rounded down to whole prefill chunks.


def require(value, message):
    if not value:
        raise RuntimeError(message)


def post(port, path, payload, timeout=1800):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def tokenize(port, text):
    """The serve's own BPE: text -> token ids (POST /tokenize). The prompt must be a string, not an id array."""
    reply = post(port, "/tokenize", dict(text=text), timeout=60)
    ids = reply.get("tokens")
    require(ids, "NO-REFERENCE: /tokenize returned no tokens")
    return [int(t) for t in ids]


def completion(port, text, count):
    """Greedy (temperature 0) text completion returning the ordered generated token ids."""
    reply = post(port, "/v1/completions",
                 dict(prompt=text, max_tokens=count, temperature=0, stream=False, return_token_ids=True))
    require("error" not in reply, f"server error: {reply.get('error')}")
    ids = (reply.get("strix") or {}).get("token_ids")
    require(ids, "NO-REFERENCE: reply carries no strix.token_ids (send return_token_ids and a build that emits them)")
    return [int(t) for t in ids]


def cached_total(port):
    """The strix_prompt_tokens_cached_total counter: prompt tokens reused from a resident prefix instead of prefilled."""
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=60) as response:
        text = response.read().decode()
    for line in text.splitlines():
        if line.startswith("strix_prompt_tokens_cached_total ") and not line.startswith("#"):
            return int(float(line.split()[1]))
    raise RuntimeError("NO-REFERENCE: /metrics has no strix_prompt_tokens_cached_total")


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def capture(args):
    text_a = Path(args.prompt).read_text()
    text_b = text_a + args.suffix_text  # a distinct continuation appended as text
    tok_a, tok_b = tokenize(args.port, text_a), tokenize(args.port, text_b)
    # BPE is context-sensitive, so appending text can re-merge the tokens straddling the join; the reuse is whatever
    # leading tokens A and B genuinely share. That shared run must hold at least one whole prefill chunk for reuse to fire at all.
    shared = common_prefix(tok_a, tok_b)
    require(shared >= CHUNK,
            f"A and B share only {shared} leading tokens; need >= {CHUNK} (A tokenizes to {len(tok_a)}); "
            "lengthen the prompt or pick a cleaner suffix")

    base0 = cached_total(args.port)
    ids_a = completion(args.port, text_a, args.generate)       # warms a slot with A's resident KV, then frees it
    base1 = cached_total(args.port)
    ids_b = completion(args.port, text_b, args.generate)       # shares `shared` leading tokens; reuses them when on
    reused_b = cached_total(args.port) - base1

    record = dict(prefix_len=shared, tokens_a=len(tok_a), tokens_b=len(tok_b), generate=args.generate,
                  ids_a=ids_a, ids_b=ids_b, reused_a=base1 - base0, reused_b=reused_b)
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(f"captured: |A|={len(tok_a)} |B|={len(tok_b)} shared_prefix={shared} reused_A={record['reused_a']} "
          f"reused_B={reused_b} gen={args.generate} -> {args.out}")
    return record


def compare_records(on, off):
    """Returns (verdict, reasons). EXACT only when tokens match across arms and reuse demonstrably fired on / not off."""
    reasons = []
    for key in ("ids_a", "ids_b", "prefix_len"):
        require(key in on and key in off, f"NO-REFERENCE: a capture lacks {key}")
    if on["ids_a"] != off["ids_a"]:
        reasons.append("A tokens differ across arms (prompt/model/sampling not matched)")
    if on["ids_b"] != off["ids_b"]:
        reasons.append("B tokens differ: prefix reuse changed the output (exactness violation)")
    whole = on["prefix_len"] // CHUNK * CHUNK
    if whole == 0 or on.get("reused_b", 0) < whole:
        reasons.append(f"ON arm reused {on.get('reused_b')} < {whole} (whole chunks of the shared {on['prefix_len']}): reuse did not fire, "
                       "so token equality proves nothing")
    if off.get("reused_b", 0) != 0:
        reasons.append(f"OFF arm reused {off.get('reused_b')} != 0: the OFF arm ran with reuse on")
    if reasons:
        bad_tokens = any("exactness" in r or "tokens differ" in r for r in reasons)
        return ("MISMATCH" if bad_tokens else "NO-REFERENCE"), reasons
    return "EXACT", [f"B reused {on['reused_b']} of {on['prefix_len']} prefix tokens, identical token ids"]


def compare(args):
    verdict, reasons = compare_records(json.loads(Path(args.on).read_text()), json.loads(Path(args.off).read_text()))
    print(f"{verdict}: " + "; ".join(reasons))
    return 0 if verdict == "EXACT" else 1


def self_test():
    base = dict(prefix_len=9000, ids_a=[5, 6, 7], ids_b=[8, 9, 10])
    on = dict(base, reused_a=0, reused_b=8192)
    off = dict(base, reused_a=0, reused_b=0)
    require(compare_records(on, off)[0] == "EXACT", "self-test: matched arms should be EXACT")
    require(compare_records(dict(on, ids_b=[8, 9, 99]), off)[0] == "MISMATCH", "self-test: changed B must be MISMATCH")
    require(compare_records(dict(on, reused_b=10), off)[0] == "NO-REFERENCE", "self-test: no-reuse ON must be NO-REFERENCE")
    require(compare_records(on, dict(off, reused_b=50))[0] == "NO-REFERENCE", "self-test: reuse-on-OFF must be NO-REFERENCE")
    print("PASS minimax prefix-cache gate self-test (comparison logic, CPU only)")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture", help="drive a running pair and record one arm")
    cap.add_argument("--port", type=int, required=True)
    cap.add_argument("--prompt", required=True, help="text file for prompt A; must tokenize to at least one prefill chunk (4096 tokens)")
    cap.add_argument("--out", required=True)
    cap.add_argument("--generate", type=int, default=32, help="tokens to generate per request (greedy)")
    cap.add_argument("--suffix-text", default="\n\nIn one word, the main theme above is:",
                     help="distinct text appended to A to form B (must append at a clean token boundary)")
    cmp = sub.add_parser("compare", help="compare an ON and an OFF capture")
    cmp.add_argument("--on", required=True)
    cmp.add_argument("--off", required=True)
    sub.add_parser("self-test", help="check the comparison logic without a server")
    args = parser.parse_args()
    if args.cmd == "capture":
        capture(args)
        return 0
    if args.cmd == "compare":
        return compare(args)
    return self_test()


if __name__ == "__main__":
    sys.exit(main())
