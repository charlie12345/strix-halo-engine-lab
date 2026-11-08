#!/usr/bin/env python3
"""CPU tests of tests/sampled_coupled_gate.py on synthetic receipts: a complete campaign passes; a missing job,
receipt, seed or evidence line, a truncated vector, a wrong policy or acceptance, a token or ledger difference, a
forced arm that never forced and a partial result directory fail; the measurement report needs every run."""
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sampled_coupled_gate as gate  # noqa: E402

failures = 0


def expect(cond, msg):
    global failures
    if not cond:
        print("FAIL: " + msg)
        failures += 1


def h(*parts):
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def tokens_of(group, role, seed, gen, eos):
    out = [int(h(group, role, seed, i)[:6], 16) % 997 + 3 for i in range(gen)]
    if eos and seed % 2 == 0:
        out = out[: gen // 2] + [eos]
    return out


def sampling(job, seed, forced_count=3):
    p = job["policy"]
    s = {"identity": "sampling-v1", "temperature": p["temperature"], "top_k": p["top_k"], "top_p": p["top_p"],
         "min_p": p["min_p"], "seed": seed, "policy_identity": 7, "speculation": job["spec"],
         "schedule": "adaptive-timed" if job["adaptive"] else "fixed", "drafts": 0 if job["reference"] else 40,
         "plain_decisions": 10, "accepted_drafts": 0 if job["reference"] else 25, "rejected_drafts": 0 if job["reference"] else 15}
    if job["acceptance"]:
        s["acceptance"] = "coupled"
        s["argmax_draft_below"] = 0.7
    if job["forced"] is not None:
        s["forced_rejection"] = job["forced"]
        s["forced_rejections"] = forced_count
    return s


def write_campaign(manifest, d, eos_token=777):
    for job in manifest["jobs"]:
        eos = eos_token if job.get("eos") else None
        recs, err = [], []
        n = len(job["roles"]) * job["repeats"]
        for i in range(n):
            role = job["roles"][0] if job["route"] == "single" else job["roles"][i]
            seed = job["seed"] + i
            toks = tokens_of(job["group"], role, seed, job["gen"], eos)
            final = 100 + len(toks)
            r = {"sampling": sampling(job, seed), "generated_tokens": toks, "final_position": final,
                 "stop_reason": "eos" if eos and toks[-1] == eos else "token-budget", "emitted_tokens": len(toks),
                 "mtp_attempted_target_windows": 0 if job["reference"] else 20, "mtp_cycles": 30, "generation_ms": 1000.0,
                 "cycle_tokens": len(toks) - 1}
            if job["state"]:
                r["final_state_position"] = final
                r["final_state_digest"] = h("state", job["group"], role, seed, job["id"])  # includes the predictor
                # target components equal per (prompt, seed); predictor components differ per job (drafting)
                r["final_state_components"] = {
                    "attention_kv": h("kv", job["group"], role, seed), "gdn.recurrent": h("gdn", job["group"], role, seed),
                    "mtp.kv": h("mtp.kv", job["id"], seed), "mtp.held": h("mtp.held", job["id"], seed)}
            recs.append(r)
            if job["evidence"]:
                err.append("mtp-step-logits begin stream=0x%x begin=99 vocab=248320" % (16 + i % 2))
                err.append("mtp-step-logits committed stream=0x%x position=99 vocab=248320 sha256=%s" % (16 + i % 2, h("x")))
                err.append("mtp-step-logits summary stream=0x%x begin=99 end=%d count=%d vocab=248320 coverage=COMPLETE sha256=%s"
                           % (16 + i % 2, final, final - 99, h("ledger", job["group"], role, seed)))
                err.append("mtp-final-hidden position=%d predictor_end=%d lineage=5 sha256=%s" % (final, final - 1 - i, h("hid", job["group"], role, seed)))
        if job["route"] == "single":
            text = "log line {not json\n" + "\n".join(json.dumps(r, indent=1) for r in recs)
        else:
            for k, r in enumerate(recs):
                r.update(request=k, slot=k)
            text = json.dumps({"mode": "native-mtp-cohort-tiled-prompt-target-verified-sampled", "mtp_cohort_mode": job["cohort"],
                               "mtp_cohort_windows": 0 if job["cohort"] == "sequential" else 12,
                               "mtp_cohort_aggregate_tokens_per_second": 50.0, "requests": recs})
        with open(os.path.join(d, job["id"] + ".json"), "w") as f:
            f.write(text)
        if job["evidence"]:
            with open(os.path.join(d, job["id"] + ".err"), "w") as f:
                f.write("\n".join(err) + "\n")


def verdict(manifest, d):
    out = io.StringIO()
    ok = gate.compare(manifest, d, out)
    return ok, out.getvalue()


def mutate(manifest, base, fn):
    d = tempfile.mkdtemp()
    try:
        for name in os.listdir(base):
            shutil.copy(os.path.join(base, name), d)
        fn(d)
        return verdict(manifest, d)
    finally:
        shutil.rmtree(d)


def edit_json(d, job_id, fn, cohort=False):
    path = os.path.join(d, job_id + ".json")
    text = open(path).read()
    if cohort:
        obj = json.loads(text)
        fn(obj["requests"])
        open(path, "w").write(json.dumps(obj))
        return
    recs = gate.receipts(text)
    recs = [r for r in recs if "generated_tokens" in r]
    fn(recs)
    open(path, "w").write("\n".join(json.dumps(r) for r in recs))


def flip_last_hex(d, job_id, prefix):
    """Change the last digit of the first line starting with `prefix` (a digest) in the job's .err file."""
    path = os.path.join(d, job_id + ".err")
    lines = open(path).read().splitlines()
    for k, line in enumerate(lines):
        if line.startswith(prefix):
            lines[k] = line[:-1] + ("0" if line[-1] != "0" else "1")
            break
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    manifest = gate.plan("smoke")
    s = manifest["summary"]
    expect(s["exact_jobs"] > 0 and s["compared_requests"] > 0, "the smoke plan has jobs to compare")
    full = gate.plan("full")
    arms = {j["arm"] for j in full["jobs"]}
    for arm in ("plain", "R2", "R4", "R8", "F248", "AD248", "FR0-R8", "FR1-R8", "FR3-R8", "FRrotate-R8", "seq-plain",
                "lock-F248", "F248rej", "lock-rej", "lock-plain"):
        expect(arm in arms, "full plan has arm " + arm)
    plain = [j for j in full["jobs"] if j["arm"] == "plain"]
    expect(all("--mtp-sample-speculation" in j["args"] and not j["spec"] for j in plain),
           "plain references are sampled runs without speculation or draft flags")
    expect(all(j["env"].get("STRIX_MTP_SAMPLED_ACCEPT") != "coupled" for j in full["jobs"] if j["arm"] in ("F248rej", "lock-rej")),
           "rejection arms keep the default acceptance")
    grid = {"%s-T%s-%s" % (r, t, a) for r in ("chat", "code", "es") for t in ("0.7", "1.0") for a in ("plain", "R2", "R4", "R8", "AD248")}
    expect({j["id"] for j in full["jobs"]} >= grid and all(j["seed"] == 1000 and j["repeats"] == 10 and j["gen"] == 512
                                                           for j in full["jobs"] if j["id"] in grid),
           "main grid: chat/code/es x T0.7/1.0 x plain/R2/R4/R8/adaptive, 10 seeds from 1000, 512 tokens")
    expect({tuple(j["roles"]) for j in full["jobs"] if j["route"] == "single"} >= {("chat",), ("code",), ("es",)},
           "chat, code and es prompts")
    expect(sorted({j["policy"]["temperature"] for j in full["jobs"]}) == [0.7, 1.0], "temperatures 0.7 and 1.0")
    expect(any(j["policy"]["min_p"] for j in full["jobs"]) and any(j["policy"]["top_k"] for j in full["jobs"]) and
           any(j["policy"]["top_p"] < 1 for j in full["jobs"]), "top-k, top-p and min-p cells")
    expect(any(j.get("eos") for j in full["jobs"]) and {j["gen"] for j in full["jobs"]} >= {7, 61, 512}, "EOS and budget tails")
    order = [j["arm"] for j in full["jobs"] if j["id"].startswith("m-chat-T1.0-")]
    expect(order == ["plain", "F248", "F248rej", "F248rej", "F248", "plain"], "measurement runs in palindromic order")

    base = tempfile.mkdtemp()
    try:
        empty_ok, text = verdict(manifest, base)
        expect(not empty_ok and "INCOMPLETE" in text, "an empty result directory fails")
        write_campaign(manifest, base)
        ok, text = verdict(manifest, base)
        expect(ok and text.rstrip().splitlines()[-1].startswith("PASS"), "a complete synthetic campaign passes:\n" + text)
        lines = gate.job_lines(manifest, base, "exact")
        eos_lines = [l for l in lines if "eos" in l.split("\t")[0]]
        expect(eos_lines and all("--eos-token" in l for l in eos_lines), "EOS jobs resolve auto EOS from the plain run")
        for job in (j for j in manifest["jobs"] if j.get("eos")):
            token = gate.resolve_eos(manifest, base, job["eos"], job["repeats"])
            source = [r["generated_tokens"] for r in gate.receipts(open(os.path.join(base, job["eos"][5:] + ".json")).read())
                      if "generated_tokens" in r][: job["repeats"]]
            expect(token is not None and all(token not in toks[: gate.EOS_MIN_POSITION] for toks in source) and
                   any(token in toks for toks in source), "auto EOS first occurs at or after position %d" % gate.EOS_MIN_POSITION)
        lock = next(j for j in manifest["jobs"] if j["part"] == "exact" and j.get("cohort") == "lockstep")
        ok_pred, text = mutate(manifest, base, lambda d: edit_json(d, lock["id"],
            lambda r: r[0]["final_state_components"].__setitem__("mtp.held", "1" * 64), cohort=True))
        expect(ok_pred, "predictor (mtp.*) component differences do not fail the gate:\n" + text)
        expect(any(l.split("\t")[1] == "WAIT" for l in gate.job_lines(manifest, tempfile.mkdtemp(), "exact")),
               "EOS jobs wait for their source job")

        candidate = next(j for j in manifest["jobs"] if j["part"] == "exact" and not j["reference"] and j["route"] == "single" and j["forced"] is None)
        forced = next(j for j in manifest["jobs"] if j["part"] == "exact" and j["forced"] is not None and j["route"] == "single")
        reference = next(j for j in manifest["jobs"] if j["part"] == "exact" and j["reference"] and j["route"] == "single")
        lock = next(j for j in manifest["jobs"] if j["part"] == "exact" and j.get("cohort") == "lockstep")
        cases = {
            "missing job": lambda d: os.remove(os.path.join(d, candidate["id"] + ".json")),
            "missing reference": lambda d: os.remove(os.path.join(d, reference["id"] + ".json")),
            "missing receipt": lambda d: edit_json(d, candidate["id"], lambda r: r.pop()),
            "truncated vector": lambda d: edit_json(d, candidate["id"], lambda r: r[0]["generated_tokens"].pop()),
            "token difference": lambda d: edit_json(d, candidate["id"], lambda r: r[-1]["generated_tokens"].__setitem__(-1, 5)),
            "wrong seed": lambda d: edit_json(d, candidate["id"], lambda r: r[0]["sampling"].__setitem__("seed", 4242)),
            "wrong acceptance": lambda d: edit_json(d, candidate["id"], lambda r: r[0]["sampling"].pop("acceptance")),
            "wrong top-k": lambda d: edit_json(d, candidate["id"], lambda r: r[0]["sampling"].__setitem__("top_k", 3)),
            "request identity": lambda d: edit_json(d, candidate["id"], lambda r: r[0]["sampling"].__setitem__("request_identity", 2)),
            "no speculation": lambda d: edit_json(d, candidate["id"], lambda r: [x["sampling"].__setitem__("accepted_drafts", 0) for x in r]),
            "forcing never fired": lambda d: edit_json(d, forced["id"], lambda r: [x["sampling"].__setitem__("forced_rejections", 0) for x in r]),
            "missing ledger": lambda d: os.remove(os.path.join(d, candidate["id"] + ".err")),
            "ledger NO-REFERENCE": lambda d: open(os.path.join(d, candidate["id"] + ".err"), "w").write(
                open(os.path.join(d, candidate["id"] + ".err")).read().replace("COMPLETE", "NO-REFERENCE")),
            "ledger difference": lambda d: flip_last_hex(d, candidate["id"], "mtp-step-logits summary"),
            "final hidden difference": lambda d: flip_last_hex(d, candidate["id"], "mtp-final-hidden"),
            "cohort target state difference": lambda d: edit_json(d, lock["id"], lambda r: r[0]["final_state_components"].__setitem__("gdn.recurrent", "0" * 64), cohort=True),
            "cohort state components missing": lambda d: edit_json(d, lock["id"], lambda r: r[0].pop("final_state_components"), cohort=True),
            "cohort only predictor components": lambda d: edit_json(d, lock["id"], lambda r: r[0].__setitem__("final_state_components", {"mtp.kv": "0" * 64}), cohort=True),
            "cohort member missing": lambda d: edit_json(d, lock["id"], lambda r: r.pop(), cohort=True),
            "budget stop short": lambda d: edit_json(d, candidate["id"], lambda r: (r[1]["generated_tokens"].pop(), r[1].__setitem__("stop_reason", "context-capacity"))),
        }
        for name, fn in cases.items():
            ok, text = mutate(manifest, base, fn)
            expect(not ok and "PASS" not in text.splitlines()[-1], "%s must fail:\n%s" % (name, text))

        # The ledger replacement above must actually change a digest: check it directly.
        err = open(os.path.join(base, candidate["id"] + ".err")).read()
        expect("coverage=COMPLETE sha256=" in err, "synthetic ledger lines are parsed")
        ev = gate.evidence_lines(err)
        expect(len(ev) == candidate["repeats"] and all(e["coverage"] == "COMPLETE" and "final_hidden" in e for e in ev),
               "evidence lines map to requests in order")

        out = io.StringIO()
        expect(gate.measure(manifest, base, out) and "coupled/rejection" in out.getvalue(), "measurement report:\n" + out.getvalue())
        m_job = next(j for j in manifest["jobs"] if j["part"] == "measure")
        ok, text = False, ""
        d = tempfile.mkdtemp()
        for name in os.listdir(base):
            shutil.copy(os.path.join(base, name), d)
        os.remove(os.path.join(d, m_job["id"] + ".json"))
        out = io.StringIO()
        expect(not gate.measure(manifest, d, out) and "INCOMPLETE" in out.getvalue(), "a partial measurement is not reported")
        shutil.rmtree(d)
    finally:
        shutil.rmtree(base)
    if failures:
        print("%d sampled coupled gate check(s) failed" % failures)
        return 1
    print("sampled coupled gate tests passed (%d mutation cases)" % len(cases))
    return 0


if __name__ == "__main__":
    sys.exit(main())
