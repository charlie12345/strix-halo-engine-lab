#!/usr/bin/env python3
"""Coupled sampled acceptance gate (STRIX_MTP_SAMPLED_ACCEPT=coupled): manifest, job list, verdict, measurement.

The engine runs elsewhere (a GPU runner turns each manifest job into one strix-hip-decode-token process and
writes RESDIR/<job>.json = stdout receipts and RESDIR/<job>.err = stderr diagnostics, the .json only on exit 0).
This script needs no GPU:

  plan    [--profile full|smoke] [--seeds N] [--gen N] --out MANIFEST    write the expected campaign
  jobs    MANIFEST [--resdir DIR] [--part exact|measure]                 one TSV line per job for the runner
  compare MANIFEST RESDIR                                                exactness verdict (exit 0 only on PASS)
  measure MANIFEST RESDIR                                                acceptance, tokens per window, tok/s

Exactness: every request of a coupled arm (fixed R2/R4/R8, fixed and adaptive schedules over 2,4,8, forced
rejections, min-p, EOS and budget tails, lockstep cohort permutations) must equal its plain sampled reference
(--mtp-sample-speculation off, no draft chain on the single route; the sequential cohort on the tiled route, each
request as alone) with the same prompt, seed and request identity: the token IDs, the committed full-vocabulary
logits ledger (STRIX_MTP_STEP_LOGITS_DIGEST=1), the final target hidden row (STRIX_MTP_LOGITS_DIGEST=1) and, on the
tiled route, the final committed target state per component (STRIX_FINAL_STATE_DIGEST=1 with
STRIX_PREFIX_CACHE_DIGEST_COMPONENTS=1: every component of the target state registry, i.e. attention KV and sparse
index rows below the frontier and the fixed recurrent/convolution/PLE ranges; the predictor's mtp.kv, mtp.index and
mtp.held are excluded because drafting legitimately changes them, and are reported as info). The verdict is PASS only
when every job, receipt, request and evidence line the manifest expects is present and complete; a partial campaign,
a missing seed, a truncated token vector or a NO-REFERENCE ledger is a FAIL.

Measurement: coupled vs rejection (the default) vs plain sampling, without diagnostics, in palindromic order
(P A B B A P per cell: ABBA with the plain arm outside) at C1 (single route) and C4 (lockstep tiled cohort).
"""
import argparse
import collections
import json
import os
import re
import shlex
import sys

SCHEMA = "sampled-coupled-gate-v1"
ROLES = ("chat", "code", "es")
TEMPERATURES = {"0.7": {"top_k": 20, "top_p": 0.8, "min_p": 0.0}, "1.0": {"top_k": 0, "top_p": 1.0, "min_p": 0.0}}
COUPLED = {"STRIX_MTP_SAMPLED_ACCEPT": "coupled"}
EVIDENCE_ENV = {"STRIX_MTP_STEP_LOGITS_DIGEST": "1", "STRIX_MTP_LOGITS_DIGEST": "1"}
STATE_ENV = {"STRIX_FINAL_STATE_DIGEST": "1", "STRIX_PREFIX_CACHE_DIGEST_COMPONENTS": "1"}
# Predictor (MTP) components of the tiled state registry: a drafted run legitimately differs from a plain run there
# (draft scratch, predictor rows written while drafting), so the gate compares every other (target) component.
PREDICTOR_COMPONENTS = ("mtp.",)
# auto EOS: no request may reach the stop token before this position, so every arm drafts before stopping.
EOS_MIN_POSITION = 24
# Single-route arms: (name, owner rows, extra args, env, forced row or None). Owner = the largest row-set entry.
SINGLE_ARMS = {
    "R2": (2, ["--mtp-target-window-rows-set", "2"], COUPLED, None),
    "R4": (4, ["--mtp-target-window-rows-set", "4"], COUPLED, None),
    "R8": (8, ["--mtp-target-window-rows-set", "8"], COUPLED, None),
    "F248": (8, ["--mtp-target-window-rows-set", "2,4,8"], COUPLED, None),
    "AD248": (8, ["--mtp-target-window-rows-set", "2,4,8", "--mtp-sample-schedule", "adaptive"], COUPLED, None),
}
for _row in ("0", "1", "3", "rotate"):
    SINGLE_ARMS["FR%s-R8" % _row] = (8, ["--mtp-target-window-rows-set", "8"],
                                     dict(COUPLED, STRIX_MTP_SAMPLED_FORCE_REJECT=_row), _row)
REJECTION_ARMS = {"F248rej": (8, ["--mtp-target-window-rows-set", "2,4,8"], {}, None)}
PLAIN_ARGS = ["--mtp-sample-speculation", "off"]
# Rough seconds per job for the turn estimate: process start, model load and replay, then per generated token.
EST_START_S, EST_PLAIN_S, EST_DRAFT_S, EST_COHORT_S, EST_DIAG = 45.0, 0.021, 0.0135, 0.009, 1.25


class GateError(Exception):
    pass


def policy_of(t, filters):
    return {"temperature": float(t), "top_k": filters["top_k"], "top_p": filters["top_p"], "min_p": filters["min_p"]}


def single_job(jid, part, role, t, filters, arm, seed, repeats, gen, reference_group, eos=None, evidence=True):
    if arm == "plain":
        rows, args, env, forced, spec = 8, list(PLAIN_ARGS), {}, None, False
    else:
        rows, args, env, forced = (SINGLE_ARMS.get(arm) or REJECTION_ARMS[arm])
        args, env, spec = list(args), dict(env), True
    if evidence:
        env.update(EVIDENCE_ENV)
    tokens = repeats * gen
    est = EST_START_S + tokens * (EST_PLAIN_S if arm == "plain" else EST_DRAFT_S) * (EST_DIAG if evidence else 1.0)
    return {
        "id": jid, "part": part, "route": "single", "roles": [role], "positional": role, "arm": arm,
        "reference": arm == "plain", "group": reference_group, "owner_rows": rows, "spec": spec,
        "env": env, "args": args, "policy": policy_of(t, filters), "seed": seed, "repeats": repeats, "gen": gen,
        "eos": eos, "acceptance": "coupled" if env.get("STRIX_MTP_SAMPLED_ACCEPT") == "coupled" else None,
        "forced": forced, "adaptive": "--mtp-sample-schedule" in args, "evidence": evidence,
        "state": False, "est_s": round(est, 1),
    }


def cohort_job(jid, part, members, t, filters, arm, seed, gen, reference_group, evidence=True, forced=None):
    lockstep = arm != "seq-plain"
    args = ["--mtp-target-window-rows-set", "2,4,8", "--mtp-cohort-mode", "lockstep" if lockstep else "sequential"]
    env = {}
    if arm == "seq-plain" or arm == "lock-plain":
        args += PLAIN_ARGS
    elif arm != "lock-rej":
        env.update(COUPLED)
    if forced is not None:
        env["STRIX_MTP_SAMPLED_FORCE_REJECT"] = forced
    if evidence:
        env.update(EVIDENCE_ENV)
        env.update(STATE_ENV)
    tokens = len(members) * gen
    per = EST_COHORT_S if lockstep and "plain" not in arm else EST_PLAIN_S
    est = EST_START_S + 15.0 * len(members) + tokens * per * (EST_DIAG if evidence else 1.0)
    return {
        "id": jid, "part": part, "route": "tiled", "roles": list(members), "positional": members[0], "arm": arm,
        "reference": arm == "seq-plain", "group": reference_group, "owner_rows": 8, "spec": True,
        "env": env, "args": args, "policy": policy_of(t, filters), "seed": seed, "repeats": 1, "gen": gen,
        "eos": None, "acceptance": "coupled" if env.get("STRIX_MTP_SAMPLED_ACCEPT") == "coupled" else None,
        "forced": forced, "adaptive": False, "evidence": evidence, "state": evidence, "cohort": "lockstep" if lockstep else "sequential",
        "est_s": round(est, 1),
    }


def plan(profile="full", seeds=None, gen=None):
    smoke = profile == "smoke"
    nseeds = seeds or (2 if smoke else 10)
    few = max(1, min(nseeds, 2 if smoke else 5))
    gen = gen or (64 if smoke else 512)
    base_seed = 1000
    roles = ROLES[:1] if smoke else ROLES
    temps = ["1.0"] if smoke else ["0.7", "1.0"]
    jobs = []

    def group(route, t, filt, g, eos=None):
        f = "k%d-p%g-m%g" % (filt["top_k"], filt["top_p"], filt["min_p"])
        return "%s/T%s/%s/gen%d%s" % (route, t, f, g, "/eos" if eos else "")

    # Main grid: plain reference and the coupled fixed, schedule and forced-rejection arms.
    main_arms = ["R2", "R4", "R8", "F248", "AD248"]
    forced_arms = ["FR0-R8", "FR1-R8", "FR3-R8", "FRrotate-R8"]
    for role in roles:
        for t in temps:
            f = TEMPERATURES[t]
            g = group("single", t, f, gen)
            jobs.append(single_job("%s-T%s-plain" % (role, t), "exact", role, t, f, "plain", base_seed, nseeds, gen, g))
            for arm in main_arms:
                jobs.append(single_job("%s-T%s-%s" % (role, t, arm), "exact", role, t, f, arm, base_seed, nseeds, gen, g))
            for arm in forced_arms:
                jobs.append(single_job("%s-T%s-%s" % (role, t, arm), "exact", role, t, f, arm, base_seed, few, gen, g))
    # min-p (filters beyond the T0.7 top-k/top-p cells).
    fm = {"top_k": 0, "top_p": 1.0, "min_p": 0.05}
    g = group("single", "1.0", fm, gen)
    for arm in ("plain", "F248", "FRrotate-R8"):
        jobs.append(single_job("chat-T1.0minp-%s" % arm, "exact", "chat", "1.0", fm, arm, base_seed, few, gen, g))
    # EOS: a stop token frequent in the plain reference output ("auto:<job>"), so requests stop inside windows.
    for role in (("chat",) if smoke else ("chat", "es")):
        eos = "auto:%s-T1.0-plain" % role
        g = group("single", "1.0", TEMPERATURES["1.0"], gen, eos)
        for arm in ("plain", "F248", "FRrotate-R8"):
            jobs.append(single_job("%s-T1.0eos-%s" % (role, arm), "exact", role, "1.0", TEMPERATURES["1.0"], arm,
                                   base_seed, few, gen, g, eos=eos))
    # Budget tails: budgets that end inside windows of every size.
    tail_role = "code" if not smoke else "chat"
    for tail in ((7,) if smoke else (7, 61)):
        g = group("single", "1.0", TEMPERATURES["1.0"], tail)
        for arm in ("plain", "R8", "F248"):
            jobs.append(single_job("%s-T1.0gen%d-%s" % (tail_role, tail, arm), "exact", tail_role, "1.0",
                                   TEMPERATURES["1.0"], arm, base_seed, nseeds, tail, g))
    # Cohort permutations on the tiled route (request i: seed + i). Each permutation's sequential plain run is the
    # C1 reference of exactly its (role, seed) pairs; references shared by permutations must agree.
    members = ["chat", "code", "es", "chat"] if not smoke else ["chat", "chat"]
    perms = [[0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]] if not smoke else [[0, 1]]
    for t in temps:
        f = TEMPERATURES[t]
        g = group("tiled", t, f, gen)
        for p, order in enumerate(perms):
            ms = [members[i] for i in order]
            if t == "0.7" and p == 2:
                continue  # two permutations at T0.7, three at T1.0
            jobs.append(cohort_job("coh%d-T%s-seq-plain" % (p, t), "exact", ms, t, f, "seq-plain", base_seed, gen, g))
            jobs.append(cohort_job("coh%d-T%s-lock-F248" % (p, t), "exact", ms, t, f, "lock-F248", base_seed, gen, g))
            if p == 0:
                jobs.append(cohort_job("coh%d-T%s-lock-FRrotate" % (p, t), "exact", ms, t, f, "lock-F248", base_seed, gen, g,
                                       forced="rotate"))
                # lockstep without drafts: separates the lockstep cohort itself from coupled acceptance
                jobs.append(cohort_job("coh%d-T%s-lock-plain" % (p, t), "exact", ms, t, f, "lock-plain", base_seed, gen, g))
    # Measurement (no diagnostics): palindromic order per cell, C1 single route and C4 lockstep cohort.
    mseeds = few
    for role in roles:
        for t in temps:
            f = TEMPERATURES[t]
            order = [("plain", "a"), ("F248", "a"), ("F248rej", "a"), ("F248rej", "b"), ("F248", "b"), ("plain", "b")]
            for arm, rep in order:
                j = single_job("m-%s-T%s-%s-%s" % (role, t, arm, rep), "measure", role, t, f, arm, 2000, mseeds, gen,
                               "measure", evidence=False)
                j["load"] = "C1"
                jobs.append(j)
    for t in temps:
        f = TEMPERATURES[t]
        ms = [members[i] for i in perms[0]]
        for arm, rep in [("lock-plain", "a"), ("lock-F248", "a"), ("lock-rej", "a"), ("lock-rej", "b"), ("lock-F248", "b"),
                         ("lock-plain", "b")]:
            j = cohort_job("m-coh-T%s-%s-%s" % (t, arm, rep), "measure", ms, t, f, arm, 2000, gen, "measure", evidence=False)
            j["load"] = "C%d" % len(ms)
            jobs.append(j)
    ids = [j["id"] for j in jobs]
    if len(ids) != len(set(ids)):
        raise GateError("duplicate job ids")
    exact = [j for j in jobs if j["part"] == "exact"]
    measure = [j for j in jobs if j["part"] == "measure"]
    return {
        "schema": SCHEMA, "profile": profile, "roles": list(roles), "temperatures": temps, "seeds": nseeds,
        "gen": gen, "jobs": jobs,
        "summary": {
            "jobs": len(jobs), "exact_jobs": len(exact), "measure_jobs": len(measure),
            "exact_requests": sum(len(j["roles"]) * j["repeats"] for j in exact),
            "compared_requests": sum(len(j["roles"]) * j["repeats"] for j in exact if not j["reference"]),
            "exact_est_s": round(sum(j["est_s"] for j in exact)), "measure_est_s": round(sum(j["est_s"] for j in measure)),
        },
    }


# ---------------------------------------------------------------- results

def receipts(text):
    dec = json.JSONDecoder()
    out, i = [], 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return out
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        i = end


LEDGER_BEGIN = re.compile(r"^mtp-step-logits begin stream=(\S+) begin=(\d+)")
LEDGER_SUMMARY = re.compile(r"^mtp-step-logits summary stream=(\S+) begin=(\d+) end=(\d+) count=(\d+) vocab=(\d+) "
                            r"coverage=(\S+) sha256=([0-9a-f]{64})")
FINAL_HIDDEN = re.compile(r"^mtp-final-hidden position=(\d+) .*sha256=([0-9a-f]{64})")


def evidence_lines(text):
    """Per request, in request order (begin lines), the ledger summary and the final hidden digest."""
    requests, current = [], {}
    last = None
    for line in text.splitlines():
        m = LEDGER_BEGIN.match(line)
        if m:
            current[m.group(1)] = len(requests)
            requests.append({"begin": int(m.group(2))})
            continue
        m = LEDGER_SUMMARY.match(line)
        if m and m.group(1) in current:
            r = requests[current.pop(m.group(1))]
            r.update(ledger_begin=int(m.group(2)), ledger_end=int(m.group(3)), ledger_count=int(m.group(4)),
                     coverage=m.group(6), ledger=m.group(7))
            last = r
            continue
        m = FINAL_HIDDEN.match(line)
        if m and last is not None and "final_hidden" not in last:
            last["final_hidden"] = (int(m.group(1)), m.group(2))
    return requests


def resolve_eos(manifest, resdir, spec, requests=None):
    """auto:<job>: a stop token chosen from that plain job's first `requests` requests (the EOS job's own seeds; the
    same seeds give the same tokens until the stop): absent from positions [0, EOS_MIN_POSITION) of every request and
    present later in as many requests as possible (then the earliest mean first occurrence, then the lowest id). So
    every request decodes EOS_MIN_POSITION tokens or more (drafted arms draft and accept) and at least one stops on it.
    None while the source has not run or has no such token."""
    if not spec:
        return None
    if not spec.startswith("auto:"):
        return int(spec)
    path = os.path.join(resdir or "", spec[5:] + ".json")
    if not os.path.exists(path):
        return None
    runs = [r.get("generated_tokens") or [] for r in receipts(open(path, encoding="utf-8", errors="replace").read())
            if "generated_tokens" in r][:requests]
    early = set(t for toks in runs for t in toks[:EOS_MIN_POSITION])
    firsts = collections.defaultdict(list)
    for toks in runs:
        seen = set()
        for n, t in enumerate(toks):
            if n >= EOS_MIN_POSITION and t not in early and t not in seen:
                seen.add(t)
                firsts[t].append(n)
    if not firsts:
        return None
    return min(firsts.items(), key=lambda kv: (-len(kv[1]), sum(kv[1]) / len(kv[1]), kv[0]))[0]


def request_records(job, resdir, failures):
    """The job's request records in request order, or None (failures say why)."""
    path = os.path.join(resdir, job["id"] + ".json")
    if not os.path.exists(path):
        failures.append("MISSING job %s (no %s.json)" % (job["id"], job["id"]))
        return None
    found = receipts(open(path, encoding="utf-8", errors="replace").read())
    if job["route"] == "single":
        records = [r for r in found if "generated_tokens" in r and "sampling" in r]
        expected = job["repeats"]
    else:
        cohorts = [r for r in found if isinstance(r.get("requests"), list)]
        if len(cohorts) != 1:
            failures.append("job %s: expected one cohort receipt, found %d" % (job["id"], len(cohorts)))
            return None
        if cohorts[0].get("mtp_cohort_mode") != job["cohort"]:
            failures.append("job %s: cohort mode %r, expected %s" % (job["id"], cohorts[0].get("mtp_cohort_mode"), job["cohort"]))
        records = cohorts[0]["requests"]
        expected = len(job["roles"])
    if len(records) != expected:
        failures.append("job %s: %d request receipts, expected %d" % (job["id"], len(records), expected))
        return None
    lines = None
    if job["evidence"]:
        err = os.path.join(resdir, job["id"] + ".err")
        lines = evidence_lines(open(err, encoding="utf-8", errors="replace").read()) if os.path.exists(err) else []
        if len(lines) != expected:
            failures.append("job %s: %d committed-logits ledgers, expected %d" % (job["id"], len(lines), expected))
            return None
    out = []
    for i, r in enumerate(records):
        roles = job["roles"]
        role = roles[0] if job["route"] == "single" else roles[i]
        seed = job["seed"] + i
        rec = {"job": job["id"], "index": i, "role": role, "seed": seed, "receipt": r}
        rec["evidence"] = lines[i] if lines is not None else None
        out.append(rec)
    return out


def check_request(job, rec, failures):
    """Receipt fields against the manifest: policy, seed, identity, acceptance, forced switch, complete vector."""
    r, where = rec["receipt"], "job %s request %d" % (job["id"], rec["index"])
    s = r.get("sampling") or {}
    pol = job["policy"]
    ok = True

    def bad(msg):
        failures.append("%s: %s" % (where, msg))
        return False

    if s.get("identity") != "sampling-v1":
        ok = bad("sampling identity %r" % s.get("identity"))
    if s.get("seed") != rec["seed"]:
        ok = bad("seed %r, expected %d" % (s.get("seed"), rec["seed"]))
    for key in ("temperature", "top_p", "min_p"):
        if abs(float(s.get(key, -1.0)) - float(pol[key])) > 1e-6:
            ok = bad("%s %r, expected %r" % (key, s.get(key), pol[key]))
    if int(s.get("top_k", -1)) != int(pol["top_k"]):
        ok = bad("top_k %r, expected %r" % (s.get("top_k"), pol["top_k"]))
    identity = s.get("request_identity", 1)
    rec["identity"] = identity
    if identity != 1:
        ok = bad("request identity %r (the gate runs the default identity 1)" % identity)
    if s.get("acceptance") != job["acceptance"]:
        ok = bad("acceptance %r, expected %r" % (s.get("acceptance"), job["acceptance"]))
    if s.get("forced_rejection") != job["forced"]:
        ok = bad("forced rejection %r, expected %r" % (s.get("forced_rejection"), job["forced"]))
    if job["route"] == "single":
        if bool(s.get("speculation")) != job["spec"]:
            ok = bad("speculation %r, expected %r" % (s.get("speculation"), job["spec"]))
        if (s.get("schedule") == "adaptive-timed") != job["adaptive"]:
            ok = bad("schedule %r" % s.get("schedule"))
    tokens = r.get("generated_tokens")
    stop = r.get("stop_reason")
    if not isinstance(tokens, list) or not tokens or not all(isinstance(x, int) for x in tokens):
        return bad("no generated token vector")
    if len(tokens) > job["gen"]:
        ok = bad("%d tokens exceed the budget %d" % (len(tokens), job["gen"]))
    if stop == "eos":
        rec["eos_stop"] = True
    elif stop in ("token-budget", "length"):
        if len(tokens) != job["gen"]:
            ok = bad("truncated token vector: %d of %d tokens with stop %s" % (len(tokens), job["gen"], stop))
    else:
        ok = bad("stop reason %r (expected eos or token-budget)" % stop)
    ev = rec["evidence"]
    if job["evidence"]:
        if not ev or ev.get("coverage") != "COMPLETE" or not ev.get("ledger"):
            ok = bad("committed-logits ledger missing or not COMPLETE (%r)" % (ev or {}).get("coverage"))
        elif "final_position" in r and ev.get("ledger_end") != r["final_position"]:
            ok = bad("ledger ends at %r, receipt final position %r" % (ev.get("ledger_end"), r["final_position"]))
        if not ev or "final_hidden" not in ev:
            ok = bad("final hidden digest missing (STRIX_MTP_LOGITS_DIGEST=1)")
        if job["state"] and not target_components(r):
            ok = bad("final committed target state components missing (STRIX_FINAL_STATE_DIGEST=1 and "
                     "STRIX_PREFIX_CACHE_DIGEST_COMPONENTS=1)")
    return ok


def target_components(receipt):
    """The final committed state per target component: every registered component except the predictor's."""
    comps = receipt.get("final_state_components")
    if not isinstance(comps, dict):
        return {}
    return {k: v for k, v in comps.items() if not k.startswith(PREDICTOR_COMPONENTS)}


def evidence_key(rec, job):
    ev = rec["evidence"] or {}
    r = rec["receipt"]
    out = {"tokens": r.get("generated_tokens")}
    if job["evidence"]:
        out["ledger"] = (ev.get("ledger_begin"), ev.get("ledger_end"), ev.get("ledger"))
        out["final_hidden"] = ev.get("final_hidden")
    if job["state"]:
        out["state"] = (r.get("final_state_position"), target_components(r))
    return out


def first_difference(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def compare(manifest, resdir, out=sys.stdout):
    if manifest.get("schema") != SCHEMA:
        raise GateError("not a %s manifest" % SCHEMA)
    jobs = [j for j in manifest["jobs"] if j["part"] == "exact"]
    failures, info = [], []
    refs, candidates = {}, []
    expected_requests = sum(len(j["roles"]) * j["repeats"] for j in jobs)
    checked = 0
    for job in jobs:
        recs = request_records(job, resdir, failures)
        if recs is None:
            continue
        drafts = accepted = forced = eos_stops = 0
        for rec in recs:
            if not check_request(job, rec, failures):
                continue
            checked += 1
            s = rec["receipt"].get("sampling") or {}
            drafts += int(s.get("drafts", 0))
            accepted += int(s.get("accepted_drafts", 0))
            forced += int(s.get("forced_rejections", 0))
            eos_stops += 1 if rec.get("eos_stop") else 0
            key = (job["group"], rec["role"], rec["seed"], rec["identity"])
            if job["reference"]:
                ev = evidence_key(rec, job)
                if key in refs and refs[key][1] != ev:
                    failures.append("references disagree for %s: %s vs %s" % (key, refs[key][0], job["id"]))
                refs.setdefault(key, (job["id"], ev, rec["receipt"].get("final_state_components") or {}))
            else:
                candidates.append((key, job, rec))
        # Coverage: speculation ran, forcing fired, EOS reached, cohort windows were shared.
        if job["spec"] and not job["reference"] and job["arm"] not in ("lock-plain",):
            if not drafts or not accepted:
                failures.append("job %s: no drafts or no accepted drafts (%d/%d): speculation did not run" % (job["id"], accepted, drafts))
        if job["forced"] is not None and not forced:
            failures.append("job %s: no forced rejection fired" % job["id"])
        if job["eos"] and job["reference"] and not eos_stops:
            failures.append("job %s: no request stopped on the EOS token (coverage)" % job["id"])
        if job["route"] == "tiled" and job.get("cohort") == "lockstep" and job["arm"] not in ("lock-plain",):
            path = os.path.join(resdir, job["id"] + ".json")
            coh = [r for r in receipts(open(path, encoding="utf-8", errors="replace").read()) if "mtp_cohort_windows" in r]
            if not coh or not coh[0].get("mtp_cohort_windows"):
                failures.append("job %s: no shared cohort window ran" % job["id"])
        info.append("%s: %d requests, drafts %d accepted %d forced %d eos-stops %d" % (job["id"], len(recs), drafts, accepted, forced, eos_stops))
    compared = 0
    for key, job, rec in candidates:
        ref = refs.get(key)
        if not ref:
            failures.append("job %s request %d: no reference for %s" % (job["id"], rec["index"], key))
            continue
        mine = evidence_key(rec, job)
        theirs = ref[1]
        compared += 1
        for name in ("tokens", "ledger", "final_hidden", "state"):
            if name in mine and name not in theirs:
                failures.append("job %s request %d: reference %s lacks %s evidence" % (job["id"], rec["index"], ref[0], name))
            elif name in mine and mine[name] != theirs[name]:
                if name == "tokens":
                    a, b = mine[name], theirs[name]
                    failures.append("FAIL %s request %d (%s seed %d) tokens differ from %s at %d (len %d vs %d)" % (
                        job["id"], rec["index"], rec["role"], rec["seed"], ref[0], first_difference(a, b), len(a), len(b)))
                elif name == "state" and mine[name][0] == theirs[name][0]:
                    a, b = mine[name][1], theirs[name][1]
                    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
                    failures.append("FAIL %s request %d (%s seed %d) target state differs from %s in %d of %d components: %s" % (
                        job["id"], rec["index"], rec["role"], rec["seed"], ref[0], len(diff), len(set(a) | set(b)),
                        " ".join(diff[:12])))
                else:
                    failures.append("FAIL %s request %d (%s seed %d) %s differs from %s" % (
                        job["id"], rec["index"], rec["role"], rec["seed"], name, ref[0]))
    predictor_differs = 0
    for key, job, rec in candidates:
        if job["state"] and key in refs:
            mine = rec["receipt"].get("final_state_components") or {}
            theirs = refs[key][2]
            predictor_differs += any(mine.get(k) != theirs.get(k) for k in set(mine) | set(theirs) if k.startswith(PREDICTOR_COMPONENTS))
    if predictor_differs:
        info.append("predictor (mtp.*) state components differ from the reference in %d requests (not compared)" % predictor_differs)
    expected_compared = manifest["summary"]["compared_requests"]
    for line in info:
        print("info " + line, file=out)
    for line in failures:
        print(line, file=out)
    complete = checked == expected_requests and compared == expected_compared
    if not complete:
        print("INCOMPLETE: %d of %d requests checked, %d of %d compared" % (checked, expected_requests, compared, expected_compared), file=out)
    ok = not failures and complete and compared > 0
    print("%s: %d requests compared with their plain sampled references (%d jobs, %d references)" % (
        "PASS" if ok else "FAIL", compared, len(jobs), len(refs)), file=out)
    return ok


def measure(manifest, resdir, out=sys.stdout):
    jobs = [j for j in manifest["jobs"] if j["part"] == "measure"]
    failures = []
    rows = collections.defaultdict(list)  # (load, cell, arm) -> per-run values
    for job in jobs:
        recs = request_records(job, resdir, failures)
        if recs is None:
            continue
        for rec in recs:
            check_request(job, rec, failures)
        if job["route"] == "single":
            acc = rej = toks = windows = cycles = 0
            ms = 0.0
            for rec in recs:
                r, s = rec["receipt"], rec["receipt"].get("sampling") or {}
                acc += int(s.get("accepted_drafts", 0))
                rej += int(s.get("rejected_drafts", 0))
                toks += int(r.get("emitted_tokens", len(r.get("generated_tokens") or [])))
                windows += int(r.get("mtp_attempted_target_windows", 0))
                cycles += int(r.get("mtp_cycles", 0))
                ms += float(r.get("generation_ms", 0.0))
            value = {"acceptance": acc / (acc + rej) if acc + rej else None,
                     "tokens_per_window": toks / windows if windows else None,
                     "tokens_per_cycle": toks / cycles if cycles else None,
                     "tok_s": 1000.0 * toks / ms if ms else None}
            cell = "%s T%s" % (job["roles"][0], job["policy"]["temperature"])
        else:
            path = os.path.join(resdir, job["id"] + ".json")
            coh = [r for r in receipts(open(path, encoding="utf-8", errors="replace").read()) if isinstance(r.get("requests"), list)]
            samplings = [rec["receipt"].get("sampling") or {} for rec in recs]
            acc = sum(int(s.get("accepted_drafts", 0)) for s in samplings)
            rej = sum(int(s.get("rejected_drafts", 0)) for s in samplings)
            cycles = sum(int(rec["receipt"].get("mtp_cycles", 0)) for rec in recs)
            toks = sum(int(rec["receipt"].get("cycle_tokens", 0)) for rec in recs)
            value = {"acceptance": acc / (acc + rej) if acc + rej else None, "tokens_per_window": None,
                     "tokens_per_cycle": toks / cycles if cycles else None,
                     "tok_s": float(coh[0].get("mtp_cohort_aggregate_tokens_per_second", 0.0)) if coh else None}
            cell = "cohort T%s" % job["policy"]["temperature"]
        arm = {"plain": "plain", "lock-plain": "plain", "F248": "coupled", "lock-F248": "coupled",
               "F248rej": "rejection", "lock-rej": "rejection"}[job["arm"]]
        rows[(job["load"], cell, arm)].append(value)
    if failures:
        for line in failures:
            print(line, file=out)
        print("INCOMPLETE: measurement not reported", file=out)
        return False

    def mean(vals, key):
        xs = [v[key] for v in vals if v[key] is not None]
        return sum(xs) / len(xs) if xs else None

    def fmt(x, spec="%.3f"):
        return "-" if x is None else spec % x
    print("load cell arm runs acceptance tokens/window tokens/cycle tok/s (each run: tok/s)", file=out)
    cells = sorted({(k[0], k[1]) for k in rows})
    for load, cell in cells:
        means = {}
        for arm in ("plain", "coupled", "rejection"):
            vals = rows.get((load, cell, arm), [])
            means[arm] = mean(vals, "tok_s")
            print("%s %s %s %d %s %s %s %s (%s)" % (load, cell, arm, len(vals), fmt(mean(vals, "acceptance")),
                  fmt(mean(vals, "tokens_per_window")), fmt(mean(vals, "tokens_per_cycle")), fmt(means[arm], "%.2f"),
                  " ".join(fmt(v["tok_s"], "%.2f") for v in vals)), file=out)
        if means["coupled"] and means["rejection"] and means["plain"]:
            print("%s %s coupled/rejection %.3f coupled/plain %.3f rejection/plain %.3f" % (
                load, cell, means["coupled"] / means["rejection"], means["coupled"] / means["plain"],
                means["rejection"] / means["plain"]), file=out)
    return True


def job_lines(manifest, resdir=None, part=None):
    out = []
    for j in manifest["jobs"]:
        if part and j["part"] != part:
            continue
        eos = resolve_eos(manifest, resdir, j.get("eos"), j["repeats"])
        if j.get("eos") and eos is None:
            out.append("%s\tWAIT\t%s" % (j["id"], j["eos"]))  # the EOS source job has not run yet
            continue
        p = j["policy"]
        args = list(j["args"]) + ["--generate", str(j["gen"]), "--mtp-sample-temperature", "%g" % p["temperature"],
                                  "--mtp-sample-seed", str(j["seed"]), "--mtp-sample-repeats", str(j["repeats"])]
        if p["top_k"]:
            args += ["--mtp-sample-top-k", str(p["top_k"])]
        if p["top_p"] != 1.0:
            args += ["--mtp-sample-top-p", "%g" % p["top_p"]]
        if p["min_p"]:
            args += ["--mtp-sample-min-p", "%g" % p["min_p"]]
        if eos is not None:
            args += ["--eos-token", str(eos)]
        env =" ".join("%s=%s" % kv for kv in sorted(j["env"].items())) or "-"
        out.append("\t".join([j["id"], "%d" % j["est_s"], j["route"], ",".join(j["roles"]), str(j["owner_rows"]),
                              "1" if j["spec"] else "0", env, " ".join(shlex.quote(a) for a in args)]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--profile", choices=("full", "smoke"), default="full")
    p.add_argument("--seeds", type=int)
    p.add_argument("--gen", type=int)
    p.add_argument("--out", required=True)
    p = sub.add_parser("jobs")
    p.add_argument("manifest")
    p.add_argument("--resdir")
    p.add_argument("--part", choices=("exact", "measure"))
    for name in ("compare", "measure"):
        p = sub.add_parser(name)
        p.add_argument("manifest")
        p.add_argument("resdir")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "plan":
            m = plan(a.profile, a.seeds, a.gen)
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(m, f, indent=1, sort_keys=True)
                f.write("\n")
            print(json.dumps(m["summary"], sort_keys=True))
            return 0
        m = json.load(open(a.manifest, encoding="utf-8"))
        if a.cmd == "jobs":
            print("\n".join(job_lines(m, a.resdir, a.part)))
            return 0
        if not os.path.isdir(a.resdir):
            print("FAIL: no result directory %s" % a.resdir)
            return 1
        return 0 if (compare if a.cmd == "compare" else measure)(m, a.resdir) else 1
    except GateError as e:
        print("FAIL: %s" % e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
