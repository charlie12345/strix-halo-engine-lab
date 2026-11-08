# Qwen all-on composition (`STRIX_QWEN_ALLON`)

One opt-in switch for composition testing: how far does the engine get with every implemented lever on.

| Value | Effect |
|---|---|
| unset, `0` | nothing changes (shipped defaults, bitwise) |
| `1`, `exact` | every exact lever of the table in [optimization-table.md](../../engine/optimization-table.md) (section "Opt-in levers") |
| `fast` | the exact levers plus the FAST levers (bit-changing, qualification in progress) |
| anything else | the process exits with status 2 at start |

Rules:

- The profile exports plain environment variables before `main`, only for variables that are unset. An explicit value
  (including `0`) wins, so a leave-one-out arm is `STRIX_QWEN_ALLON=1 STRIX_<LEVER>=0`.
- `STRIX_BASELINE=1` vetoes the profile.
- Every lever keeps its admission and its decline notice. Examples: the direct-B fragments run only on plain CLI prefill,
  the BF16 route storage and the one-byte route storage are FAST, the index projection candidate is excluded from MTP.
- A `qwen allon: mode=... N lever variables set` line is written to standard error at start, and
  `--print-effective-config` ends the prefill block with an `allon` object (mode, resolved value and class per lever).
- The profile does not set: `STRIX_QPF2K_MOE_TILE64` (an alternative organisation, declined while the tier split is
  active), `STRIX_QPF2K_MOE_TIERS=2` (instead of 1), `STRIX_QPF2K_SCHED` bit 64 (the YaRN route admits only schedule 12),
  `STRIX_QPF2K_QSA_UNION_CAP` (needs an owner-chosen value), the build-time `-DSTRIX_DPP_REDUCE=ON`, and the RTX
  options (`STRIX_QRTX_*`, the BF16 route target), which are command-line or test-fixture controls.
- The YaRN routes (262K and 1M) classify their environment. On Linux the route reads every inherited `STRIX_` variable
  and refuses any it does not classify, by name, before admission. `STRIX_QWEN_ALLON` and the exact levers are
  classified (each after a check that its files carry no rotary site), so `STRIX_QWEN_ALLON=1` is admitted on plain
  serial YaRN prefill. `STRIX_QWEN_ALLON=fast` is refused with the two levers YaRN does not qualify, the one-byte route
  storage and the attention budget; set them off to run the rest of the FAST profile:
  `STRIX_QWEN_ALLON=fast STRIX_PREFILL_MOE_ROUTE_I8=0 STRIX_QPF2K_QSA_BUDGET=0`.
  The index score and index projection levers are admitted on plain YaRN prefill and are switched off by the
  route on YaRN MTP (governed candidates stay excluded there). With `--mtp` the device handoff must be set to 0.
- Two levers have YaRN-specific behaviour: `STRIX_QPF2K_MOE_DIRECT_B=1` runs on plain YaRN prefill (its prepared planes are
  charged as a named owner of the YaRN memory estimate), and `STRIX_QPF2K_T16K=1` tries T16384 first on a plain YaRN
  prompt of at least 16,384 tokens, falling back through T8192, T4096 and T2048. Both need a node gate against the default
  before use.

The lever census, the sources of every commit and the test plan are in the engineering report that accompanies the branch.
