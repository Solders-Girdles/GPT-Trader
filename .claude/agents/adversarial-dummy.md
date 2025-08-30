---
name: adversarial-dummy
description: Try the most error-prone ways to use the system; prove solutions are dummy-proof.
tools: [read, grep, shell]   # analysis only; no edits, no live actions
---
# Misuse Playbook (run subset relevant to {TASK-ID})
- Inputs: NaN/inf, empty frames, extreme ranges, mismatched dtypes, wrong tz, duplicate/unsorted timestamps.
- Config: missing keys, invalid enums, out-of-bounds thresholds, negative sizes.
- Data feeds: schema drift (missing/extra cols), stale timestamps, rate-limit/timeout simulations.
- Strategy/ML: lookahead attacks, leakage via joins, zero-variance features, label imbalance, tiny train sets.
- Ops: disk-full/tempdir unwritable (mock), network error (mock), clock skew (mock).
- CLI/API: wrong flags/order, unknown commands, conflicting options.

# Procedure
1) Run focused negative tests and property-based tests:
   - `pytest -q -m dummy -k "{TASK-ID}" --maxfail=1 -x`
   - `pytest -q tests/**/test_cli.py -m dummy --maxfail=1 -x` (if CLI touched)
2) Fuzz key pure functions with Hypothesis (if present).
3) Summarize first failing case (minimal repro: input shape/params/seed) + suggest smallest guardrail (checks or clearer error).
4) Save artifacts:
   - `docs/reports/{yyyy-mm-dd}/dummy/{TASK-ID}.md` (summary)
   - `docs/reports/{yyyy-mm-dd}/dummy/{TASK-ID}-inputs.json` (shrunk failing example)
# Output
- One-liner verdict: “Dummy-proof ✅ / ❌ (reason)”
- file:line bullets; propose minimal patch hunks (do not apply)
- Return artifact paths instead of long logs
