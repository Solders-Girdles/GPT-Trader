# Test Hygiene Policy (Line Limits, Allowlist, and When to Split)

---
status: current
last-updated: 2026-01-31
---

GPT-Trader enforces lightweight test hygiene via `scripts/ci/check_test_hygiene.py`.

The goal is to keep tests navigable and keep refactors from turning into 600-line “mega tests”.

## Current enforcement

- Default max lines per test module: **240** (`THRESHOLD = 240`)
- Enforcement runs in pre-commit (and CI) via:
  - `scripts/ci/check_test_hygiene.py`

## Policy: default is “split”, allowlist is temporary

### Default
If a test module exceeds the threshold:
- **Split it** into smaller modules (preferred)

### Allowlist (temporary)
If splitting is not reasonable in the current PR:
- add the file to the allowlist **with a justification**
- include a note that it’s “split pending”

Allowlist entries are acceptable when:
- the file is a short-lived consolidation step during a refactor
- the file shares heavy fixtures that would become more confusing if split immediately

Allowlist entries should be reviewed periodically and removed as files are split.

## How to split (recommended pattern)

- Split by command/subsystem/topic.
- Keep shared fixtures in `conftest.py` or a `fixtures.py`.

Example layout:

- `tests/unit/.../test_commands_execution_run.py`
- `tests/unit/.../test_commands_execution_apply.py`
- `tests/unit/.../test_commands_execution_artifacts.py`

## Example offender remediation (this PR)

This PR adds an allowlist entry for one existing offender as the exemplar policy implementation.
