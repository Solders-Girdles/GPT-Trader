# GitHub Issue #578: [architecture] Centralize HealthCheckRunner registration with typed results
https://github.com/Solders-Girdles/GPT-Trader/issues/578

## Title
`[architecture] Centralize HealthCheckRunner registration with typed results`

## Why / Context
Health checks are currently wired directly inside `HealthCheckRunner._execute_checks()` and again in `run_checks_sync()`. As we add checks (ticker freshness, degradation, etc.), it’s easy for async/sync paths to drift or for naming/details conventions to diverge.

## Scope
**In scope**
- Introduce a small, typed registry/descriptor for health checks (name + callable + “blocking vs fast” or similar).
- Use that registry in both `run_checks_sync()` and `_execute_checks()` so the check set stays consistent.
- Keep behavior the same (no semantic changes to health check outcomes).

**Out of scope**
- Reworking the health server API.
- Changing check logic (other than wiring).

Constraints:
- Keep changes small/mergeable.
- Deterministic tests only.

## Acceptance Criteria (required)
- [ ] There is a single source of truth for which checks run (no duplication between async and sync paths).
- [ ] Existing unit tests still pass, and at least one test asserts that the registry drives both paths.
- [ ] No behavior change in health check results (only wiring/structure).

## Implementation Notes / Pointers
**Likely files / modules:**
- `src/gpt_trader/monitoring/health_checks.py` (`HealthCheckRunner`)

**Related tests:**
- `tests/unit/gpt_trader/monitoring/test_health_checks_runner.py`

## Commands (local)
- `make lint-fmt-fix`
- `make typecheck`
- `pytest -q tests/unit/gpt_trader/monitoring/test_health_checks_runner.py`

## PR Requirements
- PR title should match the issue.
- PR body must include: `Fixes #<issue-number>`
- CI must be green.

## Codex-Ready Checklist (for the issue creator)
- [x] Clear acceptance criteria
- [x] At least one file pointer
- [x] Commands included
- [x] No ambiguous “do the right thing” language

