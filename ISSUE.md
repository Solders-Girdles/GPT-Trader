# GitHub Issue #716: [ci] Resolve failing checks blocking PR #712
https://github.com/Solders-Girdles/GPT-Trader/issues/716

<!-- codex-handoff-key-ci-pr-712 -->

## Why / Context
PR #712 is blocked by failing CI checks that were not auto-remediated by `ci_fixer.mjs`.
- PR: https://github.com/Solders-Girdles/GPT-Trader/pull/712
- Branch: `codex/694-architecture-introduce-typed-order-event`
- Handoff reason: all fix commands failed

## Required Outcome
- Fix the CI failure(s) on the PR branch with the smallest safe change.
- Keep all required checks green.
- Avoid unrelated refactors.

## Failing Checks
- Lint & Format — https://github.com/Solders-Girdles/GPT-Trader/actions/runs/21924563914/job/63313599465
- Test Guardrails — https://github.com/Solders-Girdles/GPT-Trader/actions/runs/21924563914/job/63313599503
- Unit Tests (Core) — https://github.com/Solders-Girdles/GPT-Trader/actions/runs/21924563914/job/63313599463

## Evidence
- Lint & Format: Lint/format checks failed with auto-fixable formatting issues.
  - Lint & Format Lint (Ruff) 2026-02-11T21:55:36.3576897Z 35 | with pytest.raises(OrderEventSchemaError):
  - Lint & Format Lint (Ruff) 2026-02-11T21:55:36.3579133Z 35 | with pytest.raises(OrderEventSchemaError):
- Test Guardrails: Could not map failure to a known fix pattern.
  - Test Guardrails Check triage backlog 2026-02-11T21:55:41.2429764Z make: *** [Makefile:106: test-triage-check] Error 1
  - Test Guardrails Check triage backlog 2026-02-11T21:55:41.2446285Z ##[error]Process completed with exit code 2.
- Unit Tests (Core): Could not map failure to a known fix pattern.
  - Unit Tests (Core) Run unit tests (excluding TUI snapshots) 2026-02-11T21:55:36.3824662Z ##[group]Run uv run pytest tests/unit -n auto -q --ignore-glob=tests/unit/gpt_trader/tui/test_snapshots_*.py
  - Unit Tests (Core) Run unit tests (excluding TUI snapshots) 2026-02-11T21:55:36.3825459Z uv run pytest tests/unit -n auto -q --ignore-glob=tests/unit/gpt_trader/tui/test_snapshots_*.py

## File Pointers
- `src/gpt_trader/**`
- `tests/**`

## Verification
- `gh pr checkout 712 --repo Solders-Girdles/GPT-Trader`
- Run the failing check(s) locally and confirm pass.
- Confirm required PR checks are green.
