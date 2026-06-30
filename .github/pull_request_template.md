## Summary
<!-- What and why. Keep it crisp. Link related issues. -->
Related issue / finding / routed package:

## Type of change
- [ ] Refactor
- [ ] Feature
- [ ] Bug fix
- [ ] Tests only
- [ ] CI/Docs/Tooling

## Scope & Impact
- Affected areas / components:
- Any behavior changes? If yes, describe and link tests.

## Test Plan
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Contract tests (if applicable)
- [ ] Ran locally: `pytest -m "unit and not slow"`
- [ ] Markers used appropriately (`unit`/`integration`/`slow`/`perf`)

## Quality Gates
- [ ] `make ci-required` passes locally (lint/format, docs audits, type check, agent freshness, TUI CSS, test guardrails, core unit tests)
- [ ] `uv run mypy src/gpt_trader` clean, or new errors documented in the summary
- [ ] No `sys.path` hacks in tests; `AsyncMock` (not `MagicMock`) on `async def`
- [ ] Import boundaries respected (`scripts/ci/check_import_boundaries.py`)

## Observability & Errors
- [ ] Structured JSON logs for new/changed paths
- [ ] Errors include diagnostic context (symbol, order_id, correlation_id)

## Agent Artifacts & Config (if touched)
- [ ] If `var/agents/**` inputs changed (`scripts/agents/**`, `config/environments/.env.template`): ran `uv run agent-regenerate` and committed artifacts (`uv run agent-regenerate --verify` clean)
- [ ] If docs changed: `make agent-docs-links` passes

## Breaking Changes
- [ ] None
- [ ] Documented in PR body and the linked issue if any

## Screenshots / Logs (optional)
<!-- Paste sample structured logs or screenshots to aid review -->

## Checklist
- [ ] Acceptance criteria in linked issue(s), finding packet, or routed package met
- [ ] Affected paths listed in PR description
