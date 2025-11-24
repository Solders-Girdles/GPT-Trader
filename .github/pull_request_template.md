## Summary
<!-- What and why. Keep it crisp. Link related issues. -->
Closes #<issue-id(s)>

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

## Complexity & Quality Gates
- [ ] Mypy/type-check passed
- [ ] Xenon/radon complexity gate passed (CC ≤ 15 on orchestration)
- [ ] No `sys.path` hacks in tests
- [ ] No `MagicMock` on `async def` (use `AsyncMock`)
- [ ] Import-lint rules respected (no “import up the stack”)

## Observability & Errors
- [ ] Structured JSON logs for new/changed paths
- [ ] Errors include diagnostic context (symbol, order_id, correlation_id)
- [ ] Added/updated sampling examples in tests (if applicable)

## Configuration
- [ ] If config changed: `python -m gpt_trader.cli.config validate --profile <name>` passes
- [ ] Env docs re-generated (if applicable) and committed
- [ ] No `ConfigLoader` usages introduced (ConfigManager-only)

## Breaking Changes
- [ ] None
- [ ] Documented in PR body and release notes if any

## Screenshots / Logs (optional)
<!-- Paste sample structured logs or screenshots to aid review -->

## Checklist
- [ ] Acceptance criteria in linked issue(s) met
- [ ] Affected paths listed in PR description
- [ ] Changelog entry (if repo uses one)
