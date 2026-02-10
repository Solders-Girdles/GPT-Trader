# GitHub Issue #675: [enhancement] Add strategy profile diff command for active runtime
https://github.com/Solders-Girdles/GPT-Trader/issues/675

## Why / Context
Operators need a direct way to understand profile drift between configured and active runtime values.

## Scope
**In scope**
- Add a command that diffs active runtime profile values against configured baseline.
- Provide deterministic output suitable for automation parsing.
- Document diff semantics and ignored fields.

**Out of scope**
- Automatic profile mutation based on diff output.
- Schema changes to unrelated strategy modules.

Constraints:
- Keep changes small/mergeable.
- Deterministic tests only.
- Avoid touching unrelated generated artifacts unless required.

## Acceptance Criteria (required)
- [ ] Command prints stable diff entries for changed/unchanged/missing keys.
- [ ] Output can be emitted in machine-readable mode.
- [ ] Ignored/noisy fields are explicit and tested.
- [ ] Unit tests cover no-diff, partial-diff, and missing-profile scenarios.

## Implementation Notes / Pointers
**Likely files / modules:**
- `src/gpt_trader/features/strategy_dev/config/strategy_profile.py`
- `src/gpt_trader/features/strategy_dev/config/registry.py`
- `src/gpt_trader/cli/commands/optimize/compare.py`

**Related tests:**
- `tests/unit/gpt_trader/features/strategy_dev/config/test_strategy_profile.py`
- `tests/unit/gpt_trader/features/strategy_dev/config/test_registry.py`
- `tests/unit/gpt_trader/cli/commands/optimize/test_commands_execution.py`

**Edge cases to handle:**
- Nested config values should diff deterministically.
- Missing active-profile values should be represented explicitly.
- Ordering in rendered diffs should remain stable.

## Commands (local)
- `make fmt`
- `make lint`
- `uv run pytest -q tests/unit/gpt_trader/features/strategy_dev/config/test_strategy_profile.py tests/unit/gpt_trader/cli/commands/optimize/test_commands_execution.py`

## PR Requirements
- PR title should match the issue.
- PR body must include: `Fixes #<issue-number>`
- CI must be green; if Agent Artifacts Freshness fails, run `uv run agent-regenerate`, then commit `var/agents/...` updates.

## Codex-Ready Checklist (for the issue creator)
- [x] Clear acceptance criteria
- [x] At least one file pointer
- [x] Commands included
- [x] No ambiguous "do the right thing" language
