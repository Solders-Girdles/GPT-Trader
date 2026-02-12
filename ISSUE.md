# GitHub Issue #689: [qa] [quality] Reduce issue-queue skip category: duplicate
https://github.com/Solders-Girdles/GPT-Trader/issues/689

## Why / Context
Generated from live event-lane telemetry and ranked by the opportunity synthesizer. Value score: 0.694; source type: issue.skipped. If we reduce skip category duplicate, issue generation throughput and quality should improve.

## Dependency Metadata
- Value score: 0.694

## Scope
**In scope**
- Investigate the evidence chain and isolate the smallest bounded fix that addresses the signal.
- Implement the remediation in the likely modules listed below.
- Add deterministic verification that demonstrates signal improvement after the change.

**Out of scope**
- Broad refactors or architectural rewrites unrelated to this signal.
- Unbounded scope growth across unrelated components.

Constraints:
- Keep changes small/mergeable.
- Deterministic tests only.
- Avoid touching unrelated generated artifacts unless required.

## Acceptance Criteria (required)
- [ ] Root cause for the opportunity signal is identified and remediated with bounded blast radius.
- [ ] Deterministic tests or scripted checks are added/updated and fail before the fix but pass after.
- [ ] PR description includes before/after evidence tied to the originating signal.

## Implementation Notes / Pointers
**Likely files / modules:**
- `opportunity:opp_1bd46f779a`

**Related tests:**


**Edge cases to handle:**
- Signal may be intermittent; include at least one regression path for recurrence.
- Avoid introducing false positives in existing duplicate guards or cooldown logic.
- issue.skipped (evt_79dc76f75432)
- issue.skipped (evt_f1c080c016c7)
- issue.skipped (evt_03a32f88dea7)

## Commands (local)
- `make lint`
- `make test`

## PR Requirements
- PR title should match the issue.
- PR body must include: `Fixes #<issue-number>`
- CI must be green; if Agent Artifacts Freshness fails, run `uv run agent-regenerate`, then commit `var/agents/...` updates.

## Codex-Ready Checklist (for the issue creator)
- [x] Clear acceptance criteria
- [x] At least one file pointer
- [x] Commands included
- [x] No ambiguous "do the right thing" language
