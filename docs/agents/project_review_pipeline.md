# GPT-Trader Agent Review Pipeline

---
status: draft
last-updated: 2026-06-20
---

This document defines the recurring project-review pipeline for GPT-Trader. The
pipeline is intentionally staged: frequent scouts produce candidate findings,
and only validated, deduped findings are promoted to GitHub issues for Claw,
Hermes, and Codex review loops.

## Goals

- Keep a frequent agent scout close to current repo truth.
- Promote only actionable, evidence-backed findings.
- Use GitHub issues as the durable queue for Claw/Hermes implementation.
- Preserve GPT-Trader safety boundaries around broker access, trading
  execution, account capability, and human approval.
- Feed PR review feedback back into implementation agents without broadening
  scope.

## Stage 1: Scout

The scout is read-only and can run frequently. It should inspect current truth
before relying on memory or prior runs:

```bash
git status --short --branch
gh issue list --state open --limit 50
gh pr list --state open --limit 30
uv run agent-regenerate --verify
uv run local-ci --profile quick
```

The scout may run narrower commands when there are no relevant changes since a
recent run. It must not run real broker/API checks, canary operations, live
trading commands, production preflight, or anything that assumes account
capability unless a current owner-approved runbook explicitly opens that lane.

Good scout signals include stale generated artifacts, a repeated quick local-CI
failure, a stale or blocked open PR, a duplicated test cluster with a clear next
packet, a drift between docs and code, or a narrow architectural boundary issue.

Poor scout signals include speculative refactor ideas, transient network
failures, broad "clean up the repo" claims, or anything that lacks a named path,
command output, existing issue/PR context, or acceptance criteria.

## Stage 2: Normalize

Candidate findings use schema `gpt-trader.agent-finding.v1`.

Required fields:

```json
{
  "schema_version": "gpt-trader.agent-finding.v1",
  "finding_id": "stable-lowercase-id",
  "title": "Short actionable issue title",
  "severity": "low",
  "category": "ci",
  "summary": "One-paragraph explanation of the actionable defect or improvement.",
  "evidence": [
    {
      "kind": "command",
      "command": "uv run agent-regenerate --verify",
      "detail": "Command failed because generated files are stale."
    }
  ],
  "scope": {
    "paths": ["var/agents/index.json"],
    "out_of_scope": ["live trading changes"],
    "touches_trading_execution": false
  },
  "dedupe": {
    "search_terms": ["agent-regenerate", "stale var/agents"],
    "related_issues": []
  },
  "acceptance_criteria": [
    "Generated agent artifacts verify cleanly."
  ],
  "suggested_verification": [
    "uv run agent-regenerate --verify"
  ],
  "routing": {
    "candidate_for": ["claw"],
    "needs_human_decision": false,
    "blocked_by": []
  }
}
```

Allowed severities are `low`, `medium`, `high`, and `critical`.

Allowed categories are `bug`, `ci`, `docs`, `architecture`, `tests`,
`cleanup`, `security`, `trading-readiness`, and `tooling`.

If `scope.touches_trading_execution` is true, the finding must either be
docs/test-only and clearly execution-free, or `routing.needs_human_decision`
must be true. Use `docs/PRE_MIGRATION_DECISION_FRAMEWORK.md` for anything that
touches execution automation, broker adapters, venue support, account
capability, or AI-assisted execution.

## Stage 3: Promote

Promotion creates or updates a GitHub issue from a validated packet. Promote at
most one finding per scout run unless RJ explicitly requests a bulk pass.

Dry-run first:

```bash
uv run python scripts/maintenance/project_review_issue_promoter.py \
  --packet path/to/finding.json
```

Create or update the GitHub issue:

```bash
uv run python scripts/maintenance/project_review_issue_promoter.py \
  --packet path/to/finding.json \
  --create-issue \
  --create-labels
```

The promoter searches for the hidden `finding_id` marker before creating a new
issue. When a matching open issue exists, it comments with the refreshed packet
instead of creating a duplicate.

## Stage 4: Queue

GitHub issues are the durable queue. These labels are routing signals:

| Label | Meaning |
| --- | --- |
| `agent-review` | Finding came from the recurring GPT-Trader review lane |
| `agent-ready` | Finding passed promotion gates and is ready for implementation |
| `claw-candidate` | Candidate for Claw implementation |
| `hermes-candidate` | Candidate for Hermes implementation |
| `needs-human-decision` | Blocked on RJ or an explicit human gate |
| `codex-review-feedback` | Follow-up from Codex review comments or checks |

The promoter can create missing labels with `--create-labels`. If labels are
missing and that flag is not used, it keeps routing in the issue body and omits
unknown labels from the GitHub call.

## Stage 5: Implement

Claw and Hermes should treat each promoted issue as the implementation contract:

- work only the acceptance criteria in the issue
- preserve out-of-scope boundaries
- open a PR that links the issue
- run the smallest meaningful verification plus repo-required checks
- leave any broader discovery as a new finding instead of scope creep

The PR body should follow `.github/pull_request_template.md` and include
`Closes #<issue>`.

## Stage 6: Review Feedback

When the ChatGPT-Codex-Connector review bot or GitHub checks produce feedback,
convert each actionable item into a bounded fix packet for the active executor:

- PR URL and branch
- review comment or check URL
- exact file/line when available
- required behavior change
- verification command
- stop condition

If the feedback contradicts the issue acceptance criteria or crosses a human
gate, label the issue or PR `needs-human-decision` and stop the automation loop.

## Stage 7: Merge And Closeout

After checks and review are clean, merge according to `AGENTS.md`:

```bash
gh pr merge --auto --squash --delete-branch
```

After merge, confirm the issue is closed or comment with the merged PR and any
follow-up. New work gets a new finding packet.

## Cadence

Start with an hourly Codex automation. Tighten the cadence only after the issue
promotion rate is low, dedupe is reliable, and Claw/Hermes are not receiving
stale or duplicate work.
