# GPT-Trader Decision Log

---
status: current
---

Durable product and engineering direction for GPT-Trader. Use this log for
decisions that should outlive chat, PR receipts, and local branch state.

## 2026-06-11 — Accept Pre-Migration Direction And Trade-Idea Rails

- **Status:** accepted direction; Stage 0 trade-idea rails implemented where
  noted below
- **Owner:** Project owner
- **Decision / direction:** Use the
  [Pre-Migration Decision Framework](PRE_MIGRATION_DECISION_FRAMEWORK.md) as
  the execution-migration gate. The accepted path is approval-gated execution
  (`human_approved_execution`) as the validation phase, with
  `bounded_autonomy` only as a later destination. Coinbase spot plus CFM futures
  research are the current Coinbase-only scope; Robinhood, options execution,
  INTX-only work, and other venues stay out of scope unless a later decision
  packet explicitly reopens them.
- **Implemented evidence:** The trade-idea slice implements broker-neutral
  records (`src/gpt_trader/features/trade_ideas/models.py`), the approval
  workflow (`src/gpt_trader/features/trade_ideas/workflow.py`), append-only
  audit events (`src/gpt_trader/features/trade_ideas/audit.py`), seeded
  versioned risk budgets (`src/gpt_trader/features/trade_ideas/budget.py`),
  eligibility and approval policy (`src/gpt_trader/features/trade_ideas/eligibility.py`,
  `src/gpt_trader/features/trade_ideas/policy.py`), and operator lifecycle
  controls (`src/gpt_trader/features/trade_ideas/service.py`).
- **Safety boundary:** This decision does not authorize real broker/API calls,
  live trading commands, production preflight, canary operations, credential
  reads, money movement, or order submission. Any non-manual execution lane
  still needs a decision packet or approved runbook that names the lane,
  constraints, verification boundary, and rollback/kill-switch expectations.
- **Still pending:** Official venue/API/account capability review for any
  non-manual execution, product-specific migration envelopes, bounded-autonomy
  strategy envelopes, kill-switch drills, and any policy change that would
  allow submission without explicit approval.
- **Rejected alternatives:** Do not migrate unrestricted spot-only automation
  into the new shape. Do not use existing broker adapters or live profiles as
  proof that a product or venue should be automated.

## 2026-06-22 — Continue Trade-Ideas CLI As The Active Discovery Lane

- **Status:** accepted direction, implementation still WIP
- **Owner:** Active implementation agent
- **Reviewer:** Decision and evidence review agents
- **Decision / direction:** Treat `codex/trade-ideas-cli` as the active viable
  GPT-Trader discovery lane. It is the CLI door into the existing
  human-approved trade-idea workflow, not an execution or broker-action lane.
- **Evidence:** The current WIP matches `docs/specs/TRADE_IDEA_CLI_SPEC.md`
  and `docs/specs/TRADE_IDEA_INTERFACES_DESIGN_NOTES.md`; `gpt-trader ideas
  --help` exposes the expected command group; focused trade-ideas domain tests
  passed (`136 passed`); touched files passed `ruff`; import boundaries passed.
- **Safety boundary:** No broker API calls, account actions, credential reads,
  live trading, autonomous order submission, or execution enablement. The v1
  product thesis remains AI-assisted decision support with human-approved
  execution.
- **Trading/account/credential/runtime impact:** None from the direction
  decision. The active WIP is local code only and still needs CLI tests before
  package/PR consideration.
- **Stop condition:** Stop treating the branch as merely speculative once the
  CLI test plan exists and the focused quality gates pass; otherwise park it
  with the exact failing tests or scope gap.
- **Next bounded experiment:** Add focused CLI tests for the `ideas` command
  group, starting with propose/list/show/approve/audit verify against a
  `tmp_path` ideas root. Run black on touched files before any package step.
- **Rejected alternatives:** Do not publish stale local branches. Do not use
  OpenClaw's anchor as GPT-Trader's roadmap. Do not promote momentum receipts
  into product direction without a project-native decision entry.

## 2026-06-27 — Re-Orient On Actual State; Stabilize Before Closing The Loop

- **Status:** accepted direction
- **Owner:** Project owner
- **Context:** An AI issue-generation run produced a large `feat(trade ideas):`
  backlog implied by the staged-autonomy docs. Review found the backlog reached
  for breadth (tournaments, MCP, reporting, filtering, exports) ahead of a
  working loop. Ground-truth check established: Stage 0 rails are complete; Stage
  1 is ~2/3 built but not a closed loop; and the live TA bot and the trade-idea
  workflow are fully disconnected (zero references either direction). See
  [Project Status](STATUS.md).
- **Decision / direction:**
  1. **Stabilize and reconcile before extending the loop.** Pause net-new
     feature surface; reconcile docs to reality and keep rails/tests healthy
     first. The loop-closing spine (#1031 → #1035 → #1033) is the planned next
     build, not this stabilization pass.
  2. **Bridge the existing bot; do not grow a second proposer brain.** When the
     loop is built, trade ideas come from the existing TA/ensemble intelligence
     routed through the approval gate (#1033), not from independent duplicate
     proposers. Standalone proposers (e.g. #1034) are de-prioritized.
  3. **Docs track reality.** Added the living [Project Status](STATUS.md);
     corrected stale stage labels in the [Operating Rubric](OPERATING_RUBRIC.md).
- **Safety boundary:** No change to the autonomy boundary. Still
  `human_approved_execution`; no broker/API calls, credential reads, money
  movement, or order submission authorized by this entry.
- **Backlog action:** Open issues triaged with `triage:build-now`,
  `triage:build-next`, `triage:blocked`, `triage:defer`; three export duplicates
  (#1024, #1022, #1020) merged into #1044 and closed.
- **Rejected alternatives:** Do not build the loop or jump toward Stage 2 before
  stabilizing. Do not build trade-ideas as a second, independent proposer brain
  parallel to the existing bot. Do not treat doc-implied stages as a backlog.
