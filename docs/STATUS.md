# Project Status — Where We Actually Are

---
status: current
---

This is the **factual current-state tracker**: what is actually shipped and
working, as distinct from what the direction docs describe as the destination.

- The [Operating Rubric](OPERATING_RUBRIC.md) and
  [Pre-Migration Decision Framework](PRE_MIGRATION_DECISION_FRAMEWORK.md) define
  **where we are going** and the evidence each capability needs.
- This doc records **where we are** — updated whenever the shipped reality
  changes, so planning and backlog work start from reality, not from
  aspirational stage descriptions.

If a stage description in another doc disagrees with observed code, this doc
wins until the other doc is reconciled. Verify file/function references here
before relying on them — they reflect the dated snapshot below.

## Snapshot (2026-06-28)

| Stage | Capability | State |
|-------|------------|-------|
| **Stage 0 — Rails** | Record, workflow, audit, eligibility, budget, policy, operator controls | **Complete** — all rubric evidence shipped and tested |
| **Stage 1 — Human-approved loop** | Propose → review → paper-execute → attribute → report | **In progress (bridge remaining)** — operator-usable real-data proposal and paper-fill reconciliation surfaces exist; the live strategy path is not yet routed through the approval workflow |
| **Stage 2 — Bounded autonomy** | Auto-approval, high-water-mark, kill switch, runtime circuit breaker | **Not started** |
| **Stage 3 — Self-directed entity** | Capital allocation, self-adjusted budgets, self-review | **Not started** |

## Stage 0 — Rails (complete)

Every capability the rubric lists for Stage 0 has shipped, tested evidence in
`src/gpt_trader/features/trade_ideas/`: broker-neutral record + hashing
(`models.py`), approval-gated state machine (`workflow.py`), append-only audit
log (`audit.py`), eligibility + approval policy (`eligibility.py`, `policy.py`),
versioned risk budget (`budget.py`), and operator lifecycle controls —
reject / request-changes / resubmit / cancel / expire (`service.py`). The full
human lifecycle is exposed through the `gpt-trader ideas` CLI.

The rails are done.

## Stage 1 — Human-approved loop (in progress)

| Capability | State | Notes |
|------------|-------|-------|
| Reviewer tooling (CLI + TUI) | **Done** | `ideas list/show/approve/reject/...` plus the TUI review screen |
| Outcome attribution | **Done** | `closeout.py` records resolution + realized PnL vs max-loss |
| Track-record report | **Done** | `ideas report` |
| Expiry sweep | **Partial** | `service.expire_due_ideas()` exists; not yet wired to a scheduler |
| Proposer from **real market data** | **Operator-usable** | `snapshot_builder.py` plus `ideas snapshot build` can create Coinbase candle `MarketSnapshot` JSON for `ideas propose-baseline` — issue #1031 closed 2026-06-28 |
| Paper fills onto audit trail | **Operator-usable** | `paper_reconciliation.py` plus `ideas reconcile-paper-fills` matches persisted paper/mock EventStore fills and can apply submission/fill audit events for paper/dev/mock profiles — issue #1035 closed 2026-06-28 |

The remaining loop-closing gap is the strategy-signal bridge: the existing
TA/ensemble path still does not emit trade ideas through `TradeIdeaService`.

## The structural fact: two disconnected worlds

The live TA bot (`features/live_trade/`, the ensemble/intelligence strategies,
the broker) and the trade-idea workflow (`features/trade_ideas/`) share **zero
references in either direction** (verified 2026-06-28). The trade-idea system
never touches a broker; the live engine never touches the approval workflow.

Consequence: the trading intelligence already built does **not** flow through
the approval-gated rails. Bridging that gap — not adding new feature surface —
is the central piece of work. See the recorded direction below.

## Current stance (2026-06-28 regrounding)

1. **Route the next build through the existing spine.** Real-data snapshots and
   paper-fill reconciliation now exist as operator-usable surfaces; the next
   core build item is the strategy-signal -> trade-idea adapter (#1033).
2. **Bridge the existing bot; do not grow a second proposer brain.** When the
   loop is built, trade ideas should come from the existing TA/ensemble
   intelligence routed through the gate (issue #1033), not from independent,
   duplicate proposers. Independent/standalone proposers (e.g. #1034) are
   de-prioritized accordingly.
3. **Docs track reality.** The rubric/framework remain the destination; this
   STATUS doc is the living "you are here" and is updated as state changes.

This preserves the 2026-06-27 owner alignment while reflecting the source and
GitHub issue state observed on 2026-06-28.

## Loop-closing spine

The work that actually closes the Stage 1 loop, in order:

1. **#1031** — Coinbase `MarketSnapshot` builder (real data in) — closed 2026-06-28.
2. **#1035** — reconcile paper fills onto the audit trail (execution end) — closed 2026-06-28.
3. **#1033** — strategy-signal → trade-idea adapter (bridge the two worlds) — open and next.

Then: budget gates (#1036), preflight readiness (#1037), pending-approval
notifications (#1038). Proposer-enrichment, tournaments, MCP, reporting polish,
futures, and auto-approval are all post-loop and were triaged accordingly on the
GitHub issue tracker (`triage:*` labels).

## How to keep this doc honest

- Update it when a capability moves between Missing / Partial / Done — ideally in
  the same PR that changes the state.
- Prefer concrete pointers (file, function, issue number) over prose claims.
- When this doc and a direction doc disagree, fix the direction doc or open a
  decision-log entry; don't let the gap persist silently.
