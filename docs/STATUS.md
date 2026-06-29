# Project Status — Where We Actually Are

---
status: current
---

The factual **current-state** tracker: what is actually shipped, as distinct from
[DIRECTION.md](DIRECTION.md) (the destination and gates) and
[decisions/](decisions/README.md) (what was decided and why). This doc stays
small and points at the source of truth — it does **not** restate the ladder, the
backlog, or any decision. When a stage description elsewhere disagrees with
observed code, this doc wins until the other is reconciled.

Verify file/function/issue references before relying on them; they reflect the
dated snapshot below. The next work is the live GitHub issue queue
(`triage:build-now`), never a list copied here.

## Snapshot (2026-06-28)

Stage definitions live in [DIRECTION.md](DIRECTION.md#the-ladder); this is only
the state per stage.

| Stage | State |
|-------|-------|
| **0 — Rails** | **Complete** — all rubric evidence shipped and tested |
| **1 — Human-approved loop** | **In progress (bridge remaining)** — reviewer tooling, attribution, real-data snapshot proposal (#1031), and paper-fill reconciliation (#1035) are operator-usable; the live strategy path is not yet routed through the approval workflow |
| **2 — Bounded autonomy** | Not started |
| **3 — Self-directed entity** | Not started |

## Stage 0 — Rails (complete)

Every Stage 0 capability has shipped, tested evidence in
`src/gpt_trader/features/trade_ideas/`: broker-neutral record + hashing
(`models.py`), approval-gated state machine (`workflow.py`), append-only audit log
(`audit.py`), eligibility + approval policy (`eligibility.py`, `policy.py`),
versioned risk budget (`budget.py`), outcome attribution (`closeout.py`), and
operator lifecycle controls (`service.py`). The full lifecycle is exposed through
the `gpt-trader ideas` CLI.

## Stage 1 — Human-approved loop (in progress)

The shipped surfaces turn the rails into most of a loop: reviewer tooling
(CLI + TUI), outcome attribution, the track-record report, real-data
`MarketSnapshot` proposal (`ideas snapshot build` → `ideas propose-baseline`,
#1031 closed 2026-06-28), and paper-fill reconciliation onto the audit trail
(`ideas reconcile-paper-fills`, #1035 closed 2026-06-28).

The one remaining gap is the **strategy-signal bridge**: the existing TA/ensemble
intelligence still does not emit trade ideas through `TradeIdeaService` (#1033,
open and next). Build it through the existing spine, not a second proposer brain —
see
[stabilize-before-closing-the-loop](decisions/stabilize-before-closing-the-loop.md).
Track precise per-ticket status in the issue queue, not here.

## The structural fact

The live TA bot (`features/live_trade/`) and the trade-idea workflow
(`features/trade_ideas/`) still share **zero references in either direction**
(verified 2026-06-28): the trading intelligence already built does not yet flow
through the approval-gated rails. Bridging that seam (#1033) — not adding new
feature surface — is the central remaining piece of Stage 1.

## How to keep this doc honest

- Update it when a capability moves between missing / partial / done — ideally in
  the same PR that changes the state.
- Prefer concrete pointers (file, function, issue number) over prose claims, and
  route volatile specifics (open-issue lists, per-ticket status) to the tracker.
- When this doc and a direction doc disagree, fix the direction doc or open a
  `proposed` decision; don't let the gap persist silently.
