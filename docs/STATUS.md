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
| **1 — Human-approved loop** | **In progress** — workflow, review tooling, attribution, and paper-fill reconciliation exist; closing the loop end-to-end on real market data is the remaining work |
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

The shipped surfaces — reviewer tooling (CLI + TUI), outcome attribution,
track-record report, and paper-fill reconciliation onto the audit trail — turn
the rails into most of a loop. What closes it is a proposer fed by **real market
data** routed from the existing TA/ensemble intelligence (not hand-authored
fixtures, and not a second proposer brain — see
[stabilize-before-closing-the-loop](decisions/stabilize-before-closing-the-loop.md)).

Track the precise remaining items in the issue tracker rather than here; this
paragraph is the shape, the queue is the truth.

## The structural fact

The live TA bot (`features/live_trade/`) and the trade-idea workflow
(`features/trade_ideas/`) historically shared **zero references in either
direction**: the trading intelligence already built does not yet flow through the
approval-gated rails. Bridging that seam — not adding new feature surface — is the
central remaining piece of Stage 1.

## How to keep this doc honest

- Update it when a capability moves between missing / partial / done — ideally in
  the same PR that changes the state.
- Prefer concrete pointers (file, function, issue number) over prose claims, and
  route volatile specifics (open-issue lists, per-ticket status) to the tracker.
- When this doc and a direction doc disagree, fix the direction doc or open a
  `proposed` decision; don't let the gap persist silently.
