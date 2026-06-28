# Stabilize and reconcile before closing the Stage 1 loop

---
status: accepted
date: 2026-06-27
deciders: rj
supersedes:
superseded-by:
---

## Context

An AI issue-generation run produced a breadth-first `feat(trade ideas):` backlog
(tournaments, MCP, reporting, filtering, exports) implied by the staged-autonomy
docs — ahead of a working loop. A ground-truth check established: Stage 0 rails
are complete; Stage 1 is ~2/3 built but not a closed loop; and the live TA bot
(`features/live_trade/`) and the trade-idea workflow (`features/trade_ideas/`)
are fully disconnected (zero references in either direction).

## Decision

1. **Stabilize and reconcile before extending the loop.** Pause net-new feature
   surface; reconcile docs to reality and keep the rails and tests healthy first.
   The loop-closing spine is the planned next build, tracked as GitHub issues —
   not as a doc-resident backlog.
2. **Bridge the existing bot; do not grow a second proposer brain.** When the
   loop is built, trade ideas come from the existing TA/ensemble intelligence
   routed through the approval gate, not from independent duplicate proposers.
3. **Docs track reality.** [STATUS.md](../STATUS.md) is the living "you are here"
   tracker; the rubric/direction remain the destination.

## Consequences

- Open issues are triaged on GitHub with `triage:*` labels; the backlog lives in
  the issue tracker, not in a roadmap doc.
- Loop-closing work (real-data proposer, automatic paper-fill reconciliation,
  strategy-signal → trade-idea bridge) is sequenced as issues.

## Safety boundary

No change to the autonomy boundary; still `human_approved_execution`. No
broker/API calls, credential reads, money movement, or order submission.
