# Accept the staged-autonomy direction (human-approved execution → bounded autonomy)

---
status: accepted
date: 2026-06-11
deciders: rj
supersedes:
superseded-by:
---

## Context

GPT-Trader's destination is an autonomous trading entity. Before building any
execution path, the owner fixed *what AI may do*, *which products are in scope*,
and *where execution happens* — three decisions that must not be blended.

## Decision

- **Destination:** an autonomous trading entity (`bounded_autonomy`) that
  observes markets, does its own research, and manages funds inside
  machine-enforced limits.
- **Path:** approval-gated execution (`human_approved_execution`) is the required
  *validation phase*, not the end state. AI produces complete trade-idea records
  (thesis, entry, invalidation, max loss, sizing, expiry) that receive an
  explicit human approval decision. Autonomy is granted per strategy envelope
  only after the approval-phase track record, risk budgets, kill switches, and
  audit logs exist.
- **Hard v1 rule:** AI never submits live orders. Any API execution lane requires
  an explicit approval event **and** a decision packet or approved runbook that
  names the lane, constraints, verification boundary, and rollback/kill-switch
  expectations.
- **Scope:** Coinbase only (spot + CFM futures). Options, Robinhood, event
  contracts, and other venues are out of scope until the Coinbase lane works end
  to end and a fresh venue/API/account capability review reopens them.
- **INTX perpetuals:** frozen — no new work or tests; remove INTX-only surfaces
  opportunistically when they block other work. (Remaining default-hygiene item:
  [intx-default-derivatives-venue](intx-default-derivatives-venue.md).)
- **Credentials:** CDP JWT only; legacy Coinbase credential support is removed.
  The TUI keeps format detection solely to reject legacy keys with guidance.
- **Internal architecture:** broker-neutral trade-idea / risk / approval / audit
  records are canonical. Broker-specific payloads are derived artifacts created
  after approval; adapters adapt to the record, never reshape it.
- **Existing TA bot:** a gated implementation asset; do not expand its autonomous
  execution surface ahead of the ladder above.

## Consequences

- The Stage 0 rails are implemented and tested in
  `src/gpt_trader/features/trade_ideas/` (record + hashing, approval workflow,
  append-only audit, eligibility/policy, versioned budget, operator lifecycle).
- The migration-trigger discipline in [DIRECTION.md](../DIRECTION.md) gates any
  execution-path or broker-adapter work.
- An official venue/API/account capability review remains **pending** before any
  non-manual execution lane.

## Safety boundary

This decision does not authorize real broker/API calls, live trading commands,
production preflight, canary operations, credential reads, money movement, or
order submission.

> Absorbs the former framework "Accepted Direction" + "Initial Decision Record"
> table and the cleanup-roadmap "Decided (2026-06-11)" rows.
