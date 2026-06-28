# Trade-ideas CLI is the active discovery lane

---
status: accepted
date: 2026-06-22
deciders: rj
supersedes:
superseded-by:
---

## Context

The project needed a single viable "door" into the human-approved trade-idea
workflow, rather than several speculative branches competing as the discovery
surface.

## Decision

Treat the trade-ideas CLI (`gpt-trader ideas`) as the active discovery lane: the
operator entry point into the existing human-approved trade-idea workflow. It is
a decision-support lane, **not** an execution or broker-action lane. The v1
product thesis remains AI-assisted decision support with human-approved
execution.

## Consequences

- The `ideas` command group (propose / list / show / approve / reject / audit
  verify and the lifecycle controls) is the canonical operator surface over the
  audited service path; CLI/TUI are thin adapters over the library.
- Reviewer tooling is now shipped — see [STATUS.md](../STATUS.md) for the
  current per-capability state.

## Safety boundary

No broker API calls, account actions, credential reads, live trading, autonomous
order submission, or execution enablement.
