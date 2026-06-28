# Meaning of the prod and canary profiles under the approval ladder

---
status: proposed
date: 2026-06-28
deciders: rj
supersedes:
superseded-by:
---

## Context

Trading profiles are config snapshots, not execution approval. Under the
staged-autonomy direction
([accept-staged-autonomy-direction](accept-staged-autonomy-direction.md)),
enabling a live profile does not grant authority to submit orders — the gates in
[DIRECTION.md](../DIRECTION.md) and recorded human approval do. The `prod` and
`canary` profiles still carry historical "live-operation asset" framing that can
read as if selecting them were sufficient to operate live.

## Options

- **A — Keep `prod` / `canary` as live-operation assets**, explicitly gated by
  readiness evidence + recorded approval. Update docs/tests so the profile name
  is never treated as approval.
- **B — Redefine them as labels under the approval ladder** (e.g. capped
  validation lanes that only exist within a recorded approval envelope), retiring
  the standalone "live profile" concept.

## Decision

_Pending owner._

## Consequences

Touches `ProfileLoader` semantics, preflight gating, and profile language across
docs. No code change should treat a profile value as execution approval until
this is decided.

## Safety boundary

No execution authorized. This is a config-semantics and policy decision.
