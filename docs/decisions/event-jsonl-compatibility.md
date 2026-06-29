# Event JSONL: accepted fallback or import-only historical data

---
status: proposed
date: 2026-06-28
deciders: rj
supersedes:
superseded-by:
---

## Context

The event store historically accepted a JSONL fallback shape. As compatibility
shims are collapsed (see `docs/DEPRECATIONS.md`), the project needs to decide
whether JSONL remains a supported runtime fallback or becomes import-only
historical data.

## Options

- **A — Keep JSONL as an accepted fallback** for runtime event storage.
- **B — Make JSONL import-only historical data** — readable for backfill/analysis
  but no longer a supported write path, removing the compatibility surface.

## Decision

_Pending owner._

## Consequences

Affects the event-store write path and the remaining compatibility inventory in
`docs/DEPRECATIONS.md`. Option B removes a shim but requires confirming no live
path depends on JSONL writes.

## Safety boundary

No execution or account impact; storage-format decision only.
