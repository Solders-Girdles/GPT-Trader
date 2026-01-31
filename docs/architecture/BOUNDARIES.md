# Module Boundaries

---
status: current
last-updated: 2026-01-31
---

This document defines the module boundaries for GPT-Trader. Keep dependency
flow one-way to preserve testability, avoid circular imports, and keep slices
independent.

## Layer Map (Top to Bottom)

```
Entry points (scripts/, gpt_trader.cli, gpt_trader.tui, gpt_trader.preflight)
    v
Composition root (gpt_trader.app)
    v
Domain slices (gpt_trader.features.*, gpt_trader.backtesting)
    v
Shared platform (gpt_trader.core, errors, monitoring, validation, persistence,
                 security, logging, observability, config, utilities)
    v
Stdlib + third-party libraries
```

### Responsibilities by Layer

- Entry points: argument parsing, environment loading, process orchestration.
  No business rules. Hand off to application services quickly.
- Composition root: `ApplicationContainer` wiring, configuration assembly,
  lifecycle hooks. This is the only place that should import multiple slices
  to stitch them together.
- Domain slices: trading features and integrations. Each slice owns its
  internal modules and tests and should be independently testable.
- Shared platform: cross-cutting services and abstractions (errors, validation,
  logging, persistence). These should not depend on feature slices.

## Allowed Import Directions

| From | Allowed to import | Avoid importing |
|------|-------------------|----------------|
| Entry points | `gpt_trader.app`, feature slices, shared platform, stdlib/third-party | None (but keep logic thin) |
| `gpt_trader.app` | Feature slices, shared platform, stdlib/third-party | Entry points |
| Feature slices | Same slice, shared platform, `gpt_trader.app.config` types | Other feature slices, `ApplicationContainer` |
| Shared platform | Other shared platform packages, stdlib/third-party | Feature slices, `gpt_trader.app` |
| Tests | Any module | N/A |

### Cross-slice collaboration

- Prefer shared abstractions (protocols, data models) in the shared platform
  and have slices depend on those instead of importing each other.
- If two slices need shared logic, extract it to
  `gpt_trader.features.strategy_tools` (when strategy-focused) or a new shared
  platform package, then wire it in `ApplicationContainer`.

## Decision Tree (Where Should New Code Live?)

1. Is it a user entrypoint, CLI flag, or TUI wiring? -> `gpt_trader.cli`,
   `gpt_trader.tui`, or `scripts/`.
2. Is it dependency wiring or lifecycle setup? -> `gpt_trader.app`.
3. Is it a trading capability, integration, or feature-specific behavior? ->
   `gpt_trader.features/<slice>/` (or `gpt_trader.backtesting` for backtests).
4. Is it shared across slices (errors, validation, persistence, monitoring)? ->
   `gpt_trader/<shared-package>/`.
5. Still unsure? Start in `docs/DEVELOPMENT_GUIDELINES.md` or ask for guidance
   before creating a new top-level package.
