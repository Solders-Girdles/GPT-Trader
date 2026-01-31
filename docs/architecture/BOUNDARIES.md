# Architecture Boundaries

---
status: draft
last-updated: 2026-01-31
---

This document defines the current layering and dependency boundaries used during the cleanup
campaign. It is descriptive of the codebase as it exists today (not an aspirational target).

## Layers (inner to outer)

| Layer | Primary ownership | Notes |
|-------|-------------------|-------|
| Domain + shared foundations | `src/gpt_trader/core/`, `src/gpt_trader/errors/`, `src/gpt_trader/validation/`, `src/gpt_trader/config/`, `src/gpt_trader/utilities/`, `src/gpt_trader/logging/` | Types, math, error taxonomy, validation helpers, and shared utilities. |
| Interfaces (protocols) | `src/gpt_trader/app/protocols.py`, `src/gpt_trader/features/**/protocols.py` | Contracts for brokers, risk managers, and runtime services. Keep these import-only and light. |
| Shared configuration | `src/gpt_trader/app/config/` | `BotConfig` and profile loading. Imports strategy configs from `features/live_trade/strategies`. Used across layers as a shared input surface. |
| Feature slices (business logic) | `src/gpt_trader/features/` (live_trade, intelligence, data, research, optimize, strategy_dev, strategy_tools) | Trading logic, strategies, guard stack, research/optimization workflows. |
| Shared engines | `src/gpt_trader/backtesting/` | Canonical backtesting engine used by research and optimization slices. |
| Adapters + infrastructure | `src/gpt_trader/features/brokerages/`, `src/gpt_trader/persistence/`, `src/gpt_trader/monitoring/`, `src/gpt_trader/observability/`, `src/gpt_trader/security/` | External integrations, IO, stores, telemetry, secrets. `features/brokerages` is an adapter slice. |
| App/runtime + entrypoints | `src/gpt_trader/app/`, `src/gpt_trader/cli/`, `src/gpt_trader/tui/`, `src/gpt_trader/preflight/`, `scripts/production_preflight.py` | Composition root, config loading, runtime lifecycle, and operator entrypoints. |

### Configuration note

`BotConfig` and profile loading live in `src/gpt_trader/app/config/` and are imported by
features and adapters (for example broker factories and REST services). Treat config types as
shared inputs, but avoid importing the container or CLI/TUI layers from lower-level modules.

## Allowed dependency directions (imports)

- Domain + shared foundations depend only on stdlib and third-party packages.
- Interfaces depend on domain/shared foundations only.
- Feature slices may import domain/shared foundations, interfaces, and shared config types.
  Cross-slice imports should be rare; prefer `features/strategy_tools/` or `features/data/`
  for shared helpers.
- Adapters/infrastructure may import domain/shared foundations, interfaces, and shared config types.
  Avoid importing CLI/TUI/preflight or `app.container` from adapter code.
- App/runtime and entrypoints may import from all lower layers to wire dependencies.

## Decision tree: where to put new code

- New trading logic, strategy, execution guard, or risk rule? Add it under
  `src/gpt_trader/features/live_trade/` (or the appropriate feature slice).
- New research, optimization, or backtesting workflow? Use `src/gpt_trader/features/research/`
  or `src/gpt_trader/features/optimize/`. Core backtesting engine changes belong in
  `src/gpt_trader/backtesting/`.
- New external integration (exchange, storage, telemetry, secrets)? Use
  `src/gpt_trader/features/brokerages/`, `src/gpt_trader/persistence/`,
  `src/gpt_trader/monitoring/`, `src/gpt_trader/observability/`, or
  `src/gpt_trader/security/`.
- New operator workflow or wiring? Use `src/gpt_trader/app/` for wiring and config, and
  `src/gpt_trader/cli/`, `src/gpt_trader/tui/`, or `src/gpt_trader/preflight/` for entrypoints.
- New shared type, math, or helper used across slices? Prefer `src/gpt_trader/core/`,
  `src/gpt_trader/errors/`, `src/gpt_trader/validation/`, or `src/gpt_trader/utilities/`.
- Need a new contract for DI/testing? Add a protocol in `src/gpt_trader/app/protocols.py`
  or the relevant slice `protocols.py`.
