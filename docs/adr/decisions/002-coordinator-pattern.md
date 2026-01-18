# ADR 002: Coordinator Pattern for the Orchestration Layer

## Status
Superseded - Orchestration package removed during the DI migration (Engine Context pattern lives in `features/live_trade/engines/`)

> **Note:** This ADR is historical. The coordinator registry and orchestration
> paths referenced below were removed during the DI migration.

## Context
- The original orchestration layer coupled coordinator logic tightly to `TradingBot` (formerly `PerpsBot`), making it difficult to test modules in isolation.
- Lifecycle concerns (initialisation, background tasks, shutdown) were scattered across multiple classes, leading to inconsistent behaviour and duplicated logic.
- Adding new orchestration features required deep knowledge of bot internals because dependencies were pulled directly from the bot instance.
- Health monitoring for background processes was ad-hoc and incomplete, creating blind spots during production incidents.

## Decision
- Introduce a coordinator pattern centred on three core abstractions:
  - `CoordinatorContext`: Immutable snapshot of shared dependencies passed between coordinators.
  - `BaseCoordinator` & `Coordinator` protocol: Defines a consistent lifecycle contract (`initialize`, `start_background_tasks`, `shutdown`, `health_check`).
  - `CoordinatorRegistry`: Registers coordinators, manages lifecycle orchestration, and propagates context updates.
- Migrate Runtime, Execution, Strategy, and Telemetry concerns into concrete coordinators that each consume the shared context.
- Update `TradingBot` to build coordinators through the registry and expose them via properties for backwards compatibility.
- Refactor `LifecycleManager` to delegate initialisation, task start-up, and shutdown to the registry.
- Retire legacy facade classes in favour of importing coordinators directly, simplifying the public API.

## Consequences
### Positive
- **Reduced coupling**: Coordinators now depend on the lightweight context rather than the entire bot.
- **Improved testability**: Unit tests can exercise coordinators with minimal fixtures.
- **Consistent lifecycle**: One registry manages initialise/start/stop flows, preventing divergent behaviour.
- **Health reporting**: Every coordinator surfaces `HealthStatus`, enabling richer monitoring.
- **Easier extensibility**: New orchestration responsibilities only require implementing the coordinator protocol.

### Negative
- **Additional abstraction**: Developers must understand the registry + context pattern before extending the orchestrator.
- **Context management**: The shared context must stay up-to-date; careless mutation can lead to stale dependencies.

### Migration Strategy
- Phase 1 (Completed): Introduced context/registry infrastructure and migrated Runtime/Execution/Strategy coordinators.
- Phase 2 (Completed): Migrated Telemetry coordinator, wired `TradingBot` and `LifecycleManager` to the registry, preserved facades.
- Phase 3 (Planned): Document deprecation timeline for facade APIs; evaluate moving remaining orchestration helpers (e.g., `SystemMonitor`) into coordinators once stable.

## References
- `src/gpt_trader/features/live_trade/engines/base.py` (CoordinatorContext, BaseEngine)
- `src/gpt_trader/features/live_trade/engines/runtime/coordinator.py`
- `src/gpt_trader/features/live_trade/engines/strategy.py`
- `src/gpt_trader/orchestration/*` *(removed)*
