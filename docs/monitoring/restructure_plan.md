# Monitoring and Type System Consolidation Plan

## Monitoring Modules

1. **Inventory**
   - `src/gpt_trader/monitoring/system/`
   - `src/gpt_trader/monitoring/interfaces.py`
   - `src/gpt_trader/orchestration/system_monitor.py`
   - `src/gpt_trader/orchestration/market_monitor.py`
   - `src/gpt_trader/monitoring/domain/perps/{margin, liquidation}.py`
2. **Next Steps**
   - Expand `src/gpt_trader/monitoring/domain/` for exchange or strategy specific monitors.
   - Continue consolidating platform health tooling under `src/gpt_trader/monitoring/system/`.
   - Iterate on shared interfaces (`Monitor`, `MonitorEvent`, `MonitorConfig`) within `src/gpt_trader/monitoring/interfaces.py`.
   - Co-locate shared alerting helpers (currently in orchestration) under `src/gpt_trader/monitoring/tools/`.
   - Update import sites in orchestration and live_trade to consume the new interfaces and domain modules.
   - Align tests by mirroring structure under `tests/unit/gpt_trader/monitoring/` with subdirectories for `domain`, `system`, and `tools`.

## Type System

1. **Target Structure**
   - `src/gpt_trader/types/common.py`
   - `src/gpt_trader/types/trading.py`
   - `src/gpt_trader/types/monitoring.py`
   - Feature-specific `types.py` remain in place when purpose-built.
2. **Implementation Notes**
   - Start by moving duplicate dataclasses (orders, positions, metrics) into `common.py` and `trading.py`.
   - Replace local imports (e.g., legacy `backtest` types from the bundle) with shared types where definitions match. Legacy sources are documented in `docs/archive/legacy_recovery.md`.
   - Document migration in module docstrings to aid reviewers.
   - Provide compatibility aliases during transition to avoid broad refactors in a single PR.
3. **Testing Strategy**
   - Introduce `tests/unit/gpt_trader/types/test_common_types.py` for regression coverage.
   - Update mypy configuration to reference the new shared modules.

## Utilities Follow-up

- Continue auditing for duplicated helpers (e.g., decimal math, datetime parsing) and fold them into thematic modules under `src/gpt_trader/utilities/`.
- Add lightweight unit tests alongside each new utility module to keep parity with source layout.
