# Monitoring and Type System Consolidation Plan

## Monitoring Modules

1. **Inventory**
   - `src/bot_v2/monitoring/system/`
   - `src/bot_v2/monitoring/interfaces.py`
   - `src/bot_v2/orchestration/system_monitor.py`
   - `src/bot_v2/orchestration/market_monitor.py`
   - `src/bot_v2/monitoring/domain/perps/{margin, liquidation}.py`
2. **Next Steps**
   - Expand `src/bot_v2/monitoring/domain/` for exchange or strategy specific monitors.
   - Continue consolidating platform health tooling under `src/bot_v2/monitoring/system/`.
   - Iterate on shared interfaces (`Monitor`, `MonitorEvent`, `MonitorConfig`) within `src/bot_v2/monitoring/interfaces.py`.
   - Co-locate shared alerting helpers (currently in orchestration) under `src/bot_v2/monitoring/tools/`.
   - Update import sites in orchestration and live_trade to consume the new interfaces and domain modules.
   - Align tests by mirroring structure under `tests/unit/bot_v2/monitoring/` with subdirectories for `domain`, `system`, and `tools`.

## Type System

1. **Target Structure**
   - `src/bot_v2/types/common.py`
   - `src/bot_v2/types/trading.py`
   - `src/bot_v2/types/monitoring.py`
   - Feature-specific `types.py` remain in place when purpose-built.
2. **Implementation Notes**
   - Start by moving duplicate dataclasses (orders, positions, metrics) into `common.py` and `trading.py`.
   - Replace local imports (e.g., `features/backtest/types.py`) with shared types where definitions match.
   - Document migration in module docstrings to aid reviewers.
   - Provide compatibility aliases during transition to avoid broad refactors in a single PR.
3. **Testing Strategy**
   - Introduce `tests/unit/bot_v2/types/test_common_types.py` for regression coverage.
   - Update mypy configuration to reference the new shared modules.

## Utilities Follow-up

- Continue auditing for duplicated helpers (e.g., decimal math, datetime parsing) and fold them into thematic modules under `src/bot_v2/utilities/`.
- Add lightweight unit tests alongside each new utility module to keep parity with source layout.
