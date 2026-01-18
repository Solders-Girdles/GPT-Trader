# Legacy Debt & Migration Status

This document identifies remnants of legacy systems, architectural transitions, and documentation gaps. It serves as a work-list for finalizing the V3 architecture migration.

## 1. Code to Delete or Migrate

| Path | Status | Recommendation |
|------|--------|----------------|
| `src/gpt_trader/types/trading.py` | **Legacy/Placeholder** | Delete file. Use `src/gpt_trader/core/` for domain types or move specialized types to `backtesting`. Confirmed strictly unused in runtime code (only unit tests). |
| `src/gpt_trader/types/` | **Empty Directory** | Delete directory after removing `trading.py`. |

## 2. Refactoring Candidates (Logging)

The goal is to standardize on `gpt_trader.utilities.logging` (facade) and keep `gpt_trader.logging` for infrastructure only.

| Location | Issue | Remediation |
|----------|-------|-------------|
| `src/gpt_trader/features/live_trade/orchestrator/orchestrator.py` | Direct import of `log_execution_error`, `symbol_context` from `gpt_trader.logging` | Update imports to use `gpt_trader.utilities.logging` facade or move helpers to `utilities`. |
| `src/gpt_trader/features/live_trade/orchestrator/decision.py` | Direct import of `log_strategy_decision` | Move `log_strategy_decision` to `utilities` or `features/live_trade/telemetry`. |
| `src/gpt_trader/features/live_trade/orchestrator/logging_utils.py` | Direct import of `get_runtime_logger` | Use `gpt_trader.utilities.logging.get_logger` or similar facade. |

## 3. Documentation Gaps & Clarifications

| Document | Section | Issue | Recommended Fix |
|----------|---------|-------|-----------------|
| `README.md` | Project Structure | Missing `src/gpt_trader/persistence/` | Add `persistence/` to the tree (canonical event/order store). |
| `docs/ARCHITECTURE.md` | Component Architecture | Missing `persistence` in directory trees | Add `persistence` to the cross-cutting packages list. |
| `docs/ARCHITECTURE.md` | General | Ambiguous `backtesting` location | Explicitly document `src/gpt_trader/backtesting/` as the canonical framework and clarify that `features/research` and `features/intelligence` are adapters/wrappers. |

## 4. Legacy Terminology cleanup

| Term | Context | Action |
|------|---------|--------|
| `ServiceRegistry` | Removed code | Ensure no string references remain in comments (checked: clean). |
| `orchestration` | Removed package | `docs/plans/TUI_IMPROVEMENT_PLAN.md` references `tui/managers/ui_coordinator.py` "Update orchestration". Verify if this plan is stale. |
| `Legacy Profile Mapping` | `BotConfig.from_dict` | `docs/DEPRECATIONS.md` lists this as "remove now". Check `src/gpt_trader/app/config/bot_config.py`. |
| `gpt_trader.logging.orchestration_helpers` | Code | `docs/DEPRECATIONS.md` lists this as "remove now". Verify deletion. |

## 5. Verification Checklist

- [ ] Delete `src/gpt_trader/types/trading.py` and remove associated tests in `tests/unit/gpt_trader/types/`.
- [ ] Migrate logging helpers in `live_trade/orchestrator` to the `utilities` facade.
- [ ] Add `persistence` to `README.md` and `ARCHITECTURE.md`.
- [ ] Add `backtesting` clarification to `ARCHITECTURE.md`.
- [ ] Verify `BotConfig` no longer contains legacy profile mapping logic.
