# Deprecation Tracker

This document tracks deprecated modules, shims, and their planned removal dates.

All deprecated modules reference this tracker in their docstrings.

## Active Deprecations

### Execution Path (Remove in v3.0)

| Deprecated Path | Canonical Path | Status | Source |
|-----------------|----------------|--------|--------|
| `orchestration.live_execution.LiveExecutionEngine` | `TradingEngine.submit_order()` | Shim with warning | [live_execution.py](../src/gpt_trader/orchestration/live_execution.py) |
| `orchestration.execution.degradation` | `features.live_trade.degradation` | Re-export shim | [degradation.py](../src/gpt_trader/orchestration/execution/degradation.py) |
| `orchestration.execution.validation` | `features.live_trade.execution.validation` | Re-export shim | [validation.py](../src/gpt_trader/orchestration/execution/validation.py) |
| `orchestration.symbols` | `features.live_trade.symbols` | Re-export shim | [symbols.py](../src/gpt_trader/orchestration/symbols.py) |
| `orchestration.configuration.risk.model` | `features.live_trade.risk.config` | Re-export shim | [model.py](../src/gpt_trader/orchestration/configuration/risk/model.py) |
| `orchestration.configuration.bot_config` | `app.config` | Re-export shim | [bot_config.py](../src/gpt_trader/orchestration/configuration/bot_config/bot_config.py) |
| `orchestration.configuration` (ConfigValidationError) | `app.config.validation` | Re-export shim | [\_\_init\_\_.py](../src/gpt_trader/orchestration/configuration/__init__.py) |
| `orchestration.account_telemetry` | `features.live_trade.telemetry.account` | Re-export shim | [account_telemetry.py](../src/gpt_trader/orchestration/account_telemetry.py) |
| `orchestration.bootstrap` | `app.bootstrap` | Re-export shim | [bootstrap.py](../src/gpt_trader/orchestration/bootstrap.py) |
| `orchestration.spot_profile_service` | `features.live_trade.orchestrator.spot_profile_service` | Re-export shim | [spot_profile_service.py](../src/gpt_trader/orchestration/spot_profile_service.py) |
| `orchestration.intx_portfolio_service` | `features.brokerages.coinbase.intx_portfolio_service` | Re-export shim | [intx_portfolio_service.py](../src/gpt_trader/orchestration/intx_portfolio_service.py) |
| `orchestration.configuration.profiles` | `app.config.profile_loader` | Re-export shim | [profiles.py](../src/gpt_trader/orchestration/configuration/profiles.py) |
| `orchestration.configuration.bot_config.defaults` | `app.config.defaults` | Re-export shim | [defaults.py](../src/gpt_trader/orchestration/configuration/bot_config/defaults.py) |
| `orchestration.configuration.bot_config.rules` | `app.config.validation_rules` | Re-export shim | [rules.py](../src/gpt_trader/orchestration/configuration/bot_config/rules.py) |
| `orchestration.execution` (package) | `features.live_trade.execution` | Re-export shim | [\_\_init\_\_.py](../src/gpt_trader/orchestration/execution/__init__.py) |
| `orchestration` (package) | `app`, `features` | Re-export shim | [\_\_init\_\_.py](../src/gpt_trader/orchestration/__init__.py) |
| `OrderRouter.execute()` | `OrderRouter.execute_async()` | Method deprecated | [router.py](../src/gpt_trader/features/live_trade/execution/router.py) |

> **Note**: As of 2026-01-09, production code (`src/`) no longer imports from `gpt_trader.orchestration` outside the shim package. All orchestration modules are now pure re-export shims with `DeprecationWarning` on import.

### Configuration (Remove after v3.0)

| Deprecated | Canonical | Status |
|------------|-----------|--------|
| `PERPS_FORCE_MOCK` env var | `MOCK_BROKER` | Warning emitted | Fallback still works |
| `SYMBOLS` env var | `TRADING_SYMBOLS` | Silent fallback | Legacy support |
| `POSITION_FRACTION` env var | `RISK_POSITION_FRACTION` | Silent fallback | RISK_ prefix preferred |

### Legacy Backward Compatibility (Keep indefinitely)

These are kept for external consumer compatibility and are not scheduled for removal:

| Item | Location | Purpose |
|------|----------|---------|
| Module-level `health_state` | `app.health_server` | Legacy access pattern |
| Module-level `secrets_manager` | `security.secrets_manager` | Legacy access pattern |
| Legacy API key detection | `tui.services.credential_validator` | Coinbase legacy key format support |
| Legacy order payload builders | `features.brokerages.coinbase.rest.base` | `_build_order_payload()`, `_execute_order_payload()` |

## v3.0 Removal Checklist

When removing deprecated orchestration shims in v3.0:

### Modules to Delete

```
src/gpt_trader/orchestration/
├── __init__.py
├── account_telemetry.py
├── bootstrap.py
├── config_controller.py
├── configuration/
│   ├── __init__.py
│   ├── bot_config/
│   │   ├── __init__.py
│   │   ├── bot_config.py
│   │   ├── defaults.py
│   │   ├── rules.py
│   │   └── validator.py
│   ├── profile_loader.py
│   ├── profiles.py
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── logging_utils.py
│   │   └── model.py
│   └── validation.py
├── derivatives_discovery.py
├── derivatives_products.py
├── deterministic_broker.py
├── execution/
│   ├── __init__.py
│   ├── broker_executor.py
│   ├── degradation.py
│   ├── guards/
│   │   ├── __init__.py
│   │   ├── api_health.py
│   │   ├── daily_loss.py
│   │   ├── liquidation_buffer.py
│   │   └── volatility.py
│   ├── order_submission.py
│   ├── state_collection.py
│   └── validation.py
├── hybrid_paper_broker.py
├── intx_portfolio_service.py
├── live_execution.py
├── protocols.py
├── runtime_paths.py
├── spot_profile_service.py
├── strategy_orchestrator/
│   ├── __init__.py
│   ├── context.py
│   ├── logging_utils.py
│   ├── orchestrator.py
│   └── spot_filters.py
├── symbols.py
├── system_monitor_positions.py
└── trading_bot/
    ├── __init__.py
    └── bot.py
```

### Pre-Removal Verification

1. Run: `rg -n "gpt_trader\.orchestration" src | rg -v "src/gpt_trader/orchestration"` → should return empty
2. Run: `uv run pytest tests/unit -q` → all tests pass
3. Update CHANGELOG.md with migration notes
4. Update any external documentation

### Post-Removal Tasks

1. Delete `tests/unit/gpt_trader/orchestration/` directory
2. Remove `orchestration` from any `__all__` exports
3. Update `docs/ARCHITECTURE.md` to remove orchestration references
4. Update any CI/CD that references orchestration paths

## General Removal Checklist

Before removing any deprecated item:

1. Search for internal usage: `grep -r "deprecated_name" src/ tests/`
2. Check dynamic imports: Review `importlib` and entry points
3. Update docs that reference the deprecated path
4. Add migration notes to CHANGELOG.md

## Recently Removed

| Item | Removed In | Migration Path |
|------|------------|----------------|
| *(none yet)* | | |

---

*Last updated: 2026-01-08*
