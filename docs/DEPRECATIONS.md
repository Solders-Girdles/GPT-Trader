# Deprecation Tracker

This document tracks deprecated modules, shims, and their planned removal dates.

All deprecated modules reference this tracker in their docstrings.

## Active Deprecations

### Execution Path (Remove in v3.0)

| Deprecated Path | Canonical Path | Status | Source |
|-----------------|----------------|--------|--------|
| `orchestration.live_execution.LiveExecutionEngine` | `TradingEngine.submit_order()` | Shim with warning | [live_execution.py](../src/gpt_trader/orchestration/live_execution.py) |
| `orchestration.execution.degradation` | `features.live_trade.degradation` | Re-export shim | [degradation.py](../src/gpt_trader/orchestration/execution/degradation.py) |
| `orchestration.configuration.risk.model` | `features.live_trade.risk.config` | Re-export shim | [model.py](../src/gpt_trader/orchestration/configuration/risk/model.py) |
| `orchestration.configuration.bot_config` | `app.config` | Re-export shim | [bot_config.py](../src/gpt_trader/orchestration/configuration/bot_config/bot_config.py) |
| `OrderRouter.execute()` | `OrderRouter.execute_async()` | Method deprecated | [router.py](../src/gpt_trader/features/live_trade/execution/router.py) |

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

## Removal Checklist

Before removing a deprecated item:

1. Search for internal usage: `grep -r "deprecated_name" src/ tests/`
2. Check dynamic imports: Review `importlib` and entry points
3. Update docs that reference the deprecated path
4. Add migration notes to CHANGELOG.md

## Recently Removed

| Item | Removed In | Migration Path |
|------|------------|----------------|
| *(none yet)* | | |

---

*Last updated: 2026-01-07*
