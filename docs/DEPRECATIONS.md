# Deprecation Tracker

This document tracks deprecated modules, shims, and their planned removal dates.

## Active Deprecations

### Logging Helpers (Remove after v4.1)

| Deprecated | Replacement | Notes |
|------------|-------------|-------|
| `logging.orchestration_helpers.get_orchestration_logger` | `logging.runtime_helpers.get_runtime_logger` | Shim emits DeprecationWarning |

### Configuration (Remove after v4.0)

No active configuration deprecations.

### Legacy Backward Compatibility (Keep indefinitely)

These are kept for external consumer compatibility and are not scheduled for removal:

| Item | Location | Purpose |
|------|----------|---------|
| Module-level `health_state` | `app.health_server` | Legacy access pattern |
| Module-level `secrets_manager` | `security.secrets_manager` | Legacy access pattern |
| Legacy API key detection | `tui.services.credential_validator` | Coinbase legacy key format support |
| Legacy order payload builders | `features.brokerages.coinbase.rest.base` | `_build_order_payload()`, `_execute_order_payload()` |

## General Removal Checklist

Before removing any deprecated item:

1. Search for internal usage: `grep -r "deprecated_name" src/ tests/`
2. Check dynamic imports: Review `importlib` and entry points
3. Update docs that reference the deprecated path
4. Add migration notes to CHANGELOG.md

## Recently Removed

| Item | Removed In | Migration Path |
|------|------------|----------------|
| Coinbase REST legacy position dict fallback | v4.0 | Require `PositionStateStore` injection |
| `PERPS_FORCE_MOCK` env var | v4.0 | Use `MOCK_BROKER` |
| `SYMBOLS` env var | v4.0 | Use `TRADING_SYMBOLS` |
| `POSITION_FRACTION` env var | v4.0 | Use `RISK_MAX_POSITION_PCT_PER_SYMBOL` |
| `OrderRouter.execute()` | v4.0 | Use `OrderRouter.execute_async()` |
| `gpt_trader.orchestration` package | v3.0 | Use canonical paths: `app.*`, `features.live_trade.*` |
| `orchestration.configuration.bot_config` | v3.0 | `gpt_trader.app.config.BotConfig` |
| `orchestration.configuration.risk.model` | v3.0 | `gpt_trader.features.live_trade.risk.config.RiskConfig` |
| `features.live_trade.execution.LiveExecutionEngine` | v4.0 | TradingEngine guard stack (`_validate_and_place_order` live loop; `submit_order` external) |
| `orchestration.execution.*` | v3.0 | `gpt_trader.features.live_trade.execution.*` |
| `orchestration.symbols` | v3.0 | `gpt_trader.features.live_trade.symbols` |

### v3.0 Removal Summary (Phase 13)

The entire `src/gpt_trader/orchestration/` package was removed. Migration phases:

- **Phase 9-10**: Converted orchestration modules to pure re-export shims
- **Phase 11**: Moved tests to canonical locations, created deprecation test bucket
- **Phase 12**: Updated scripts/ci tooling, extended CI guards
- **Phase 13**: Hard removal of orchestration package and shim tests
- **Phase 14**: Post-removal cleanup (pytest marker, test hygiene allowlist, inventory generator)

---

*Last updated: 2026-01-09*
