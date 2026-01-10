# Deprecation Tracker

This document tracks deprecated modules, shims, and their planned removal dates.

## Active Deprecations

### Configuration (Remove after v4.0)

| Deprecated | Canonical | Status |
|------------|-----------|--------|
| `PERPS_FORCE_MOCK` env var | `MOCK_BROKER` | Warning emitted |
| `SYMBOLS` env var | `TRADING_SYMBOLS` | Silent fallback |
| `POSITION_FRACTION` env var | `RISK_POSITION_FRACTION` | Silent fallback |
| `OrderRouter.execute()` | `OrderRouter.execute_async()` | Method deprecated |

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
| `gpt_trader.orchestration` package | v3.0 | Use canonical paths: `app.*`, `features.live_trade.*` |
| `orchestration.configuration.bot_config` | v3.0 | `gpt_trader.app.config.BotConfig` |
| `orchestration.configuration.risk.model` | v3.0 | `gpt_trader.features.live_trade.risk.config.RiskConfig` |
| `orchestration.live_execution.LiveExecutionEngine` | v3.0 | TradingEngine guard stack (`_validate_and_place_order` live loop; `submit_order` external) |
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
