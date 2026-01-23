# Deprecation Tracker

---
status: current
last-updated: 2026-01-23
---

This document tracks deprecated modules, shims, and their planned removal dates.

## Active Deprecations

All active deprecations must be listed here with an owner and a removal date.
CI enforces this registry for new deprecation shims.

| Deprecated | Replacement | Location | Owner | Removal Date | Notes |
|------------|-------------|----------|-------|--------------|-------|
No active deprecations.

## Legacy Inventory (Status)

Classification of legacy shims/fallbacks and compatibility keepers.

| Item | Location | Owner | Status | Notes |
|------|----------|-------|--------|-------|
| `BotConfig.from_dict` legacy profile-style YAML mapping | `src/gpt_trader/app/config/bot_config.py` | Core Config | deprecate | Emits `DeprecationWarning`; target removal in v4.0. |
| `EventStore.events` list + `EventStore.path` JSONL alias | `src/gpt_trader/persistence/event_store.py` | Persistence | deprecate | Internal call sites now use `list_events()`/`root`; legacy properties emit deprecation warnings. |
| `RiskConfig.daily_loss_limit` (absolute dollars) | `src/gpt_trader/features/live_trade/risk/config.py` | Risk | evaluate | Legacy absolute-dollar limit; prefer `daily_loss_limit_pct`. |
| TUI legacy guard shapes (`active_guards`) | `src/gpt_trader/tui/state.py`, `src/gpt_trader/monitoring/status_reporter.py` | TUI | deprecate | `active_guards` removed from TUI state/output; legacy inputs are normalized into `guards`. |

### Configuration (Remove after v4.0)

No active configuration deprecations.

### Legacy Backward Compatibility (Keep indefinitely)

These are kept for external consumer compatibility and are not scheduled for removal:

| Item | Location | Owner | Purpose |
|------|----------|-------|---------|
| Legacy API key detection | `src/gpt_trader/tui/services/credential_validator.py` | TUI | Coinbase legacy key format support |
| Legacy order payload builders | `src/gpt_trader/features/brokerages/coinbase/rest/base.py` | Brokerage | `_build_order_payload()`, `_execute_order_payload()` |

## General Removal Checklist

Before removing any deprecated item:

1. Search for internal usage: `grep -r "deprecated_name" src/ tests/`
2. Check dynamic imports: Review `importlib` and entry points
3. Update this registry with owner + removal date
4. Update docs that reference the deprecated path
5. Add migration notes to CHANGELOG.md

## Recently Removed

| Item | Removed In | Migration Path |
|------|------------|----------------|
| `BotConfig` flat compat accessors (`short_ma`, `long_ma`, etc.) | Unreleased | Use nested config: `config.strategy.*` and `config.risk.*`. |
| `gpt_trader.logging.orchestration_helpers` | Unreleased | Use `gpt_trader.logging.runtime_helpers`. |
| `StatusReporter.update_interval` + `StatusReporter.get_status_dict()` | Unreleased | Use `file_write_interval` + `get_status()`. |
| `get_failure_tracker()` fallback without container | Unreleased | Set application container; use `container.validation_failure_tracker`. |
| `Alert.id` / `Alert.timestamp` aliases | Unreleased | Use `alert_id` / `created_at`. |
| `CoinbaseRestServiceBase` alias | Unreleased | Use `CoinbaseRestServiceCore`. |
| `daily_loss_limit` in profile schema | Unreleased | Use `daily_loss_limit_pct`. |
| Perps strategy compat aliases (`short_ma`/`long_ma` props, `StrategyConfig`) | Unreleased | Use `PerpsStrategyConfig` and `short_ma_period`/`long_ma_period`. |
| CLI fallback for unknown profile YAML | Unreleased | Use a `Profile` enum value or `--config`. |
| Legacy risk templates | Unreleased | Removed from tree; use git history for reference. |
| Module-level `health_state` | Unreleased | Use `ApplicationContainer.health_state` and pass `HealthState` explicitly to health helpers. |
| Module-level `secrets_manager` helpers | Unreleased | Use `ApplicationContainer.secrets_manager` or instantiate `SecretsManager`. |
| Data module singletons (`store_data`/`fetch_data` functions) | Unreleased | Use `DataService` and its instance methods. |
| Yahoo data source stub (`DataSource.YAHOO`, `download_from_yahoo`) | Unreleased | Use `DataSource.COINBASE` and `download_from_coinbase`. |
| TUI legacy preferences fallback (`config/tui_preferences.json`) | Unreleased | Use runtime preferences path or `GPT_TRADER_TUI_PREFERENCES_PATH` |
| TUI legacy status CSS aliases (`good`/`bad`/`risk-status-*`) | Unreleased | Use `status-ok`, `status-warning`, `status-critical` |
| TUI validation `ValidationError` alias | Unreleased | Use `FieldValidationError` |
| `COINBASE_ENABLE_DERIVATIVES` env var alias | Unreleased | Use `COINBASE_ENABLE_INTX_PERPS` |
| Coinbase REST legacy position dict fallback | v4.0 | Require `PositionStateStore` injection |
| `PERPS_FORCE_MOCK` env var | v4.0 | Use `MOCK_BROKER` |
| `SYMBOLS` env var | v4.0 | Use `TRADING_SYMBOLS` |
| `POSITION_FRACTION` env var | v4.0 | Use `RISK_MAX_POSITION_PCT_PER_SYMBOL` |
| `OrderRouter.execute()` | v4.0 | Use `OrderRouter.execute_async()` |
| `gpt_trader.orchestration` package | DI migration | Use canonical paths: `app.*`, `features.live_trade.*` |
| `orchestration.configuration.bot_config` | DI migration | `gpt_trader.app.config.BotConfig` |
| `orchestration.configuration.risk.model` | DI migration | `gpt_trader.features.live_trade.risk.config.RiskConfig` |
| `features.live_trade.execution.LiveExecutionEngine` | v4.0 | TradingEngine guard stack (`_validate_and_place_order` live loop; `submit_order` external) |
| `orchestration.execution.*` | DI migration | `gpt_trader.features.live_trade.execution.*` |
| `orchestration.symbols` | DI migration | `gpt_trader.features.live_trade.symbols` |

### DI Migration Removal Summary (Phase 13)

The entire `src/gpt_trader/orchestration/` package was removed during the DI migration. Migration phases:

- **Phase 9-10**: Converted orchestration modules to pure re-export shims
- **Phase 11**: Moved tests to canonical locations, created deprecation test bucket
- **Phase 12**: Updated scripts/ci tooling, extended CI guards
- **Phase 13**: Hard removal of orchestration package and shim tests
- **Phase 14**: Post-removal cleanup (pytest marker, test hygiene allowlist, inventory generator)

---
