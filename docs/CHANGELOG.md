# Changelog

---
status: current
last-updated: 2026-01-23
---

## Unreleased

### Global State Cleanup
- **BREAKING**: Health server helpers now require explicit `HealthState` injection; the module-level fallback was removed.
- **BREAKING**: Secrets manager no longer exposes module-level accessors; use `ApplicationContainer.secrets_manager` or instantiate `SecretsManager`.
- Data layer now uses explicit `DataService` instances; `initialize_data_layer` returns a service instead of mutating module globals.

### Coinbase REST Legacy Mode Removal
- **BREAKING**: Coinbase REST now requires `PositionStateStore` injection; legacy `_positions` fallbacks have been removed.
- Service/tests now read and write positions exclusively through the shared store.
- **BREAKING**: Removed `CoinbaseRestServiceBase`; use `CoinbaseRestServiceCore` directly.

### Logging Helper Rename
- **BREAKING**: Removed `gpt_trader.logging.orchestration_helpers`; use `runtime_helpers.get_runtime_logger`.

### TUI Legacy Fallback Cleanup
- Removed the legacy `config/tui_preferences.json` fallback; TUI preferences now read/write only the runtime path (or `GPT_TRADER_TUI_PREFERENCES_PATH`).
- Dropped legacy status class aliases (`good`/`bad`/`risk-status-*`) and Log widget CSS; widgets now use canonical `status-*` classes.
- TUI validation events now use `FieldValidationError` directly (the `ValidationError` alias was removed).

### LiveExecutionEngine Removal
- **BREAKING**: Removed `LiveExecutionEngine` and its container factory; use `TradingEngine.submit_order()` and guard-stack helpers instead.
- Migrated execution tests/integration flows to TradingEngine and updated docs to reflect the canonical path.

### Risk Config Schema Hardening
- `RiskConfig.from_env` now funnels through a `RiskConfigModel` Pydantic schema, using `RuntimeSettings.snapshot_env` to keep defaults and aliases in lockstep with runtime settings.
- Environment parsing raises `EnvVarError` with the offending var and logs the failure; JSON inputs emit precise `ValidationError`s when mappings or percentages are malformed.
- Regression tests snapshot the env key list, cover legacy aliases, and assert percentage bounds; operator docs now describe the stricter validation guarantees.

### Naming Alignment: `qty` → `quantity`
- Core brokerage interfaces now expose `quantity` exclusively; legacy `qty` aliases have been removed across serializers and dataclasses.
- Coinbase adapter, live execution, deterministic broker stub, and strategy paths emit `quantity` only in logs and telemetry to keep downstream metrics consistent.
- CLI order tooling only accepts `--order-quantity`; the legacy `--order-qty` alias has been removed.
- Historical rollout details live in repository history (Wave 1 status notes).

### March 2025 Test Harmonization
- Removed the legacy `scripts/run_spot_profile.py` and `scripts/sweep_strategies.py` shims; docs now reference the maintained backtest entry points directly.
- Removed the behavioral utilities walkthrough; see `docs/testing.md` for current testing guidance.
- Audited `uses_mock_broker` suites and documented opt-in usage while legacy market-impact hooks remain pending rebuild.
- Retired `tests/unit/gpt_trader/test_removed_aliases.py`; compatibility status now lives in `docs/DEPRECATIONS.md`.

### Module Cleanup & Broker Modernization
- **BREAKING**: Removed deprecated modules (`execution_v3`, `week2_filters`, `perps_baseline_v2`, legacy Coinbase helper modules).
- **BREAKING**: Migrated from `MockBroker` to `DeterministicBroker` for development/testing workflows.
- **FIX**: Resolved CLI import failures and enforced module removal consistency via `test_removed_aliases.py`.
- **DOCS**: Updated architecture, development guides, and paper trading documentation to reflect current patterns.
- **TESTS**: Consolidated test structure, removed stale files, and eliminated legacy testing patterns.
- All active test suites pass with the modernized broker implementation and clean module structure.

### Container Requirements
- **BREAKING**: `get_failure_tracker()` now requires an application container; fallback tracker has been removed.

### Profile/Config Cleanup
- **BREAKING**: `BotConfig.from_dict` no longer maps legacy profile keys; use profile schema or BotConfig schema directly.
- **BREAKING**: CLI no longer falls back to raw profile YAML for unknown profile names; use `--config`.
- Profile schema removed `daily_loss_limit` (absolute); use `daily_loss_limit_pct`.

### Monitoring Alert Cleanup
- Removed legacy `Alert.id` and `Alert.timestamp` aliases; use `alert_id` and `created_at`.

### Docs Cleanup
- Removed legacy risk templates from the tree; use git history for reference.
