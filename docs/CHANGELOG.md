# Changelog

## Unreleased

### Coinbase REST Legacy Mode Removal
- **BREAKING**: Coinbase REST now requires `PositionStateStore` injection; legacy `_positions` fallbacks have been removed.
- Service/tests now read and write positions exclusively through the shared store.

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
- Removed the behavioral utilities walkthrough; see `docs/guides/testing.md` for current testing guidance.
- Audited `uses_mock_broker` suites and documented opt-in usage while legacy market-impact hooks remain pending rebuild.
- Retired `tests/unit/gpt_trader/test_removed_aliases.py`; compatibility status now lives in `docs/DEPRECATIONS.md`.

### Module Cleanup & Broker Modernization
- **BREAKING**: Removed deprecated modules (`execution_v3`, `week2_filters`, `perps_baseline_v2`, legacy Coinbase helper modules).
- **BREAKING**: Migrated from `MockBroker` to `DeterministicBroker` for development/testing workflows.
- **FIX**: Resolved CLI import failures and enforced module removal consistency via `test_removed_aliases.py`.
- **DOCS**: Updated architecture, development guides, and paper trading documentation to reflect current patterns.
- **TESTS**: Consolidated test structure, removed stale files, and eliminated legacy testing patterns.
- All active test suites pass with the modernized broker implementation and clean module structure.
