# Changelog

## Unreleased

### October 2025 - Major Repository Cleanup & Modular Refactoring
- **Refactoring**: Split monolithic `cli.py` into modular structure (`cli/commands/`, `cli/handlers/`, `cli/parser.py`)
- **Refactoring**: Decomposed `monitoring/runtime_guards.py` into modular package (base, builtins, manager)
- **Refactoring**: Reorganized state management into `state/backup/services/` and `state/utils/`
- **Refactoring**: Extracted live trading helpers (`dynamic_sizing_helper.py`, `stop_trigger_manager.py`)
- **Cleanup**: Removed legacy live_trade facade layer (adapters, broker_connection, brokers, execution, live_trade)
- **Cleanup**: Removed legacy monitoring modules (`alerting_system.py`, monolithic `runtime_guards.py`)
- **Cleanup**: Deleted 12 outdated documentation files from `docs/archive/`
- **Tests**: Added comprehensive test suites (+33K lines): data providers, adaptive portfolio, position sizing, paper trading, orchestration, state management
- **Tests**: Added 121 test files with 335 test classes covering previously untested modules
- **Docs**: Updated ARCHITECTURE.md and DASHBOARD_GUIDE.md to reflect new modular structure
- **Docs**: Updated TRAINING_GUIDE.md to remove legacy facade references
- **CI**: Applied black formatting and removed trailing whitespace across codebase
- **Config**: Updated .gitignore to exclude entire `archived/` directory
- Net impact: 172 files changed, +33K insertions, -5.5K deletions

### Naming Alignment: `qty` → `quantity`
- Core brokerage interfaces now expose `quantity` exclusively; legacy `qty` aliases have been removed across serializers and dataclasses.
- Coinbase adapter, live execution, deterministic broker stub, and strategy paths emit `quantity` only in logs and telemetry to keep downstream metrics consistent.
- CLI order tooling only accepts `--order-quantity`; the legacy `--order-qty` alias has been removed.
- Historical rollout details live in repository history (Wave 1 status notes).

### March 2025 Test Harmonization
- Removed the legacy `scripts/run_spot_profile.py` and `scripts/sweep_strategies.py` shims; docs now reference the maintained backtest entry points directly.
- Moved the behavioral utilities walkthrough to `docs/testing/behavioral_scenarios_demo.md` and dropped the broad CI demo test.
- Audited `uses_mock_broker` suites and documented opt-in usage while legacy market-impact hooks remain pending rebuild.
- Retired `tests/unit/bot_v2/test_removed_aliases.py`; compatibility status now lives in `tests/fixtures/DEPRECATED.md`.

### Module Cleanup & Broker Modernization
- **BREAKING**: Removed deprecated modules (`execution_v3`, `week2_filters`, `perps_baseline_v2`, legacy Coinbase helper modules).
- **BREAKING**: Migrated from `MockBroker` to `DeterministicBroker` for development/testing workflows.
- **FIX**: Resolved CLI import failures and enforced module removal consistency via `test_removed_aliases.py`.
- **DOCS**: Updated architecture, development guides, and paper trading documentation to reflect current patterns.
- **TESTS**: Consolidated test structure, removed stale files, and eliminated legacy testing patterns.
- All active test suites pass with the modernized broker implementation and clean module structure.
