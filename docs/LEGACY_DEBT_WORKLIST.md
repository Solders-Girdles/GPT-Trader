# Legacy Debt Worklist

This is a running checklist of likely legacy remnants (naming, organization, compatibility shims, and stale artifacts) that can accumulate after architecture pivots and migrations.

Use this as a backlog: each item should either be **confirmed as intentionally supported** (keep), or **removed/rewritten** (pay down debt).

This is the canonical legacy-debt tracker; other reports (e.g. `docs/LEGACY_DEBT.md`) should be folded into this file to avoid drift.

## P0 — Broken / misleading today

- [x] Fix outdated composition-root example that imports the removed `gpt_trader.orchestration` package: `examples/composition_root_example.py`
- [x] Remove (or justify) tracked test-run artifacts that don’t belong in source control:
  - `coverage_output.txt`
  - `test_output.txt`
- [x] Resolve monitoring stack duplication and naming drift:
  - `monitoring/` stack uses `gpt-trader-*` container names
  - `scripts/monitoring/docker-compose.yml.example` uses legacy `coinbase_trader_*` names

## P1 — Versioning / naming drift (high confusion risk)

- [x] Fix Coinbase migration wording drift in `docs/reference/coinbase_complete.md` (legacy v2 vs Advanced Trade v3).
- [x] Clarify the project’s “V2 vs DI migration vs Coinbase API v3” vocabulary and align references:
  - Package/docstrings now avoid “GPT-Trader V2” branding (e.g. `src/gpt_trader/__init__.py`, `src/gpt_trader/errors/__init__.py`)
  - DI migration docs avoid internal semantic-version labeling (`docs/MIGRATION_STATUS.md`)
- [x] Update `pyproject.toml` metadata that still reads like an older “equities-first scaffold” description.
- [x] Retire the `coinbase-trader` CLI alias (`pyproject.toml`, docs).

## P2 — Backward-compatibility shims to evaluate (remove when safe)

- [ ] Config alias properties and “compat” accessors (remove once no longer needed):
  - `src/gpt_trader/app/config/bot_config.py` (e.g., `short_ma`, `long_ma`, etc.)
  - `src/gpt_trader/features/live_trade/strategies/perps_baseline/strategy.py` (e.g., `StrategyConfig = PerpsStrategyConfig`, `short_ma`, `long_ma`)
  - Status: internal call sites now prefer canonical `*.short_ma_period` / `*.long_ma_period` access; aliases remain for backward compatibility.
- [x] Persistence compatibility:
  - `src/gpt_trader/persistence/event_store.py` (`events` list property and `path` JSONL alias)
  - Status: internal call sites migrated to `list_events()`/`root`; legacy properties emit deprecation warnings.
- [x] Legacy import fallback in resource monitoring:
  - `src/gpt_trader/utilities/performance/resource.py` checks `sys.modules["gpt_trader.utilities.performance_monitoring"]`
- [x] Legacy / deprecated fields still represented in models or UI:
  - `src/gpt_trader/monitoring/status_reporter.py` (removed legacy accessors)
  - TUI now uses `guards` list only; `active_guards` is normalized during ingestion.
- [x] Legacy env var aliases still supported:
  - `COINBASE_ENABLE_DERIVATIVES` (alias for INTX/perps gating)
  - Status: removal horizon documented in `docs/DEPRECATIONS.md` and warnings issued when set.

## P2 — Legacy modules and facade drift

- [x] Remove confirmed-unused placeholder types module:
  - `src/gpt_trader/types/trading.py`
  - `src/gpt_trader/types/`
  - `tests/unit/gpt_trader/types/`
  - Status: only referenced by its unit tests; prefer `src/gpt_trader/core/` for domain types.
- [x] Standardize on the `gpt_trader.utilities.logging` facade (keep `gpt_trader.logging` for infrastructure only):
  - `src/gpt_trader/features/live_trade/orchestrator/orchestrator.py` (stop importing `log_execution_error`, `symbol_context`)
  - `src/gpt_trader/features/live_trade/orchestrator/decision.py` (stop importing `log_strategy_decision`)
  - `src/gpt_trader/features/live_trade/orchestrator/logging_utils.py` (stop importing `get_runtime_logger`)

## P2 — Partially removed integrations (choose: finish or delete)

- [ ] Data layer still models a Yahoo source but only stubs/redirects it:
  - `src/gpt_trader/features/data/types.py` (`DataSource.YAHOO`)
  - `src/gpt_trader/features/data/data.py` (`download_from_yahoo` stub)
  - Status: `DataSource.YAHOO` is now treated as a legacy alias for Coinbase in `DataService.fetch_data`; decide whether to complete `yfinance` or drop the Yahoo path.
  - decide whether to complete the optional `yfinance` integration or drop the Yahoo path entirely

## P3 — Low-impact cleanup candidates

- [x] Remove or justify unused equities-market constants:
  - `src/gpt_trader/config/constants.py` (removed `MARKET_OPEN_HOUR`, `MARKET_CLOSE_HOUR`)
- [x] Close remaining doc gaps so architecture docs match the current tree:
  - Add `src/gpt_trader/persistence/` to `README.md` and `docs/ARCHITECTURE.md`.
  - Clarify canonical `src/gpt_trader/backtesting/` and how `features/research` relates (wrapper/adapter).
- [ ] Clean up remaining legacy terminology drift:
  - Remove/clarify `ServiceRegistry` mentions in runtime comments (keep migration docs intentional).
  - Update stale “orchestration” wording in `docs/plans/TUI_IMPROVEMENT_PLAN.md`.
  - Verify `BotConfig.from_dict` no longer needs legacy profile-mapping schema support (or add explicit deprecation warnings + a removal plan).
  - Confirm `gpt_trader.logging.orchestration_helpers` is fully removed (code is gone; keep only migration notes in docs).
- [ ] Reconcile / prune stale entries in the deprecation tracker so it reflects reality:
  - `docs/DEPRECATIONS.md` (some “remove now” items appear already removed or migrated)
