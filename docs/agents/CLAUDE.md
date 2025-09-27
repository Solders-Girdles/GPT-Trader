# CLAUDE.md

Guidance for Claude Code agents working inside GPT-Trader V2. Read this before planning or editing.

## Project Snapshot
- **Live mode:** Coinbase **spot** trading. Perpetual futures logic remains compiled but Coinbase now gates INTX access; treat perps paths as future-ready only.
- **Primary entry point:** `poetry run perps-bot --profile dev --dev-fast` (Stage 3 runner simply forwards here).
- **Architecture:** Vertical slices under `src/bot_v2/features/`, orchestrated by `src/bot_v2/orchestration/perps_bot.py`. The codebase has grown beyond the original 500-token slices—expect cross-file work.

## Core Commands
```bash
# Install / update environment
poetry install

# Spot trading (mock fills, dev profile)
poetry run perps-bot --profile dev --dev-fast

# Canary/production dry-run (spot)
poetry run perps-bot --profile canary --dry-run

# Account telemetry snapshot
poetry run perps-bot --account-snapshot

# Treasury helpers
poetry run perps-bot --convert USD:USDC:1000
poetry run perps-bot --move-funds portfolio_a:portfolio_b:50

# Metrics exporter (Prometheus + JSON)
poetry run python scripts/monitoring/export_metrics.py --metrics-file var/data/perps_bot/prod/metrics.json

# Tests (see notes below)
poetry run pytest --collect-only
```

## Production Components (Spot)
| Area | Role |
|------|------|
| `src/bot_v2/orchestration/perps_bot.py` | Main loop, risk guards, telemetry, per-symbol decisions, INTX checks. |
| `src/bot_v2/cli.py` | Profiles, flag parsing, account snapshot/treasury commands. |
| `src/bot_v2/features/brokerages/coinbase/adapter.py` | Coinbase Advanced Trade integration (spot + dormant perps). |
| `src/bot_v2/features/brokerages/coinbase/account_manager.py` | Fee/limit snapshots, convert, move funds. |
| `src/bot_v2/orchestration/live_execution.py` | Runtime safety rails (PnL caps, liquidation buffer, volatility CB, correlation checks). |
| `scripts/monitoring/export_metrics.py` | Prometheus/JSON metrics service.

## Perps & Experimental Modules
- Perps code paths should stay intact but **never assume live connectivity**. Document INTX gating in every perps-related change.
- Modules flagged `__experimental__ = True` (backtest, ml_strategy, market_regime, monitoring dashboard) are off the production path. The old workflow engine was removed; reach back in git history if you truly need it. Touch these slices only when requested and keep optional dependencies isolated.

## Testing & Dependencies
- `poetry run pytest --collect-only` currently discovers 455 tests (446 selected after deselection). Run `poetry install` after pulling so new dependencies (e.g., `pyotp`) are available for the suite.
- Add regression tests for any changes to risk guards, telemetry, or CLI helpers.
- Legacy integration suites were retired; add any new cross-slice checks under `tests/unit/bot_v2/` (or build focussed fixtures alongside the components you touch).

## Safety & Operational Checks
- Risk controls live in `LiveExecutionEngine`: daily loss guard, liquidation distance, mark staleness, volatility circuit breaker, correlation risk.
- Account telemetry runs asynchronously; if you add new metrics, ensure `metrics.json` serialization remains stable.
- The bot defaults to mock trading when derivatives are disabled. Validate `COINBASE_ENABLE_DERIVATIVES` before enabling features that rely on perps endpoints.

## Documentation Expectations
Whenever your change affects behavior:
1. Update `README.md` (commands, quick start).
2. Update `docs/ARCHITECTURE.md` (system overview).
3. Sync `docs/agents/Agents.md`, this file, and `docs/agents/Gemini.md`.
4. Note the INTX dependency anywhere perps functionality is mentioned.

## Workflow Tips for Claude
1. Start by reading the relevant slice README/tests; they stay authoritative for local logic.
2. Use `rg` for navigation—repo size makes `grep` slow.
3. Favor incremental commits and keep doc updates with code changes when practical.
4. Highlight risk impacts and testing instructions in your responses to users.
