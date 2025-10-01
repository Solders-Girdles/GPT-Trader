# GPT-Trader Agent Guide

This is the shared orientation document for all AI agents working in this repository. Keep it open while you plan or execute tasks.

## 1. Current Mission Snapshot
- **Live focus:** Coinbase **spot** trading. Perpetual futures logic remains in the tree but real endpoints stay locked behind the Coinbase INTX gate (`COINBASE_ENABLE_DERIVATIVES` must be `1` *and* INTX access is required).
- **Primary entry point:** `poetry run perps-bot --profile dev --dev-fast`. The legacy Stage 3 wrapper has been removed.
- **Architecture style:** Vertical slices under `src/bot_v2/features/`, but the codebase has grown to 181 Python files—expect multi-file workflows instead of single 500-token modules.

## 2. Directory Compass
| Area | Purpose |
|------|---------|
| `src/bot_v2/orchestration/perps_bot.py` | Core orchestrator used for spot profiles; enforces risk guards, telemetry, and optional perps hooks. |
| `src/bot_v2/cli.py` | CLI wiring (profiles, account snapshots, treasury helpers). |
| `src/bot_v2/features/brokerages/coinbase/` | Coinbase adapter, account manager, telemetry helpers. |
| `src/bot_v2/features/brokerages/coinbase/account_manager.py` | Fee/limit snapshots plus treasury helpers (`convert`, `move-funds`). |
| `src/bot_v2/orchestration/live_execution.py` | Runtime safety rails (PnL caps, liquidation buffer, volatility CB, correlation checks). |
| `scripts/monitoring/export_metrics.py` | Prometheus/JSON metrics service for runtime telemetry. |
| `src/bot_v2/monitoring/` | Additional observability helpers (metrics serialisation, dashboards). |
| `docs/guides/paper_trading.md` | Deep dive on mock/paper trading workflows. |
| `docs/ARCHITECTURE.md` | High-level design doc—update alongside code changes. |
| `README.md` | Fast-install + day-to-day runbook. |

## 3. Core Commands

```bash
poetry install                                   # Set up or refresh dependencies
poetry run perps-bot --profile dev --dev-fast    # Spot trading (mock fills)
poetry run perps-bot --profile canary --dry-run  # Canary validation without live orders
poetry run perps-bot --account-snapshot          # Coinbase fee/limit snapshot
poetry run perps-bot --convert USD:USDC:1000     # Treasury helpers
poetry run perps-bot --move-funds a:b:50         # Treasury helpers
poetry run python scripts/monitoring/export_metrics.py --metrics-file var/data/perps_bot/prod/metrics.json
poetry run pytest --collect-only                 # Test discovery snapshot
poetry run pytest -q                             # Full unit suite
```

## 4. Trading Modes & Perps Status
1. **Spot (default)**
   - Profiles `dev`, `demo`, `prod`, `canary` auto-normalize symbols to spot markets.
   - `perps_bot` turns on the deterministic broker stub unless derivatives are explicitly enabled *and* credentials pass validation.
2. **Perps (future-ready)**
   - Keep the code paths compiling and tested, but call out the INTX dependency in any user-facing change.
   - Guard new work behind checks for `COINBASE_ENABLE_DERIVATIVES` to avoid surprise production enablement.

## 5. Experimental vs Production Slices
Treat these modules as **experimental** (documented with `__experimental__ = True`):
- `src/bot_v2/features/backtest/`
- `src/bot_v2/features/ml_strategy/`
- `src/bot_v2/features/market_regime/`
- `src/bot_v2/monitoring/monitoring_dashboard.py`

The retired workflow engine was removed; retrieve it from git history if needed. Only touch the remaining experimental slices when specifically asked. Everything else in `features/` is either production-critical or demo-supporting.

## 6. Operational Tooling
- **Account telemetry:** `poetry run perps-bot --account-snapshot` (dumps permissions, fee schedule, and limits).
- **Treasury helpers:**
  - `poetry run perps-bot --convert USD:USDC:1000`
  - `poetry run perps-bot --move-funds from_portfolio_uuid:to_portfolio_uuid:50`
- **Metrics:** `poetry run python scripts/monitoring/export_metrics.py --metrics-file var/data/perps_bot/prod/metrics.json` exposes `/metrics` (Prometheus) and `/metrics.json`.
- **Risk guards (runtime):** Daily PnL stops, liquidation-buffer checks, mark staleness, volatility circuit breakers, and correlation checks all live inside `LiveExecutionEngine`.

## 7. Testing Expectations
- **Command:** `poetry run pytest --collect-only` shows current test count and selection status (100% pass rate expected on active code).
- **Dependencies:** `pyotp` remains part of the base Poetry environment; run `poetry install` after pulling to ensure security tests pass.
- Keep unit tests under `tests/unit/bot_v2/` up to date, and add coverage for new risk or telemetry paths.

## 8. Common Workflows for Agents
1. **Feature work:**
   - Read the relevant slice README + tests.
   - Implement in the slice + orchestration glue.
   - Update docs (`README.md`, `docs/ARCHITECTURE.md`) if behavior shifts.
2. **Bugfix:**
   - Reproduce with the dev profile (`--dev-fast` is useful).
   - Add or adjust regression tests before patching.
3. **Documentation pass:**
   - Sync this guide, `docs/agents/CLAUDE.md`, and `docs/agents/Gemini.md` whenever the architecture or operations change.

## 9. Source of Truth Checklist
Whenever you ship a change, confirm:
- [ ] README reflects the new instructions.
- [ ] Architecture doc matches the live system.
- [ ] Tests either pass or document the dependency gap (e.g., `pyotp`).
- [ ] Agent guides (this file + per-agent files) stay consistent.

Stay explicit about spot vs perps mode, and note the INTX gate in every perps-related change description.

## 10. Agent-Specific Notes

- **Claude Code:** Lean on the planning tool for multi-slice edits, keep responses explicit about testing status, and highlight any risk-surface changes inline for reviewers.
- **Gemini Code Assistants:** Prefer concise diffs with command references (keep `rg`/`fd` usage visible) and call out environment prerequisites such as `poetry install` when suggesting test runs.
