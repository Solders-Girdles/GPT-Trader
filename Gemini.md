# Gemini.md

Reference sheet for Gemini agents assisting with GPT-Trader V2.

## Quick Facts
- **Live trading:** Coinbase **spot** only. Perpetuals logic is checked in but requires Coinbase INTX access; use it for analysis, not production execution.
- **Main CLI:** `poetry run perps-bot --profile dev --dev-fast`. Stage 3 runner delegates to this command.
- **Core files:**
  - `src/bot_v2/orchestration/perps_bot.py` – orchestrator + risk guards + telemetry.
  - `src/bot_v2/cli.py` – profiles, account snapshot, treasury helpers.
  - `src/bot_v2/features/brokerages/coinbase/` – adapter, account manager, specs.
  - `src/bot_v2/orchestration/live_execution.py` – runtime safety checks.
  - `scripts/monitoring/export_metrics.py` – Prometheus/JSON exporter.

## Commands to Know
```bash
poetry install                                   # setup
poetry run perps-bot --profile dev --dev-fast    # spot dev run
poetry run perps-bot --account-snapshot          # dump Coinbase limits/fees
poetry run perps-bot --convert USD:USDC:1000     # convert helper
poetry run perps-bot --move-funds a:b:50         # treasury helper
poetry run python scripts/monitoring/export_metrics.py --metrics-file data/perps_bot/prod/metrics.json
poetry run pytest --collect-only                 # current test discovery
```

## Spot vs Perps Guidance
- Spot profiles normalize symbols to spot markets and fall back to the mock broker when derivatives are disabled.
- When touching perps code paths, mention the INTX requirement and keep functionality behind `COINBASE_ENABLE_DERIVATIVES` checks.

## Experimental Modules
These directories are tagged `__experimental__` and sit outside the production path:
- `src/bot_v2/features/backtest/`
- `src/bot_v2/features/ml_strategy/`
- `src/bot_v2/features/market_regime/`
- `src/bot_v2/monitoring/monitoring_dashboard.py`
- `src/bot_v2/workflows/`
Only modify them when explicitly instructed.

## Testing Notes
- `pytest --collect-only` yields 478 tests (420 selected after deselection). Run `poetry install` to pull in the latest dependencies (including `pyotp`) before running the suite.
- Add tests whenever adjusting risk guards, telemetry, or CLI surface area.

## Operational Checklist
1. Ensure README + `docs/ARCHITECTURE.md` match code changes.
2. Update `Agents.md`, `CLAUDE.md`, and this file with any new workflows.
3. Document spot/perps impacts and note the INTX gate for perps.
4. Confirm metrics serialization (`metrics.json`) still works if you add fields.

## Working Style Tips
- Use `rg`/`fd` for quick navigation (`rg --files -g '*.py' src/bot_v2` shows scope).
- Keep changeset responses explicit about testing (call out the `pyotp` caveat).
- Mention safety mechanisms when altering execution or risk code.
- Coordinate doc updates alongside code to keep agent knowledge in sync.
