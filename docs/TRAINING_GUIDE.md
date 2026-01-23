# GPT-Trader Training Guide (gpt_trader)

This guide orients new contributors to the current spot-first stack. It replaces
legacy onboarding material that referenced the deprecated `src/bot` package and
its ML workflows.

## Module 1 – System Overview

- **Architecture**: Review `docs/ARCHITECTURE.md` for the vertical slice layout
  under `src/gpt_trader/`.
- **Execution loop**: `TradingBot` + `TradingEngine` (in
  `features/live_trade/bot.py` and `features/live_trade/engines/strategy.py`)
  coordinate guards, orders, and telemetry.
- **Brokerage**: `features/brokerages/coinbase` contains REST/WS adapters for
  Coinbase Advanced Trade spot markets.

## Module 2 – Environment Setup

1. Install uv (if needed): `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Install project dependencies:

   ```bash
   uv sync
   ```

3. Copy environment template and adjust credentials when required:

   ```bash
   cp config/environments/.env.template .env
   ```

4. Run the dev smoke test:

   ```bash
   uv run gpt-trader run --profile dev --dev-fast
   ```

   This executes one control cycle using the mock broker.

## Module 3 – Feature Deep Dive

- **Position sizing**: `features/intelligence/sizing/position_sizer.py` houses
  Kelly-style sizing helpers used by research/backtesting.
- **Risk guards**: See `docs/RISK_INTEGRATION_GUIDE.md` for guard coverage and
  thresholds.
- **Telemetry**: Metrics persist to `runtime_data/<profile>/metrics.json` and
  `runtime_data/<profile>/events.db`; the exporter in `scripts/monitoring/export_metrics.py`
  reads events.db first and converts them to Prometheus.

## Module 4 – Development Workflow

1. Create a feature branch and implement changes in the relevant slice.
2. Add or update tests under `tests/unit/gpt_trader/`.
3. Run `uv run pytest -q` before opening a pull request.
4. Update documentation (`docs/README.md`, architecture/risk guides) alongside
   code changes.

## Module 5 – Operational Awareness

- Study `docs/operations/RUNBOOKS.md` and `docs/MONITORING_PLAYBOOK.md` for
  on-call expectations.
- **Monitoring**: Launch via `uv run gpt-trader tui` (add `--mode live/paper/demo` to skip the selector).
- Manual tooling:

  ```bash
  uv run gpt-trader account snapshot
  uv run gpt-trader treasury convert --from USD --to USDC --amount 250
  uv run gpt-trader treasury move --from-portfolio from_uuid --to-portfolio to_uuid --amount 25
  ```

- Guardrails default to spot-only. CFM futures (US) are opt-in via `TRADING_MODES=cfm` + `CFM_ENABLED=1`. INTX perps remain locked behind INTX access and `COINBASE_ENABLE_INTX_PERPS=1`.

## Module 6 – Further Reading

- Historical guides for the previous generation system were removed from the
  tree; refer to git history if you need them.
- Review agent-specific docs (`docs/guides/agents.md`) for collaboration patterns.
