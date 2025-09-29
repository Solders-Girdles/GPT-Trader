# GPT-Trader Training Guide (bot_v2)

This guide orients new contributors to the current spot-first stack. It replaces
legacy onboarding material that referenced the deprecated `src/bot` package and
its ML workflows.

## Module 1 – System Overview

- **Architecture**: Review `docs/ARCHITECTURE.md` for the vertical slice layout
  under `src/bot_v2/`.
- **Execution loop**: `LiveExecutionEngine` (in
  `features/live_trade/live_trade.py`) coordinates guards, orders, and telemetry.
- **Brokerage**: `features/brokerages/coinbase` contains REST/WS adapters for
  Coinbase Advanced Trade spot markets.

## Module 2 – Environment Setup

1. Install Poetry (if needed): `pipx install poetry`.
2. Install project dependencies:

   ```bash
   poetry install
   ```

3. Copy environment template and adjust credentials when required:

   ```bash
   cp config/environments/.env.template .env
   ```

4. Run the dev smoke test:

   ```bash
   poetry run perps-bot --profile dev --dev-fast
   ```

   This executes one control cycle using the mock broker.

## Module 3 – Feature Deep Dive

- **Position sizing**: `features/position_sizing/` implements Kelly-style
  sizing with risk caps.
- **Risk guards**: See `docs/RISK_INTEGRATION_GUIDE.md` for guard coverage and
  thresholds.
- **Telemetry**: Metrics persist to `var/data/perps_bot/<profile>/metrics.json`; the
  exporter in `scripts/monitoring/export_metrics.py` converts them to Prometheus.

## Module 4 – Development Workflow

1. Create a feature branch and implement changes in the relevant slice.
2. Add or update tests under `tests/unit/bot_v2/`.
3. Run `poetry run pytest -q` before opening a pull request.
4. Update documentation (`docs/README.md`, architecture/risk guides) alongside
   code changes.

## Module 5 – Operational Awareness

- Study `docs/ops/operations_runbook.md` and `docs/MONITORING_PLAYBOOK.md` for
  on-call expectations.
- Manual tooling:

  ```bash
  poetry run perps-bot --account-snapshot
  poetry run perps-bot --convert USD:USDC:250
  poetry run perps-bot --move-funds from_uuid:to_uuid:25
  ```

- Guardrails default to spot-only. Derivatives remain locked behind INTX access
  and `COINBASE_ENABLE_DERIVATIVES=1`.

## Module 6 – Further Reading

- Historical guides for the previous generation system were removed from the
  tree; refer to git history if you need them.
- Review agent-specific docs (`docs/agents/Agents.md`, `docs/agents/CLAUDE.md`, `docs/agents/Gemini.md`) for
  collaboration patterns.
