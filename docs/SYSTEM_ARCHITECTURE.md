# GPT-Trader System Architecture (bot_v2)

This document summarizes the live architecture for the spot-focused
`perps-bot` stack. For an in-depth design discussion, refer to
`docs/ARCHITECTURE.md`.

## High-Level Flow

```
CLI (perps-bot) → Config (BotConfig) → Service Registry → LiveExecutionEngine →
Risk Guards → Coinbase Brokerage Adapter → Metrics + Telemetry
```

### Entry Point

- `poetry run perps-bot` loads `bot_v2/cli.py` and derives a `BotConfig` from
  CLI arguments plus environment overrides.
- `bot_v2/orchestration/bootstrap.py` wires dependencies using
  `ServiceRegistry`.

### Core Subsystems

| Module | Purpose |
|--------|---------|
| `bot_v2/features/live_trade` | Control loop, position tracking, and order routing. |
| `bot_v2/features/brokerages/coinbase` | REST/WS integration for Coinbase Advanced Trade spot markets. |
| `bot_v2/features/position_sizing` | Kelly-style position sizing with guardrails. |
| `bot_v2/features/monitor` | Metrics aggregation and persistence. |
| `bot_v2/features/utils` | Shared helpers (clock, price normalisation, etc.). |

### Risk & Safety Rails

- Implemented inside `LiveExecutionEngine` and guard modules under
  `bot_v2/features/live_trade/guards/`.
- Daily loss, volatility, correlation, and staleness checks emit structured
  events and gate order placement.

### Telemetry

- Metrics persisted to `var/data/perps_bot/<profile>/metrics.json`.
- Optional Prometheus exporter (`scripts/monitoring/export_metrics.py`).
- Logs routed through `bot_v2/logging_setup.py`.

### Derivatives Gate

- Perpetual futures remain behind `COINBASE_ENABLE_DERIVATIVES` and Coinbase
  INTX credentials.
- Code paths stay compiled; risk defaults assume spot-only until the gate is
  open.

## Legacy Context

Previous docs described the monolithic `src/bot` package, Streamlit dashboards,
and ML orchestrators. Those resources were removed from the tree; fetch them
from git history if you need a reference. The live stack is Coinbase spot-first
and those documents no longer reflect the production code base.
