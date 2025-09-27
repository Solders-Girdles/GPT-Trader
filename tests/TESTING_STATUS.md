## Test Suite Status and Scope

This repository contains both the active bot_v2 codepath and legacy v1 components. To keep development focused and metrics honest, legacy suites are explicitly skipped with reasons.

### Active Suites (Maintained)
- `bot_v2/features/live_trade/` — Perps trading logic, execution, risk
- `bot_v2/orchestration/` — Bot coordination and runtime
- `bot_v2/features/brokerages/coinbase/` — Coinbase exchange adapter, HTTP, WS

Target: ≥90% pass rate across these active suites.

### Legacy Suites (Skipped)
- `paper_engine v1` — Replaced by bot_v2 perps path and mock-based flows
- `backtest v1` — Superseded by bot_v2/features/backtest
- `Week 1–3` top-level tests — Reflect early prototypes and API shapes

Each skipped module includes a `pytest.mark.skip` with a reason string that documents the architectural evolution.

### Notes
- Some integration checks (e.g., environment and “reality check”) remain informational and may be tolerant of drift.
- If a legacy area is revived, remove the skip marker and realign tests with current interfaces.

