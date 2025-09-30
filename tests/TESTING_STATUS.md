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
- `@pytest.mark.uses_mock_broker` suites remain opt-in (`-m uses_mock_broker`). They currently rely on legacy market-impact hooks that were removed from `LiveRiskManager`, so leave them out of CI until that guard is rebuilt.
- The behavioral utilities walkthrough lives in `docs/testing/behavioral_scenarios_demo.md`; it was removed from the unit suite to avoid price anchoring on real markets.
- Compatibility guards for removed alias modules are now tracked in `tests/fixtures/DEPRECATED.md` instead of a dedicated test file.

### Fixture Organization
- Shared data builders now live in `tests/fixtures/factories/` with modules per domain (market, portfolio, strategy, trade, risk) to keep responsibilities focused.
- Suite-wide fixtures are registered via `pytest_plugins` in `tests/conftest.py`, so new fixtures can be added under `tests/fixtures/` without touching per-suite `conftest` modules.
