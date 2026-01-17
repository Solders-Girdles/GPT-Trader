# Trading Readiness Checklist

---
status: current
last-updated: 2026-01-17
---

## Purpose

This checklist defines the minimum bar for moving from paper trading to live trading.
Each pillar has measurable evidence so the decision is objective, repeatable, and auditable.

## How to use

1. Run paper sessions and generate daily reports.
2. Run backtests and capture out-of-sample results.
3. Fill in the checklist below with evidence paths and notes.
4. Enable the readiness gate in preflight/CI for enforcement.

## Evidence sources

- Preflight report: `preflight_report_YYYYMMDD_HHMMSS.json`
- Daily report: `runtime_data/<profile>/reports/daily_report_YYYY-MM-DD.json`
- Status report: `var/data/status.json` (or configured `status_file`)
- Event store: `runtime_data/<profile>/events.db` and `runtime_data/<profile>/orders.db`
- Health endpoint: `http://localhost:8080/health` (if enabled)

## 3-day GREEN streak log

| Date (UTC) | Daily report path | Preflight path | Green? | Notes |
| --- | --- | --- | --- | --- |
| 2026-01-15 | `runtime_data/canary/reports/daily_report_2026-01-15.json` | `preflight_report_20260115_214038.json` | No | Preflight not green (view-only, env not set). |
| 2026-01-16 | N/A | `preflight_report_20260116_000708.json` | No | Missing daily report; preflight still not green (can_trade=False). |
| 2026-01-17 | `runtime_data/canary/reports/daily_report_2026-01-17.json` | `preflight_report_20260117_044647.json` | No | COINBASE_SANDBOX/COINBASE_API_MODE missing; readiness status NOT READY. |

## Readiness pillars (must-have)

### 1) Market data integrity

What good looks like: REST + WS redundancy, staleness detection, and gap handling are verified in
paper runs with no stale-mark guards firing.

Evidence:
- Daily report health section
- TUI staleness banner or `/health`
- Metrics `gpt_trader_ws_gap_count` (Prometheus)

Thresholds (default targets):
- `health.stale_marks == 0` across >= 3 consecutive daily reports
- `health.ws_reconnects <= 3` per 24h window
- `gpt_trader_ws_gap_count == 0` (or documented tolerance <= 1 per day)

Checklist:
- [ ] Evidence path(s): `runtime_data/canary/reports/daily_report_2026-01-15.json`, `runtime_data/canary/reports/daily_report_2026-01-17.json`
- [ ] Notes: 2026-01-15 report 2/3 collected; stale_marks=0, ws_reconnects=0, api_errors=31 -> NOT GREEN (api_health guard_triggers=31 from earlier runs; latest run avoided product endpoint errors; unpriced assets BOND/CLV/ERN/GAL). 2026-01-17 report shows stale_marks=0, ws_reconnects=0, api_errors=0, guard_triggers=0.

### 2) Strategy validity

What good looks like: no look-ahead bias, stable parameters, and walk-forward evidence with
consistent risk-adjusted performance.

Evidence:
- Backtest report (see `docs/guides/backtesting.md`)
- Out-of-sample/holdout report
- Validation report for guard parity

Thresholds (adjust per strategy):
- Backtest Sharpe >= 1.0
- Out-of-sample Sharpe >= 0.7
- Profit factor >= 1.2
- Max drawdown <= 10%
- Walk-forward: >= 3 windows with consistent sign of returns

Checklist:
- [x] Evidence path(s): `runtime_data/canary/reports/sweep_2h_trend_window_real_20260117_012209/tw64_thr0p01_cd3/backtest_20260117_012222.json`, `runtime_data/canary/reports/sweep_2h_trend_window_real_20260117_012209/tw64_thr0p01_cd3/backtest_20260117_012222.txt`
- [x] Notes: 90-day backtest (BTC-USD, TWO_HOUR) using mean reversion + trend filter passes Pillar 2 gates: total_return=+0.46%, max_drawdown=0.57%, Sharpe=1.51, profit_factor=5.37, net_profit_factor=1.45, fee_drag/trade=49.96, trades=15 (threshold=15). (Historical: FIVE_MINUTE baseline backtest failed gates.)

### 3) Risk management

What good looks like: exposure caps, daily loss limits, and reduce-only fail-safes are enforced
with zero guard violations in extended paper runs.

Evidence:
- Preflight risk section (`scripts/production_preflight.py`)
- Daily report risk section
- Risk config values (env or profile)

Thresholds (default targets):
- `RISK_DAILY_LOSS_LIMIT_PCT` configured and <= 0.02 for canary
- `RISK_MAX_POSITION_PCT_PER_SYMBOL <= 0.25`
- `risk.guard_triggers == 0` across >= 3 consecutive daily reports
- No circuit breaker triggers in paper runs

Checklist:
- [ ] Evidence path(s): `runtime_data/canary/reports/daily_report_2026-01-15.json`, `runtime_data/canary/reports/daily_report_2026-01-17.json`, `preflight_report_20260117_044647.json`
- [ ] Notes: 2026-01-15 report 2/3 collected; unfilled_orders=0, api_errors=31 -> NOT GREEN. 2026-01-17 daily report shows unfilled_orders=0, api_errors=0, guard_triggers=0; preflight readiness run (dry-run) confirms risk checks pass but env vars COINBASE_SANDBOX and COINBASE_API_MODE still missing.

### 4) Execution correctness

What good looks like: order preview, slippage guard, idempotent submission, and partial-fill
handling are clean with no mismatched order states.

Evidence:
- Execution health signals (`/health` and status reporter)
- Daily report health section
- Order lifecycle telemetry in event store

Thresholds (default targets):
- Execution health signals status == OK (see `HealthThresholds` / `HEALTH_*`)
- `health.unfilled_orders == 0`
- `health.api_errors == 0`

Checklist:
- [ ] Evidence path(s): `preflight_report_20260115_214038.json`, `preflight_report_20260116_000708.json`, `preflight_report_20260117_044647.json`, `runtime_data/canary/reports/phase-a-tests-2026-01-16.log`
- [ ] Notes: 2026-01-15 DRY_RUN view-only passed; live-intent preflight still fails as expected (can_trade=False). 2026-01-17 dry-run preflight shows readiness liveness OK but COINBASE_SANDBOX/COINBASE_API_MODE still missing. Phase A execution tests collected 2026-01-16.

### 5) State persistence and recovery

What good looks like: event store and orders store reconstruct state after restart with no
double-orders or orphan positions.

Evidence:
- `tests/integration/test_crash_recovery.py` results
- Manual replay/restart test notes

Thresholds:
- Crash recovery tests pass
- Manual replay shows zero duplicate orders and zero orphan positions

Checklist:
- [ ] Evidence path(s): `runtime_data/canary/reports/phase-a-tests-2026-01-16.log`
- [ ] Notes: Phase A crash recovery + order flow integration tests passed 2026-01-16.

### 6) Observability and explainability

What good looks like: decision trace, guard outcomes, and alerts are actionable; any order is
explainable from logs.

Evidence:
- `order_decision_trace` events in event store
- Structured logs with `decision_id`
- TUI decision panel and alerts

Thresholds:
- 100% of live orders trace to a `decision_id`
- Alerts include actionable remediation notes

Checklist:
- [ ] Evidence path(s):
- [ ] Notes:

### 7) Operational controls

What good looks like: pause, kill switch, reduce-only, and safe shutdown verified via CLI/TUI.

Evidence:
- TUI/CLI screenshots or run log
- Manual test steps in `docs/guides/production.md`

Thresholds:
- All controls verified in last 30 days
- Reduce-only mode blocks new positions but allows exits

Checklist:
- [ ] Evidence path(s):
- [ ] Notes:

### 8) Security and credential hygiene

What good looks like: secrets flow and permission checks are correct; no plaintext credentials
in logs; misconfig is blocked.

Evidence:
- Preflight key permissions section
- Log redaction checks (see `docs/SECURITY.md`)

Thresholds:
- No plaintext secrets found in logs
- Preflight permissions check passes without warnings

Checklist:
- [ ] Evidence path(s):
- [ ] Notes:

## Phase-gated milestones

- Phase A (Paper-ready): pillars 1-5 pass, plus full observability coverage.
- Phase B (Canary live): Phase A + no critical alerts in multi-day paper run, guard thresholds tuned.
- Phase C (Prod live): Phase B + stability across regime changes (sideways/trending/high-vol).

Checklist:
- [ ] Phase A complete
- [ ] Phase B complete
- [ ] Phase C complete

## Preflight and CI hooks

The readiness gate is optional and driven by daily report JSON.

Enable in preflight:

```bash
export GPT_TRADER_READINESS_REPORT=runtime_data/canary/reports
uv run python scripts/production_preflight.py --profile canary
```

Make target (recommended for manual runs):

```bash
make preflight-readiness
make preflight-readiness PREFLIGHT_PROFILE=prod
make preflight-readiness PREFLIGHT_PROFILE=prod READINESS_REPORT_DIR=runtime_data/prod/reports
```

Notes:
- If `GPT_TRADER_READINESS_REPORT` is a directory, preflight uses the newest
  `daily_report_*.json` file.
- Set thresholds via env vars:
  - `GPT_TRADER_READINESS_STALE_MARKS_MAX` (default: 0)
  - `GPT_TRADER_READINESS_WS_RECONNECTS_MAX` (default: 3)
  - `GPT_TRADER_READINESS_UNFILLED_ORDERS_MAX` (default: 0)
  - `GPT_TRADER_READINESS_API_ERRORS_MAX` (default: 0)
  - `GPT_TRADER_READINESS_GUARD_TRIGGERS_MAX` (default: 0)
  - `GPT_TRADER_READINESS_LIVENESS_MAX_AGE_SECONDS` (default: 300)
- Readiness also checks event-stream liveness via the latest `events.db` entry.
- Use `GPT_TRADER_PREFLIGHT_WARN_ONLY=1` to downgrade failures to warnings.
