# Phase A Readiness Execution Plan

---
status: draft
last-updated: 2026-01-15
---

## Highest priority (right now)

Make the readiness evidence loop real and automated:
- Daily reports must be generated from canonical persistence (`runtime_data/<profile>/events.db`, `orders.db`) and include meaningful `health` + `risk` sections.
- The preflight readiness gate must run against the latest daily report and fail loudly when thresholds are breached.

## Current blockers

- `uv run gpt-trader report daily` currently reads `runtime_data/<profile>/events.jsonl` + `metrics.json` (legacy) which are not produced in current runs, so reports are empty.
- `make preflight-readiness` fails because `runtime_data/<profile>/reports/daily_report_*.json` does not exist yet.

## Next milestone: “Readiness Gate Operational” (P0)

Definition of done:
- `uv run gpt-trader report daily --profile canary --report-format both` writes `runtime_data/canary/reports/daily_report_YYYY-MM-DD.json` with real (non-placeholder) `health` + `risk` fields when runtime events exist.
- `make preflight-readiness PREFLIGHT_PROFILE=canary` passes when thresholds are met and fails when they’re exceeded.
- Docs reflect the canonical runtime sources (`runtime_data/…`) and don’t depend on legacy `var/data/…` paths.

## Work plan (two-week sprint)

### Workstream A — Reporting & data plumbing (Backend)

P0 tasks:
- Update the daily report loaders to support SQLite (`events.db`) when JSONL is absent.
  - Source of truth: `runtime_data/<profile>/events.db` (`events.timestamp`, `events.event_type`, `events.payload`).
  - Normalize timestamps to ISO-8601 UTC for consistent filtering and sorting.
  - Keep JSONL support for backwards compatibility.
- Add unit tests covering DB-backed loads and timestamp normalization.
- (Optional but recommended) If `metrics.json` is absent, derive equity/account snapshot from the newest “cycle metrics” event (or document the limitation clearly in the report output).

Acceptance checks:
- `uv run gpt-trader report daily --profile dev --report-format json --output-format json --no-save` no longer prints “Events file not found” when `runtime_data/dev/events.db` exists.
- Unit tests under `tests/unit/gpt_trader/monitoring/daily_report/` pass.

### Workstream B — Canary ops & evidence collection (Ops)

Daily checklist:
1. Run canary (dry-run OK): `uv run gpt-trader run --profile canary --dry-run`
2. Generate report: `uv run gpt-trader report daily --profile canary --report-format both`
3. Gate: `make preflight-readiness PREFLIGHT_PROFILE=canary`
4. Record evidence path(s) in `docs/READINESS.md` (reference local paths; do not commit secrets or account data).

Targets (from `docs/READINESS.md`, default thresholds):
- `health.stale_marks == 0`
- `health.ws_reconnects <= 3`
- `health.unfilled_orders == 0`
- `health.api_errors == 0`
- `risk.guard_triggers == 0` (total)
- Circuit breaker not triggered

### Workstream C — Strategy validity (Quant)

- Run backtests and capture out-of-sample results per `docs/guides/backtesting.md`.
- Attach evidence paths and a short summary in `docs/READINESS.md` pillar 2.
- Stop here if thresholds are not met; do not tune live risk to “force pass” before the strategy is valid.

### Workstream D — Risk, recovery, and controls (Backend + QA)

- Crash recovery: `uv run pytest -q tests/integration/test_crash_recovery.py`
- Order flow sanity: `uv run pytest -q tests/integration/test_order_flow.py`
- Validate operational controls (pause/kill switch/reduce-only) via CLI/TUI per `docs/guides/production.md`.

## Communication cadence

- Daily: async check-in with (1) report path, (2) `make preflight-readiness` result, (3) any guard triggers/errors.
- Weekly: readiness review; decide whether to widen canary window or promote to Phase B.
