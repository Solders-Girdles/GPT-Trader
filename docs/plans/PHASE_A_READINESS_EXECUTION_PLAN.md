# Phase A Readiness Execution Plan

---
status: draft
last-updated: 2026-01-17
---

## Highest priority (right now)

Maintain the readiness evidence loop and keep liveness green:
- Run canary sessions that write to `runtime_data/<profile>/events.db` and keep last_event_age_seconds <= 300.
- Generate daily reports and preflight readiness from the canonical DB-backed telemetry.

## Current blockers

- Daily reports and readiness preflight now read from `runtime_data/<profile>/events.db` (with JSONL fallback), so the legacy JSONL/metrics pipeline is no longer blocking readiness signals.
- Remaining blockers are operational (canary liveness, environment vars for preflight readiness, and clean guard runs).

## Next milestone: “Readiness Gate Operational” (P0)

Definition of done:
- `uv run gpt-trader report daily --profile canary --report-format both` writes `runtime_data/canary/reports/daily_report_YYYY-MM-DD.json` with real (non-placeholder) `health` + `risk` fields when runtime events exist.
- `make preflight-readiness PREFLIGHT_PROFILE=canary` passes when thresholds are met and fails when they’re exceeded.
- Docs reflect the canonical runtime sources (`runtime_data/…`) and don’t depend on legacy `var/data/…` paths.

Status: ✅ Achieved (2026-01-17). Remaining work is operational evidence collection.

## Work plan (two-week sprint)

(Updated 2026-01-17: readiness gate is operational; focus on ops cadence.)

### Workstream A — Reporting & data plumbing (Backend)

P0 tasks:
- Daily report loaders now use SQLite (`events.db`) first with JSONL fallback, and metrics fall back to the latest `cycle_metrics` event when `metrics.json` is absent.
- Unit tests cover DB-backed loads and timestamp normalization (kept for regression).
- Monitoring exporter and dashboard now read from `runtime_data/<profile>/events.db` first.

Acceptance checks:
- `uv run gpt-trader report daily --profile dev --report-format json --output-format json --no-save` no longer prints “Events file not found” when `runtime_data/dev/events.db` exists.
- Unit tests under `tests/unit/gpt_trader/monitoring/daily_report/` pass.

### Workstream B — Canary ops & evidence collection (Ops)

Daily checklist:
1. Verify liveness: `make canary-liveness` (RED if last_event_age_seconds > 300)
2. Run canary (dry-run OK): `uv run gpt-trader run --profile canary --dry-run`
3. Generate report: `uv run gpt-trader report daily --profile canary --report-format both`
4. Gate: `make preflight-readiness PREFLIGHT_PROFILE=canary`
5. Record evidence path(s) in `docs/READINESS.md` (reference local paths; do not commit secrets or account data).

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
