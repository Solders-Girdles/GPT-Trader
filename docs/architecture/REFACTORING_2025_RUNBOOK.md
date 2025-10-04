# Refactoring 2025 Runbook

> **Scope:** Phase 0–3 modular extraction campaign (Sep–Oct 2025). Use this runbook as the single entry point for architecture, testing, and operational guidance after the stabilization commit `b2e71ea`.

## Current State Snapshot
- ✅ 6/7 targeted refactors shipped; remaining candidate parked for future signal.
- ✅ Coinbase API alignment validated against production flows (95% target coverage).
- ✅ 5,007 automated tests passing (up from 2,145) across unit, integration, and characterization suites.
- ✅ Legacy monoliths replaced by thin façades that delegate to well-tested collaborators.
- ➡️ Feature flags guard every major extraction, allowing same-day rollback without code changes.

## Subsystem Inventory
| Subsystem | Primary Module(s) | Key Behaviours | Reference |
|-----------|-------------------|----------------|-----------|
| Advanced Execution | `features/live_trade/advanced_execution.py` | Decision pipeline, telemetry export, validation hooks | [`ADVANCED_EXECUTION_REFACTOR.md`](ADVANCED_EXECUTION_REFACTOR.md)
| Liquidity & Order Policy | `features/live_trade/liquidity_service.py`, `features/live_trade/order_policy.py` | Liquidity scoring, risk gates, order policy composition | [`LIQUIDITY_SERVICE_REFACTOR.md`](LIQUIDITY_SERVICE_REFACTOR.md), [`ORDER_POLICY_REFACTOR.md`](ORDER_POLICY_REFACTOR.md)
| Rate Limits & Broker Glue | `features/live_trade/rate_limit_tracker.py`, `features/live_trade/broker_adapter.py` | Coinbase alignment, rate-limit tracking, broker abstraction | [`COINBASE_API_AUDIT.md`](COINBASE_API_AUDIT.md)
| Portfolio & PnL | `features/live_trade/portfolio_valuation.py`, `features/live_trade/position_valuer.py`, `features/live_trade/liquidity_metrics_tracker.py` | Portfolio valuation, mark management, liquidity metrics | [`PORTFOLIO_VALUATION_REFACTOR.md`](PORTFOLIO_VALUATION_REFACTOR.md), [`PNL_TRACKER_REFACTOR.md`](PNL_TRACKER_REFACTOR.md)
| Adaptive Portfolio | `features/adaptive_portfolio/strategy_selector.py`, `strategy_handlers/*` | Strategy registry, signal filtering, position sizing | [`DYNAMIC_SIZING_HELPER_REFACTOR.md`](DYNAMIC_SIZING_HELPER_REFACTOR.md)
| Paper Trading | `features/paper_trade/strategy_runner.py`, `dashboard/*` | Offline trading loop + dashboard façade | [`PAPER_TRADE_PHASE_5_COMPLETE.md`](../archive/refactoring-2025-q1/PAPER_TRADE_PHASE_5_COMPLETE.md)
| State Backups | `state/backup/workflow.py`, `backup/retention_manager.py`, `backup/scheduler.py` | Tiered backups, retention policy, scheduling | [`BACKUP_OPERATIONS_REFACTOR.md`](BACKUP_OPERATIONS_REFACTOR.md)
| State Repositories | `state/repositories/*`, `state/repository_factory.py` | Repository coordination, metrics collector wiring | [`STATE_REPOSITORIES_REFACTOR.md`](STATE_REPOSITORIES_REFACTOR.md)
| Recovery Workflow | `state/recovery/workflow.py` | Recovery orchestration, telemetry integration | [`RECOVERY_ORCHESTRATOR_REFACTOR.md`](RECOVERY_ORCHESTRATOR_REFACTOR.md)
| Perps Orchestrator | `orchestration/strategy_orchestrator.py`, `orchestration/perps_bot_builder.py` | Builder pattern, service orchestration, feature flags | [`STRATEGY_ORCHESTRATOR_REFACTOR.md`](STRATEGY_ORCHESTRATOR_REFACTOR.md), [`PHASE_3_COMPLETE_SUMMARY.md`](../archive/refactoring-2025-q1/PHASE_3_COMPLETE_SUMMARY.md)
| CLI Facade | `cli/commands/*`, `cli/parser.py`, `cli/bot_config_builder.py` | Modular command handlers, validation, lifecycle controller | [`CLI_ORDERS_PHASE_3_COMPLETE.md`](../archive/refactoring-2025-q1/CLI_ORDERS_PHASE_3_COMPLETE.md)

> See the **Archive Index** below for the full historical log of baseline, phase, and plan documents.

## Feature Flags & Rollback Levers
| Flag | Default | Scope | Rollback Behaviour |
|------|---------|-------|--------------------|
| `USE_NEW_MARKET_DATA_SERVICE` *(retired Oct 2025)* | — | Delegates PerpsBot mark updates to `MarketDataService` | Flag removed; MarketDataService is always active. |
| `USE_NEW_STREAMING_SERVICE` *(retired Oct 2025)* | — | Routes streaming through `StreamingService` | Flag removed; StreamingService is always active. Legacy methods removed. |
| `USE_PERPS_BOT_BUILDER` | `true` | Switches PerpsBot construction to `PerpsBotBuilder` | Set to `false` to run `_legacy_init()` (emits `DeprecationWarning`). |
| `USE_NEW_CLI_HANDLERS` | `true` | Enables modular CLI command handlers | Set to `false` to revert to monolithic command execution (kept for parity tests). |

> Rollbacks are config driven—no redeploy is required. Always pair a rollback with targeted smoke tests and re-enable once mitigated.

## Testing Posture
- **Unit suites:** 4,700+ tests covering extracted modules (adaptive portfolio, live trading, state, CLI).
- **Integration/characterization:** 300+ tests assert parity between new services and legacy behaviour, including flag-controlled rollback paths.
- **Hygiene allowlist:** 39 large suites added to the hygiene allowlist—see `tests/README.md` for execution guidance.
- **Recommended cadence:**
  - CI: `pytest` default (fast unit suites)
  - Nightly: `pytest -m characterization` + integration smoke
  - Release: `pytest --maxfail=1` + targeted Coinbase API contract tests

## Operational Checklist
- Confirm production dashboards & alerts cover:
  - Market data / streaming service health (heartbeat, lag, rate limits)
  - Coinbase API error classes introduced during alignment
  - Backup/recovery workflows (success, retention, drift)
- Run a 48-hour drift review post-deploy:
  - Compare live metrics vs. Phase 0 baseline
  - Inspect new telemetry series for regressions or noise
  - Capture learnings in the Monitoring Playbook
- Keep the stabilization backlog groomed:
  - Tag signal-driven items (extract → test → compose) vs. optional polish
  - Archive completed doc tasks in `docs/archive/refactoring-2025-q1/`

## Extending the Architecture
1. **Reuse the Extract → Test → Compose checklist:** Start with characterization tests, build focused unit coverage, then delegate from the façade.
2. **Lock sharing patterns:** MarketDataService and StreamingService expect the PerpsBot lock instances—pass them explicitly when creating new services.
3. **Telemetry:** Use `MetricsCollector` for new counters/gauges; document expectations in `MONITORING_PLAYBOOK.md`.
4. **CLI additions:** Register new commands under `cli/commands/`, wire into `cli/parser.py`, and cover via unit tests in `tests/unit/bot_v2/cli/`.

## Archive Index
Historical phase-by-phase notes, baselines, and planning artifacts now live at:
```
docs/archive/refactoring-2025-q1/
```
Highlights include:
- Phase summaries (`PHASE_0_COMPLETE_SUMMARY.md` → `PHASE_3_COMPLETE_SUMMARY.md`)
- CLI baseline → completion logs
- Paper-trade dashboard iterations
- Recovery telemetry plan and refactoring survey

Retain these documents for forensic or onboarding purposes; they no longer clutter the active architecture directory.

## Open Follow-Ups
- Monitor production telemetry and log signatures for the next 48 hours; document any drift.
- Revisit backlog triage after the monitoring window to decide when to resume Phase 4+ work.
- Schedule knowledge transfer only if runtime signals or bandwidth constraints emerge.

For gaps or new discoveries, append them here and create linked issues so this runbook remains the authoritative source.
