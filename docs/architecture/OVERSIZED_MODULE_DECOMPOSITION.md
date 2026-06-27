# Oversized Module Decomposition Plan

---
status: current
---

This plan records the current oversized-module evidence for Issue #972 and
names reviewable seams for follow-up PRs. It is a decomposition plan, not a
license for broad rewrites.

## Current Evidence

Line counts from current `main`:

| Area | Current path | Lines | Notes |
| --- | --- | ---: | --- |
| Strategy engine | `src/gpt_trader/features/live_trade/engines/strategy.py` | 3506 | Issue #972 named the old `features/live_trade/strategy.py` path; current source uses the engine path. |
| Order submission | `src/gpt_trader/features/live_trade/execution/order_submission.py` | 1346 | Public `OrderSubmitter` imports are used by strategy engine tests and execution tests. |
| Health checks | `src/gpt_trader/monitoring/health_checks.py` | 1396 | Current PR extracts pure dependency planning to `health_check_planning.py`. |
| Backtesting simulation broker | `src/gpt_trader/backtesting/simulation/broker.py` | 1216 | `SimulatedBroker` is imported by backtesting engine, metrics, validation, optimization, and integration tests. |
| Live risk manager | `src/gpt_trader/features/live_trade/risk/manager/__init__.py` | 839 | `LiveRiskManager` is a public import for containers, risk protocols, and risk tests. |

## Candidate Seams

### Strategy engine

- Responsibilities: lifecycle setup, stream handling, open-order recovery,
  reconciliation, sizing, telemetry, and order handoff.
- Public surface: `TradingEngine` from
  `gpt_trader.features.live_trade.engines.strategy`.
- Test anchors: `tests/unit/gpt_trader/features/live_trade/engines/`.
- Safe decomposition targets: move telemetry/streaming helpers or
  reconciliation helpers behind private engine-local collaborators while
  keeping `TradingEngine` as the public import.

### Order submission

- Responsibilities: client-order-id generation, preview recording, pending and
  final submission records, retry decisions, metrics, and broker submission
  result normalization.
- Public surface: `OrderSubmitter`, `OrderSubmissionOutcome`, and
  `OrderSubmissionOutcomeStatus`.
- Test anchors: `tests/unit/gpt_trader/features/live_trade/execution/test_order_submission_*.py`.
- Safe decomposition targets: normalize/persist helpers and retry classification
  are separable from broker calls, but the first extraction should not change
  constructor behavior or submission ordering.

### Health checks

- Responsibilities: pure dependency planning, result normalization, concrete
  broker/websocket/ticker/degradation checks, runner lifecycle, and execution
  health-signal evaluation.
- Public surface: `gpt_trader.monitoring.health_checks` remains the
  compatibility import for runner, planner, descriptors, and check functions.
- Test anchors: `tests/unit/gpt_trader/monitoring/test_health_checks*.py`.
- Safe decomposition targets: pure planner types first, then execution
  health-signal helpers, then concrete probe groups.

### Backtesting simulation broker

- Responsibilities: simulated account state, products, balances, positions,
  quotes/candles, order placement, fills, funding, fees, and chaos hooks.
- Public surface: `SimulatedBroker`.
- Test anchors: `tests/unit/gpt_trader/backtesting/simulation/test_simulated_broker*.py`.
- Safe decomposition targets: account/position read models, order fill helpers,
  and funding/fee delegation. Keep `SimulatedBroker` as the public facade.

### Live risk manager

- Responsibilities: persisted risk state, order checks, exposure and margin
  validation, daily PnL, mark staleness, volatility circuit breakers, and
  risk status payloads.
- Public surface: `LiveRiskManager`, `RiskValidationError`, `ValidationError`,
  `VolatilityCheckOutcome`.
- Test anchors: `tests/unit/gpt_trader/features/live_trade/test_risk_manager*.py`
  and `tests/unit/gpt_trader/features/live_trade/risk/`.
- Safe decomposition targets: persisted state IO and status payload building
  are lower-risk than changing pre-trade validation, liquidation, or reduce-only
  behavior.

## First Boundary Selected

The first reviewable extraction is health-check dependency planning:

- Move `HealthCheckDescriptor`, `HealthCheckPlanner`, and related planning
  errors into `src/gpt_trader/monitoring/health_check_planning.py`.
- Re-export those names from `health_checks.py` so existing imports keep
  working.
- Verify with planner/runner tests and monitoring docs checks.

This boundary is intentionally pure. It does not touch broker calls, live
trading commands, production preflight, canary operations, order submission, or
account/venue capability.

## Focused Follow-Up Target

Open a follow-up PR or issue for the next health-check split:

- Target: extract execution health-signal evaluation from
  `src/gpt_trader/monitoring/health_checks.py` into a module named
  `gpt_trader.monitoring.execution_health_signals`.
- Include helpers currently grouped around
  `compute_execution_health_signals`, `_extract_execution_metrics`,
  `_build_execution_health_signals`, counter aggregation, p95 approximation,
  guard-trip counting, missing decision ID counting, and timeout-decision
  evaluation.
- Keep `compute_execution_health_signals` import-compatible from
  `health_checks.py`.
- Behavior tests should stay under
  `tests/unit/gpt_trader/monitoring/test_health_checks_signals.py` or move to a
  same-behavior test module with no snapshot-only success metric.

## Test Discipline

Decomposition PRs should use line counts only as supporting evidence. The
acceptance signal must be behavior-focused tests for the moved responsibility:

- planner/order tests for planning changes
- submission flow and retry tests for `OrderSubmitter` changes
- strategy engine reconciliation or telemetry tests for engine changes
- simulated broker market/trading tests for backtesting changes
- risk-manager daily/protection/volatility tests for risk changes
