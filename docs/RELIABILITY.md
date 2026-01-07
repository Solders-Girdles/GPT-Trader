# Reliability Layer

This document describes the reliability layer that protects GPT-Trader during
startup and live trading. It covers the decision matrix, configuration defaults,
the chaos harness for fault injection, and a rollout checklist.

## Scope

Reliability is delivered by four packages that work together:

- API Health Guard: runtime checks of error rate, rate limits, and circuit breakers.
- Preflight Diagnostics: startup checks for API connectivity, accounts, and market data.
- Graceful Degradation: pause and reduce-only responses when safety checks trip.
- Chaos Harness: fault injection utilities for testing degradation behavior.

Implementation references:

- `src/gpt_trader/orchestration/execution/guards/api_health.py`
- `src/gpt_trader/preflight/checks/diagnostics.py`
- `src/gpt_trader/orchestration/execution/degradation.py`
- `tests/support/chaos.py`

## Decision Matrix

| Trigger | Signal | Response | Config |
| --- | --- | --- | --- |
| Preflight diagnostics fail | `gpt-trader preflight` errors on API, accounts, or market data | Exit non-zero; block startup unless warn-only | `--warn-only` or `GPT_TRADER_PREFLIGHT_WARN_ONLY=1` |
| API health trip | ApiHealthGuard sees open circuit or thresholds | Cancel open orders, reduce-only, global pause | `RISK_API_HEALTH_COOLDOWN_SECONDS` |
| Mark staleness | `check_mark_staleness` true | Pause symbol; allow reduce-only if configured | `RISK_MARK_STALENESS_COOLDOWN_SECONDS`, `RISK_MARK_STALENESS_ALLOW_REDUCE_ONLY` |
| Slippage failures | Repeated slippage guard ValidationError | Pause symbol after threshold | `RISK_SLIPPAGE_FAILURE_PAUSE_AFTER`, `RISK_SLIPPAGE_PAUSE_SECONDS` |
| Validation infra failure | ValidationFailureTracker escalation | Reduce-only + global pause | `RISK_VALIDATION_FAILURE_COOLDOWN_SECONDS` |
| Preview failures | Preview exceptions reach threshold | Disable preview for the session | `RISK_PREVIEW_FAILURE_DISABLE_AFTER` |
| Broker read failures | Consecutive balance/position read failures | Global pause (reduce-only allowed) | `RISK_BROKER_OUTAGE_MAX_FAILURES`, `RISK_BROKER_OUTAGE_COOLDOWN_SECONDS` |

Notes:

- Global and symbol pauses allow reduce-only orders by default.
- Reduce-only blocks new positions but allows position-closing orders.

## Config Knobs and Defaults

API Health Guard:

| Env Var | Default | Purpose |
| --- | --- | --- |
| `RISK_API_ERROR_RATE_THRESHOLD` | `0.2` | Error rate threshold to trip the guard |
| `RISK_API_RATE_LIMIT_USAGE_THRESHOLD` | `0.9` | Rate limit usage threshold to trip the guard |

Graceful Degradation:

| Env Var | Default | Purpose |
| --- | --- | --- |
| `RISK_API_HEALTH_COOLDOWN_SECONDS` | `300` | Global pause duration after API health trip |
| `RISK_MARK_STALENESS_COOLDOWN_SECONDS` | `120` | Per-symbol pause when mark data is stale |
| `RISK_MARK_STALENESS_ALLOW_REDUCE_ONLY` | `1` | Allow reduce-only during mark staleness |
| `RISK_SLIPPAGE_FAILURE_PAUSE_AFTER` | `3` | Failures before symbol pause |
| `RISK_SLIPPAGE_PAUSE_SECONDS` | `60` | Per-symbol pause duration for slippage |
| `RISK_VALIDATION_FAILURE_COOLDOWN_SECONDS` | `180` | Global pause after validation infra failure |
| `RISK_PREVIEW_FAILURE_DISABLE_AFTER` | `5` | Failures before preview auto-disable |
| `RISK_BROKER_OUTAGE_MAX_FAILURES` | `3` | Failures before global pause |
| `RISK_BROKER_OUTAGE_COOLDOWN_SECONDS` | `120` | Global pause duration for broker outage |

Preflight Diagnostics:

| Flag/Env | Default | Purpose |
| --- | --- | --- |
| `--warn-only` | off | Downgrade preflight errors to warnings |
| `GPT_TRADER_PREFLIGHT_WARN_ONLY` | `0` | Env alias for warn-only |
| `COINBASE_PREFLIGHT_SKIP_REMOTE` | unset | Skip remote checks (dev/offline) |
| `COINBASE_PREFLIGHT_FORCE_REMOTE` | unset | Force remote checks even on dev |

## Chaos Harness (Fault Injection)

The chaos harness is intended for deterministic tests that validate the
degradation responses without flaky network dependencies.

Key types and helpers (from `tests/support/chaos.py`):

- `FaultAction`: describes a fault (after_calls, times, raise_exc, return_value).
- `FaultPlan`: manages ordered faults with per-method call tracking.
- `ChaosBroker`: proxy that applies FaultPlan to a wrapped broker.
- Helpers: `fault_once`, `fault_after`, `fault_always`, `fault_sequence`.

Example: inject a broker outage after two balance reads.

```python
from tests.support.chaos import ChaosBroker, FaultPlan, fault_after
from gpt_trader.orchestration.deterministic_broker import DeterministicBroker

plan = FaultPlan().add(
    "list_balances",
    fault_after(2, raise_exc=TimeoutError("broker read timeout")),
)
broker = ChaosBroker(DeterministicBroker(), plan)
```

Example: use a scenario preset.

```python
from tests.support.chaos import ChaosBroker, api_outage_scenario
from gpt_trader.orchestration.deterministic_broker import DeterministicBroker

broker = ChaosBroker(DeterministicBroker(), api_outage_scenario())
```

Tip: patch `time.sleep` in tests if you use delayed faults to keep unit tests fast.

## Rollout Checklist

1. Run preflight in canary and prod:
   - `gpt-trader preflight --profile canary`
   - `gpt-trader preflight --profile prod`
2. Confirm reliability defaults in `.env` or exported overrides for RISK_* vars.
3. Run chaos tests:
   - `pytest tests/unit/support/test_chaos.py`
   - `pytest tests/unit/gpt_trader/features/live_trade/engines/test_strategy_engine_chaos.py`
4. Canary deploy with reduce-only for 24h; monitor guard and pause logs.
5. Promote to prod; keep preflight reports and guard events archived.
6. Only use `--warn-only` during incident response and document the reason.
