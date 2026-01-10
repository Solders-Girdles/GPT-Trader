# Observability Reference

This document describes the metrics, traces, and structured logging used throughout the GPT-Trader system.

## Metrics

All metrics use the `gpt_trader_` prefix and follow Prometheus naming conventions.

### Order Submission Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `gpt_trader_order_submission_total` | Counter | `result`, `reason`, `side` | Total order submissions |
| `gpt_trader_order_submission_latency_seconds` | Histogram | `result`, `side` | End-to-end order submission latency |

**Label Values:**

- `result`: `success`, `rejected`, `failed`
- `reason`: `none` (success), or failure reason taxonomy (see below)
- `side`: `buy`, `sell`

### Broker Call Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `gpt_trader_broker_call_latency_seconds` | Histogram | `operation`, `outcome`, `reason` | Per-call broker API latency |

**Label Values:**

- `operation`: `submit`, `cancel`, `preview`
- `outcome`: `success`, `failure`
- `reason`: `none` (success), `timeout`, `network`, `rate_limit`, `error`

### Guard Metrics

Guard failures are tracked via the existing guard telemetry system:

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `gpt_trader_guard_checks_total` | Counter | `guard`, `result` | Guard check outcomes |
| `gpt_trader_guard_trips_total` | Counter | `guard`, `category`, `recoverable` | Guard trip counts |

## Order Submission Reason Taxonomy

Order rejections and failures are classified into standardized categories:

| Reason | Description |
|--------|-------------|
| `rate_limit` | API rate limit exceeded (429, "rate limit", "too many") |
| `insufficient_funds` | Insufficient balance, margin, or funds |
| `invalid_size` | Order size outside allowed bounds |
| `invalid_price` | Price doesn't meet tick/increment requirements |
| `timeout` | Request timed out |
| `network` | Connection, DNS, SSL, or socket errors |
| `market_closed` | Market is closed or trading halted |
| `rejected` | Generic broker rejection |
| `failed` | Generic failure |
| `unknown` | Unclassified error |

Classification logic for submission metrics: `order_submission._classify_rejection_reason()`

## Order Rejection Event Reasons

`event_type=order_rejected` payloads in the event store use normalized reason codes
plus an optional `reason_detail` with the raw value (for diagnostics without
exploding metric cardinality).

**Stable `reason` codes:**

- `paused`
- `quantity_zero`
- `reduce_only`
- `security_validation`
- `mark_staleness`
- `exchange_rules`
- `pre_trade_validation`
- `slippage_guard`
- `order_preview`
- `guard_error`
- `guard_failure`
- `broker_status`
- `broker_rejected`
- `invalid_request`
- `insufficient_funds`
- `rate_limit`
- `timeout`
- `network`
- `market_closed`
- `unknown`

## Traces

Distributed tracing spans are created for key operations:

### order_submit

Wraps the entire order submission flow.

**Attributes:**
- `bot_id`: Bot identifier
- `client_order_id`: Unique order identifier (stable across retries)
- `symbol`: Trading symbol
- `side`: `BUY` or `SELL`
- `order_type`: `MARKET`, `LIMIT`, etc.
- `quantity`: Order quantity
- `reduce_only`: Boolean

**Error Tags (on failure):**
- `error`: True
- `error.message`: Error description

### guard_check

Wraps each runtime guard invocation.

**Attributes:**
- `guard_name`: Name of the guard (e.g., `daily_loss`, `volatility`)
- `outcome`: `success` or `failure`
- `error_type`: Exception type (on failure)
- `recoverable`: Boolean (on failure)

## Structured Logging

All execution-path logs use structured logging with consistent fields:

### Common Fields

| Field | Description |
|-------|-------------|
| `operation` | High-level operation (e.g., `order_placement`, `order_route`) |
| `stage` | Sub-stage within operation (e.g., `start`, `success`, `failure`) |
| `symbol` | Trading symbol |
| `side` | Order side |
| `client_order_id` | Client order identifier (when available) |
| `order_id` | Broker-assigned order ID (when available) |

### Log Examples

```python
# Order submission start
logger.info(
    "Executing order",
    symbol="BTC-USD",
    action="BUY",
    operation="order_placement",
    stage="start",
)

# Router success
logger.info(
    "Executed spot order",
    symbol="BTC-USD",
    side="BUY",
    quantity="0.01",
    order_id="abc123",
    operation="order_route",
    stage="spot_success",
)
```

## Histogram Buckets

Default latency buckets (seconds):

```python
(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
```

These cover sub-millisecond to 10-second latencies, suitable for both fast market orders and slower limit order flows.

## Integration with OpenTelemetry

The `trace_span` helper auto-detects OpenTelemetry availability:

- When OTel is configured, spans are exported to the configured backend
- When OTel is not available, `trace_span` is a no-op (zero overhead)

To enable tracing:

```python
from gpt_trader.observability import init_tracing

init_tracing(
    service_name="gpt-trader",
    endpoint="http://localhost:4317",
    enabled=True,
)
```

## Dashboards and Alerts

Recommended Prometheus/Grafana queries:

### Order Success Rate

```promql
sum(rate(gpt_trader_order_submission_total{result="success"}[5m]))
/
sum(rate(gpt_trader_order_submission_total[5m]))
```

### P99 Submission Latency

```promql
histogram_quantile(0.99, rate(gpt_trader_order_submission_latency_seconds_bucket[5m]))
```

### Broker Call Failure Rate

```promql
sum(rate(gpt_trader_broker_call_latency_seconds_count{outcome="failure"}[5m]))
/
sum(rate(gpt_trader_broker_call_latency_seconds_count[5m]))
```

## Health Signals

Health signals provide threshold-based alerting for execution quality. Each signal has OK/WARN/CRIT status levels.

### Signal Definitions

| Signal Name | Unit | WARN Threshold | CRIT Threshold | Description |
|-------------|------|----------------|----------------|-------------|
| `order_error_rate` | ratio | 0.05 (5%) | 0.15 (15%) | Order submission failure rate |
| `order_retry_rate` | ratio | 0.10 (10%) | 0.25 (25%) | Orders requiring retry |
| `broker_latency_p95` | ms | 1000 | 3000 | 95th percentile broker API latency |
| `guard_trip_count` | count | 3 | 10 | Guard trips in rolling window |
| `ws_staleness` | seconds | 30 | 60 | Time since last WebSocket message |

### Status Levels

| Status | Meaning |
|--------|---------|
| `OK` | All signals within healthy range |
| `WARN` | One or more signals above warning threshold |
| `CRIT` | One or more signals above critical threshold |
| `UNKNOWN` | Unable to compute signals (e.g., no data) |

### Configuration

Health thresholds are configurable via environment variables:

```bash
# Order error rate thresholds
HEALTH_ORDER_ERROR_RATE_WARN=0.05
HEALTH_ORDER_ERROR_RATE_CRIT=0.15

# Order retry rate thresholds
HEALTH_ORDER_RETRY_RATE_WARN=0.10
HEALTH_ORDER_RETRY_RATE_CRIT=0.25

# Broker latency thresholds (milliseconds)
HEALTH_BROKER_LATENCY_MS_WARN=1000
HEALTH_BROKER_LATENCY_MS_CRIT=3000

# WebSocket staleness thresholds (seconds)
HEALTH_WS_STALENESS_SECONDS_WARN=30
HEALTH_WS_STALENESS_SECONDS_CRIT=60

# Guard trip count thresholds
HEALTH_GUARD_TRIP_COUNT_WARN=3
HEALTH_GUARD_TRIP_COUNT_CRIT=10
```

Or via `BotConfig.health_thresholds` in code.

### Integration Points

**Health Endpoint (`/health`):**

```json
{
  "status": "healthy",
  "live": true,
  "ready": true,
  "signals": {
    "status": "OK",
    "message": "All signals OK",
    "signals": [
      {
        "name": "order_error_rate",
        "status": "OK",
        "value": 0.02,
        "threshold_warn": 0.05,
        "threshold_crit": 0.15,
        "unit": ""
      }
    ]
  }
}
```

**Status Reporter (`BotStatus`):**

```python
status.health_state  # "OK", "WARN", "CRIT", or "UNKNOWN"
status.execution_signals  # Full signal summary dict
status.health_issues  # List of issues including WARN/CRIT signals
```

### Alert Examples

Trigger alerts when overall health degrades:

```promql
# Alert on CRIT status (unhealthy)
gpt_trader_health_status{status="CRIT"} == 1

# Alert on sustained WARN status
gpt_trader_health_status{status="WARN"} == 1
  for 5m
```

## Health Signals Pipeline

Health signals flow through a multi-stage pipeline from raw metrics to actionable status:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Metric Sources                                       │
│  • ExecutionTelemetryCollector (order success/retry rates)                  │
│  • BrokerExecutor (latency histograms)                                       │
│  • GuardManager (trip counts)                                                │
│  • WebSocket connection (staleness)                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    health_checks.py                                          │
│                                                                             │
│  • compute_execution_signals(collector, thresholds) → HealthSummary        │
│  • Evaluates each signal against WARN/CRIT thresholds                       │
│  • Returns worst-case status across all signals                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                        ┌─────────────┴─────────────┐
                        ▼                           ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│      /health endpoint          │   │     status_reporter.py         │
│                                │   │                                │
│  • health_server.py            │   │  • BotStatus.health_state      │
│  • Returns JSON with signals   │   │  • BotStatus.execution_signals │
│  • Used by load balancers      │   │  • Used by TUI and dashboards  │
└───────────────────────────────┘   └───────────────────────────────┘
```

**Key Files:**

| File | Purpose |
|------|---------|
| `src/gpt_trader/monitoring/health_signals.py` | Signal models (`HealthSignal`, `HealthSummary`, `HealthThresholds`) |
| `src/gpt_trader/monitoring/health_checks.py` | Signal computation and threshold evaluation |
| `src/gpt_trader/app/health_server.py` | HTTP `/health` endpoint serving signal summary |
| `src/gpt_trader/monitoring/status_reporter.py` | `BotStatus` integration for TUI/dashboards |

## Metrics & Tracing Coverage

The execution pipeline is fully instrumented:

### Submission Latency

| Metric | Instrumentation Point |
|--------|----------------------|
| `gpt_trader_order_submission_latency_seconds` | `OrderSubmitter.submit_order()` |
| `gpt_trader_broker_call_latency_seconds` | `BrokerExecutor._execute_broker_order()` |

### Broker Call Histograms

Each broker API call records:
- Operation type (`submit`, `cancel`, `preview`)
- Outcome (`success`, `failure`)
- Failure reason (`timeout`, `network`, `rate_limit`, `error`)

### Guard Trace Spans

Runtime guard checks emit `guard_check` spans with:
- `guard_name`: Which guard was evaluated
- `outcome`: `success` or `failure`
- `error_type`: Exception class on failure
- `recoverable`: Whether the failure is recoverable

### Correlation Context

The `correlation_context` manager threads trace IDs through:
- Log records (`trace_id`, `span_id` fields)
- Metric labels (where applicable)
- Event store records

Example:
```python
with correlation_context(symbol="BTC-USD", order_id="abc123"):
    # All logs and spans inherit context
    result = await submitter.submit_order(order)
```

---

*Last updated: 2026-01-08*
