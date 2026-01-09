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
| `gpt_trader_guard_check_total` | Counter | `guard_name`, `outcome` | Guard check outcomes |
| `gpt_trader_guard_failure_total` | Counter | `guard_name`, `error_type` | Guard failures by type |

## Failure Reason Taxonomy

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

Classification logic: `order_submission._classify_rejection_reason()`

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

---

*Last updated: 2026-01-08*
