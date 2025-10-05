# Soak Test - Prometheus Query Reference

Quick reference for monitoring queries during sandbox soak test.

---

## Guardrails

### Guard States
```promql
# All active guards
bot_guard_active == 1

# Specific guard active
bot_guard_active{guard="daily_loss"} == 1
bot_guard_active{guard="circuit_breaker"} == 1
bot_guard_active{guard="dry_run"} == 1
```

### Guard Trip Events
```promql
# Total trips by guard type
sum by (guard, reason) (bot_guard_trips_total)

# Trips in last hour
increase(bot_guard_trips_total[1h])

# Max trade value blocks
bot_guard_trips_total{guard="max_trade_value"}

# Position limit blocks
bot_guard_trips_total{guard="position_limit"}
```

### Error Tracking
```promql
# Current error streak
bot_guard_error_streak

# Error streak over time
bot_guard_error_streak[1h]

# Average error streak
avg_over_time(bot_guard_error_streak[24h])
```

### Daily Loss Tracking
```promql
# Current daily loss
bot_guard_daily_loss_usd

# Loss approaching limit (>80%)
bot_guard_daily_loss_usd > 8

# Loss over time
bot_guard_daily_loss_usd[24h]
```

---

## Streaming Health

### Connection State
```promql
# Current connection state (1=connected, 0=disconnected)
bot_streaming_connection_state

# Connection uptime percentage (last 24h)
avg_over_time(bot_streaming_connection_state[24h]) * 100

# Connection downtime in last hour
(1 - avg_over_time(bot_streaming_connection_state[1h])) * 3600
```

### Heartbeat Monitoring
```promql
# Current heartbeat lag
bot_streaming_heartbeat_lag_seconds

# Heartbeat lag p95
histogram_quantile(0.95, rate(bot_streaming_heartbeat_lag_seconds[5m]))

# Stale heartbeat (>15s)
bot_streaming_heartbeat_lag_seconds > 15
```

### Reconnect Tracking
```promql
# Total reconnect attempts
bot_streaming_reconnect_total{status="attempt"}

# Successful reconnects
bot_streaming_reconnect_total{status="success"}

# Reconnect success rate
bot_streaming_reconnect_total{status="success"} /
bot_streaming_reconnect_total{status="attempt"}

# Reconnects in last 5 minutes
increase(bot_streaming_reconnect_total{status="attempt"}[5m])

# Reconnect storm detection (>3 in 5min)
increase(bot_streaming_reconnect_total{status="attempt"}[5m]) > 3
```

### Message Gaps
```promql
# Median inter-message gap
histogram_quantile(0.5, rate(bot_streaming_inter_message_seconds_bucket[5m]))

# p95 inter-message gap
histogram_quantile(0.95, rate(bot_streaming_inter_message_seconds_bucket[5m]))

# p99 inter-message gap
histogram_quantile(0.99, rate(bot_streaming_inter_message_seconds_bucket[5m]))

# Large gaps (>5s)
histogram_quantile(0.99, rate(bot_streaming_inter_message_seconds_bucket[5m])) > 5
```

### Fallback Status
```promql
# Fallback currently active
bot_streaming_fallback_active == 1

# Fallback uptime percentage
avg_over_time(bot_streaming_fallback_active[24h]) * 100

# Total time in fallback (last 24h, assuming 15s scrape interval)
sum_over_time(bot_streaming_fallback_active[24h]) * 15
```

---

## System Health

### Uptime
```promql
# Current uptime in seconds
bot_uptime_seconds

# Uptime in hours
bot_uptime_seconds / 3600

# Uptime in days
bot_uptime_seconds / 86400
```

### Cycle Performance
```promql
# Median cycle duration
histogram_quantile(0.5, rate(bot_cycle_duration_seconds_bucket[5m]))

# p95 cycle duration
histogram_quantile(0.95, rate(bot_cycle_duration_seconds_bucket[5m]))

# p99 cycle duration
histogram_quantile(0.99, rate(bot_cycle_duration_seconds_bucket[5m]))

# Slow cycles (>10s)
histogram_quantile(0.95, rate(bot_cycle_duration_seconds_bucket[5m])) > 10
```

### Resource Usage
```promql
# Current memory usage (MB)
bot_memory_used_bytes / 1024 / 1024

# Memory growth rate (MB/hour)
rate(bot_memory_used_bytes[1h]) / 1024 / 1024 * 3600

# CPU usage percentage
bot_cpu_percent

# High CPU (>80%)
bot_cpu_percent > 80
```

### Background Tasks
```promql
# Active background tasks
bot_background_tasks

# Tasks by type
bot_background_tasks{task="streaming"}
bot_background_tasks{task="telemetry"}
bot_background_tasks{task="reconciliation"}
```

---

## Order Execution

### Order Statistics
```promql
# Total order attempts
bot_order_attempts_total{status="attempted"}

# Successful orders
bot_order_attempts_total{status="success"}

# Failed orders
bot_order_attempts_total{status="failed"}

# Success rate
rate(bot_order_attempts_total{status="success"}[5m]) /
rate(bot_order_attempts_total{status="attempted"}[5m]) * 100

# Failure rate
rate(bot_order_attempts_total{status="failed"}[5m]) /
rate(bot_order_attempts_total{status="attempted"}[5m]) * 100
```

### Guard-Blocked Orders
```promql
# Orders blocked by any guard
sum(rate(bot_guard_trips_total[5m]))

# Orders blocked by specific guard
rate(bot_guard_trips_total{guard="max_trade_value"}[5m])
rate(bot_guard_trips_total{guard="position_limit"}[5m])
```

---

## Error Tracking

### Error Rates
```promql
# Total errors
bot_errors_total

# Errors by component
sum by (component) (bot_errors_total)

# Errors by severity
sum by (severity) (bot_errors_total)

# Error rate (errors/min)
rate(bot_errors_total[5m]) * 60

# High error rate (>2 errors/min)
rate(bot_errors_total[5m]) * 60 > 2
```

### Error Analysis
```promql
# Critical errors
bot_errors_total{severity="critical"}

# Errors by component
bot_errors_total{component="order_placement"}
bot_errors_total{component="streaming"}
bot_errors_total{component="risk_manager"}
```

---

## Alerts

### Active Alerts
```promql
# All firing alerts
ALERTS{alertstate="firing"}

# Critical alerts
ALERTS{alertstate="firing", severity="critical"}

# Warning alerts
ALERTS{alertstate="firing", severity="warning"}

# Specific alert
ALERTS{alertname="BotDailyLossLimitBreached"}
ALERTS{alertname="BotCircuitBreakerTripped"}
ALERTS{alertname="BotStreamingDisconnected"}
```

### Alert History
```promql
# Alert firing count
sum by (alertname) (ALERTS{alertstate="firing"})

# Time in alert state
sum by (alertname) (time() - ALERTS_FOR_STATE{alertstate="firing"})
```

---

## Composite Queries

### Overall Health Score
```promql
# Composite health (0-100)
(
  (bot_streaming_connection_state * 30) +
  ((1 - clamp_max(bot_streaming_heartbeat_lag_seconds / 15, 1)) * 20) +
  ((1 - clamp_max(bot_guard_error_streak / 5, 1)) * 20) +
  (avg_over_time(bot_streaming_connection_state[5m]) * 30)
)
```

### Data Quality Score
```promql
# Streaming quality (0-100)
(
  (bot_streaming_connection_state * 40) +
  ((1 - bot_streaming_fallback_active) * 30) +
  ((1 - clamp_max(bot_streaming_heartbeat_lag_seconds / 30, 1)) * 30)
)
```

### Guard Pressure
```promql
# Number of active guards
sum(bot_guard_active)

# Guard activation rate
rate(bot_guard_trips_total[5m])

# Guard pressure composite
(sum(bot_guard_active) * 50) + (rate(bot_guard_trips_total[5m]) * 100)
```

---

## Time-Range Queries

### Last Hour Summary
```promql
# Streaming uptime
avg_over_time(bot_streaming_connection_state[1h]) * 100

# Average error streak
avg_over_time(bot_guard_error_streak[1h])

# Total guard trips
increase(bot_guard_trips_total[1h])

# Order success rate
sum(increase(bot_order_attempts_total{status="success"}[1h])) /
sum(increase(bot_order_attempts_total{status="attempted"}[1h]))
```

### Last 24 Hours Summary
```promql
# Streaming uptime
avg_over_time(bot_streaming_connection_state[24h]) * 100

# Total downtime (seconds)
(1 - avg_over_time(bot_streaming_connection_state[24h])) * 86400

# Total reconnects
increase(bot_streaming_reconnect_total[24h])

# Memory growth
(max_over_time(bot_memory_used_bytes[24h]) -
 min_over_time(bot_memory_used_bytes[24h])) / 1024 / 1024
```

---

## Advanced Analysis

### Correlation Queries

**Guard Trips vs Error Streak**:
```promql
# Plot both on same graph
bot_guard_error_streak
rate(bot_guard_trips_total[5m]) * 10  # Scaled for visibility
```

**Streaming Issues vs Fallback**:
```promql
# Connection issues trigger fallback
(1 - bot_streaming_connection_state)
bot_streaming_fallback_active
```

**CPU vs Cycle Duration**:
```promql
bot_cpu_percent
histogram_quantile(0.95, rate(bot_cycle_duration_seconds_bucket[5m])) * 100  # Scaled
```

### Anomaly Detection

**Unusual Error Rate**:
```promql
# Detect if current error rate > 2x average
rate(bot_errors_total[5m]) >
2 * avg_over_time(rate(bot_errors_total[5m])[1h:5m])
```

**Unusual Cycle Duration**:
```promql
# Detect if p95 > 2x normal
histogram_quantile(0.95, rate(bot_cycle_duration_seconds_bucket[5m])) >
2 * avg_over_time(histogram_quantile(0.95, rate(bot_cycle_duration_seconds_bucket[5m]))[1h:5m])
```

**Streaming Instability**:
```promql
# Detect frequent reconnects
increase(bot_streaming_reconnect_total{status="attempt"}[5m]) >
avg_over_time(increase(bot_streaming_reconnect_total{status="attempt"}[5m])[1h:5m]) * 3
```

---

## Export Queries

For final report generation:

```bash
# Export all metrics for specific time range
START="2025-10-04T00:00:00Z"
END="2025-10-06T00:00:00Z"

# Guardrail metrics
curl -G "http://localhost:9090/api/v1/query_range" \
  --data-urlencode "query=bot_guard_active" \
  --data-urlencode "start=${START}" \
  --data-urlencode "end=${END}" \
  --data-urlencode "step=60s"

# Streaming metrics
curl -G "http://localhost:9090/api/v1/query_range" \
  --data-urlencode "query=bot_streaming_connection_state" \
  --data-urlencode "start=${START}" \
  --data-urlencode "end=${END}" \
  --data-urlencode "step=15s"

# System metrics
curl -G "http://localhost:9090/api/v1/query_range" \
  --data-urlencode "query=bot_uptime_seconds" \
  --data-urlencode "start=${START}" \
  --data-urlencode "end=${END}" \
  --data-urlencode "step=60s"
```
