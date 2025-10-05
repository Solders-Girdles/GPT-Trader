# Sandbox Soak Test Plan - Phase 3.2/3.3 Validation

**Duration**: 24-48 hours
**Environment**: Coinbase Sandbox
**Objective**: Validate guardrails, streaming resilience, and observability infrastructure under real trading conditions

---

## Test Objectives

### Primary Goals
1. **Guardrail Validation** - Verify all safety guards trigger correctly
2. **Streaming Resilience** - Validate WebSocket fallback and recovery
3. **Observability** - Confirm metrics/alerts fire as expected
4. **Data Quality** - Ensure REST fallback maintains price accuracy

### Success Criteria
- ✅ All guardrails trigger within expected thresholds
- ✅ Streaming fallback activates within 30s of disconnect
- ✅ No data gaps >10s during fallback periods
- ✅ Prometheus alerts fire correctly (no false positives/negatives)
- ✅ Bot recovers automatically from all error scenarios
- ✅ /health endpoint reflects accurate system state

---

## Test Configuration

### Bot Profile: `canary`
```yaml
# config/profiles/canary.yaml
trading:
  mode: reduce_only
  symbols: [BTC-USD]

risk_management:
  max_leverage: 1
  max_position_size: 500
  daily_loss_limit: 10
  max_trade_value: 100

order_policy:
  time_in_force: IOC

session:
  start_time: "00:00"
  end_time: "23:59"
  days: ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

monitoring:
  metrics:
    interval_seconds: 5
  streaming:
    enabled: true
    level: 1
    rest_poll_interval: 5.0
```

### Guardrails Configuration
- **Error threshold**: 5 consecutive errors
- **Error cooldown**: 300 seconds (5 minutes)
- **Daily loss limit**: $10 USD
- **Max trade value**: $100 USD per order
- **Symbol position caps**: BTC-USD: 0.01 BTC

### Environment Variables
```bash
COINBASE_API_KEY=<sandbox_key>
COINBASE_API_SECRET=<sandbox_secret>
PERPS_ENABLE_STREAMING=1
PERPS_STREAM_LEVEL=1
PERPS_STREAMING_REST_INTERVAL=5.0
```

---

## Test Scenarios

### Scenario 1: Normal Operation Baseline
**Duration**: First 4 hours
**Expected Behavior**:
- WebSocket connected and stable
- Regular heartbeats every 1-2 seconds
- Mark prices updating via stream
- No guards active
- No alerts firing

**Metrics to Monitor**:
- `bot_streaming_connection_state` = 1
- `bot_streaming_heartbeat_lag_seconds` < 5
- `bot_guard_active` = 0 for all guards
- `bot_uptime_seconds` continuously increasing

---

### Scenario 2: Order Cap Guard
**Trigger**: Attempt order exceeding $100 USD
**Method**: Manually configure decision with large quantity
**Expected Behavior**:
- Order blocked by `max_trade_value` guard
- `bot_guard_trips_total{guard="max_trade_value"}` increments
- Order stats show attempted but not failed
- System continues normal operation

**Validation**:
```promql
# Should increment during test
bot_guard_trips_total{guard="max_trade_value"}

# Should NOT activate (order-level guard)
bot_guard_active{guard="max_trade_value"} == 0
```

---

### Scenario 3: Symbol Position Cap
**Trigger**: Attempt order exceeding 0.01 BTC
**Method**: Place order for 0.02 BTC
**Expected Behavior**:
- Order blocked by `position_limit` guard
- Guard trip counter increments
- Specific reason logged with symbol

**Validation**:
```promql
bot_guard_trips_total{guard="position_limit",reason=~".*BTC-USD.*"}
```

---

### Scenario 4: Daily Loss Limit
**Trigger**: Accumulate >$10 in realized losses
**Method**: Close losing positions or simulate loss via test harness
**Expected Behavior**:
- `bot_guard_active{guard="daily_loss"}` = 1
- Bot enters reduce-only mode
- Alert `BotDailyLossLimitBreached` fires
- Metric `bot_guard_daily_loss_usd` >= 10

**Validation**:
```promql
# Guard should activate
bot_guard_active{guard="daily_loss"} == 1

# Reduce-only mode active
# (Check via /health endpoint or logs)

# Alert fires
ALERTS{alertname="BotDailyLossLimitBreached",severity="critical"}
```

**Reset**: Wait for UTC midnight or restart bot to reset daily tracking

---

### Scenario 5: Circuit Breaker Error Streak
**Trigger**: Force 5 consecutive order errors
**Method**:
- Configure broker to reject orders (mock mode)
- Or trigger validation errors repeatedly

**Expected Behavior**:
- `bot_guard_error_streak` increments to 5
- `bot_guard_active{guard="circuit_breaker"}` = 1
- Alert `BotCircuitBreakerTripped` fires
- Trading halts until cooldown (5min) or manual reset

**Validation**:
```promql
# Streak builds up
bot_guard_error_streak >= 5

# Circuit breaker activates
bot_guard_active{guard="circuit_breaker"} == 1

# Alert fires
ALERTS{alertname="BotCircuitBreakerTripped",severity="critical"}
```

**Reset**:
- Wait 5 minutes (cooldown)
- Or record successful order to reset streak

---

### Scenario 6: Streaming Disconnect & Fallback
**Trigger**: Kill WebSocket connection
**Method**:
- Firewall block to Coinbase WebSocket endpoint
- Or SIGSTOP the bot process briefly

**Expected Behavior**:
- `bot_streaming_connection_state` = 0
- REST fallback activates within 30 seconds
- `bot_streaming_fallback_active` = 1
- Mark prices continue updating via REST polling
- Alert `BotStreamingDisconnected` fires after 1 minute

**Validation**:
```promql
# Connection drops
bot_streaming_connection_state == 0

# Fallback activates
bot_streaming_fallback_active == 1

# Alert fires after 1min
ALERTS{alertname="BotStreamingDisconnected",severity="critical"}

# Heartbeat lag increases but REST fills gap
bot_streaming_heartbeat_lag_seconds > 15
```

**Recovery**:
- Restore connection
- WebSocket should reconnect automatically
- Fallback should deactivate
- Connection state returns to 1

---

### Scenario 7: Reconnect Storm
**Trigger**: Repeated connection failures
**Method**:
- Intermittent network issues (manual firewall flapping)
- Or restart Coinbase sandbox (if possible)

**Expected Behavior**:
- `bot_streaming_reconnect_total{status="attempt"}` increments
- REST fallback activates after 2nd attempt
- Alert `BotStreamingReconnectStorm` fires after 3 reconnects in 5min
- Eventually stabilizes or hits max retries

**Validation**:
```promql
# Multiple reconnect attempts
increase(bot_streaming_reconnect_total{status="attempt"}[5m]) > 3

# Alert fires
ALERTS{alertname="BotStreamingReconnectStorm",severity="warning"}

# Fallback active during instability
bot_streaming_fallback_active == 1
```

---

### Scenario 8: Heartbeat Stale
**Trigger**: No messages for >15 seconds
**Method**:
- Pause message flow (possible via mock transport)
- Or very quiet market conditions

**Expected Behavior**:
- `bot_streaming_heartbeat_lag_seconds` > 15
- Alert `BotStreamingHeartbeatStale` fires after 2min
- REST fallback activates if lag persists

**Validation**:
```promql
# Heartbeat lag grows
bot_streaming_heartbeat_lag_seconds > 15

# Alert fires after 2min sustained
ALERTS{alertname="BotStreamingHeartbeatStale",severity="warning"}
```

---

### Scenario 9: Daily P&L Reset
**Trigger**: UTC midnight rollover
**Expected Behavior**:
- `bot_guard_daily_loss_usd` resets to 0
- If `daily_loss` guard was active, it clears
- Trading resumes if reduce-only was enabled

**Validation**:
- Monitor metric across midnight boundary
- Verify guard clears automatically
- Check logs for "New trading day detected, resetting daily P&L tracking"

---

### Scenario 10: Config Hot Reload
**Trigger**: Update canary.yaml and send SIGHUP
**Changes**: Modify daily_loss_limit, streaming_rest_poll_interval
**Expected Behavior**:
- Bot reloads config without restart
- Guardrails update with new limits
- Streaming service updates poll interval
- No connection drops or errors

**Validation**:
- Check logs for "Applying configuration change diff=..."
- Verify new limits in /health endpoint
- Metrics continue without gaps

---

## Monitoring & Data Collection

### Prometheus Queries for Analysis

**Guardrail Effectiveness**:
```promql
# Total guard trips by type
sum by (guard, reason) (bot_guard_trips_total)

# Current guard states
bot_guard_active

# Error streak evolution
bot_guard_error_streak

# Daily loss tracking
bot_guard_daily_loss_usd
```

**Streaming Health**:
```promql
# Connection uptime %
avg_over_time(bot_streaming_connection_state[24h]) * 100

# Reconnect frequency
rate(bot_streaming_reconnect_total[1h])

# Fallback activation time
sum_over_time(bot_streaming_fallback_active[24h]) * 5s  # assuming 5s scrape

# Message latency percentiles
histogram_quantile(0.95, rate(bot_streaming_inter_message_seconds_bucket[5m]))
histogram_quantile(0.99, rate(bot_streaming_inter_message_seconds_bucket[5m]))
```

**Order Execution**:
```promql
# Order success rate
rate(bot_order_attempts_total{status="success"}[5m]) /
rate(bot_order_attempts_total{status="attempted"}[5m])

# Guard-blocked orders
rate(bot_guard_trips_total{guard=~"max_trade_value|position_limit"}[5m])
```

**System Health**:
```promql
# Uptime
bot_uptime_seconds

# Cycle execution time
histogram_quantile(0.95, rate(bot_cycle_duration_seconds_bucket[5m]))

# Memory/CPU trends
rate(bot_memory_used_bytes[5m])
bot_cpu_percent
```

### Grafana Dashboard

Create temporary soak test dashboard with panels for:
1. **Guardrails Overview** - Guard states, trips, error streak, daily loss
2. **Streaming Health** - Connection state, heartbeat lag, reconnects, fallback
3. **Data Quality** - Mark price gaps, message latency, REST vs WS comparison
4. **System Resources** - CPU, memory, cycle duration
5. **Alerts Timeline** - Alert firing history

### Alert Validation Checklist

During soak test, verify each alert:
- [ ] `BotDailyLossLimitBreached` - Fires correctly at $10 loss
- [ ] `BotDailyLossApproachingLimit` - Fires at 80% threshold ($8)
- [ ] `BotErrorStreakElevated` - Fires at 3 consecutive errors
- [ ] `BotCircuitBreakerTripped` - Fires when streak hits 5
- [ ] `BotStreamingDisconnected` - Fires after 1min disconnect
- [ ] `BotStreamingHeartbeatStale` - Fires after 2min stale heartbeat
- [ ] `BotStreamingReconnectStorm` - Fires after 3 reconnects in 5min
- [ ] No false positives during normal operation

---

## Data Collection

### Metrics Export
After 24-48 hours, export:
```bash
# Export all bot metrics for analysis
curl 'http://localhost:9090/api/v1/query_range?query=bot_.*&start=<start>&end=<end>&step=15s' > soak_metrics.json

# Export alert history
curl 'http://localhost:9090/api/v1/query?query=ALERTS' > soak_alerts.json
```

### Log Analysis
Collect and analyze logs for:
- Guard activation/deactivation events
- Streaming connect/disconnect cycles
- REST fallback start/stop events
- Configuration reload events
- Error patterns and recovery

```bash
# Extract guard events
grep -E "guard|circuit|daily_loss|max_trade" bot.log > guard_events.log

# Extract streaming events
grep -E "streaming|websocket|ws_|fallback" bot.log > streaming_events.log

# Extract errors
grep -E "ERROR|CRITICAL" bot.log > errors.log
```

### /health Endpoint Snapshots
Capture /health responses at key moments:
- Baseline (normal operation)
- During streaming fallback
- During guard activation
- After recovery

```bash
# Periodic health snapshots
while true; do
  date +%s >> health_snapshots.jsonl
  curl -s http://localhost:9090/health >> health_snapshots.jsonl
  echo >> health_snapshots.jsonl
  sleep 300  # Every 5 minutes
done
```

---

## Success Metrics

### Guardrails (Target: 100% reliability)
- [ ] All cap guards blocked orders correctly (0 bypasses)
- [ ] Daily loss guard activated at exactly $10
- [ ] Circuit breaker engaged after 5 errors
- [ ] Guards auto-cleared after cooldown/reset conditions
- [ ] No false activations during normal operation

### Streaming (Target: >99% uptime with graceful degradation)
- [ ] WebSocket uptime >99%
- [ ] Fallback activated within 30s of disconnect
- [ ] No mark price gaps >10s during fallback
- [ ] Reconnect successful within 3 attempts
- [ ] Fallback deactivated within 10s of stream recovery

### Observability (Target: 100% alert accuracy)
- [ ] All expected alerts fired correctly
- [ ] Zero false positive alerts
- [ ] Metrics reflected actual system state
- [ ] /health endpoint always accurate
- [ ] Grafana dashboard usable for real-time monitoring

### System Stability (Target: Zero crashes/hangs)
- [ ] No bot crashes or unexpected exits
- [ ] No deadlocks or hangs
- [ ] Memory usage stable (no leaks)
- [ ] CPU usage <50% average
- [ ] Cycle duration <2s p95

---

## Risk Mitigation

### Pre-Test Validation
- [ ] Verify sandbox API credentials work
- [ ] Test Prometheus/Grafana connectivity
- [ ] Confirm alertmanager routing
- [ ] Backup current production config
- [ ] Document rollback procedure

### During Test
- Monitor actively for first 4 hours
- Check /health endpoint every 15 minutes
- Review logs for unexpected errors
- Ready to stop bot if critical issues detected

### Abort Criteria
Stop test immediately if:
- Bot crashes repeatedly (>3 in 1 hour)
- Memory leak detected (>200MB/hour growth)
- Data corruption detected
- Unrecoverable guard state
- Security issue identified

---

## Post-Test Deliverables

### 1. Metrics Report
- Guardrail activation summary (count by guard type)
- Streaming uptime statistics
- Fallback activation frequency and duration
- Alert firing accuracy
- System resource utilization

### 2. Incident Log
Document all issues encountered:
- Description and severity
- Root cause analysis
- Resolution steps
- Preventive measures

### 3. Recommendations
Based on findings:
- Configuration tuning suggestions
- Alert threshold adjustments
- Code fixes required
- Production readiness assessment

### 4. Go/No-Go Decision
Final assessment for production readiness:
- **GO** if all success criteria met
- **NO-GO** if critical issues remain
- **GO with caveats** if minor issues can be monitored

---

## Timeline

**Day 0 (Prep)**:
- Deploy to sandbox
- Verify monitoring stack
- Run smoke tests
- Start soak test at 00:00 UTC

**Day 1**:
- Hour 0-4: Active monitoring (baseline)
- Hour 4-8: Execute test scenarios 1-5
- Hour 8-12: Execute scenarios 6-8
- Hour 12-24: Passive monitoring, UTC rollover test

**Day 2** (if 48hr):
- Hour 24-36: Passive monitoring
- Hour 36-44: Repeat critical scenarios
- Hour 44-48: Final validation, collect data

**Day 2-3 (Analysis)**:
- Export all metrics and logs
- Analyze results
- Write report
- Present findings

---

## Next Steps After Soak Test

1. **If Successful**: Proceed to Phase 3.4 Production Planning
2. **If Issues Found**: Create remediation plan, fix, re-test
3. **Document Lessons Learned**: Update runbooks and deployment procedures
