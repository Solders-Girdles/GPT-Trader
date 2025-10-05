# Sandbox Soak Test - Quick Start Guide

**Phase 3.2/3.3 Validation** | **Duration**: 24-48 hours | **Environment**: Coinbase Sandbox

---

## Prerequisites

- ✅ Coinbase Sandbox API credentials ([Get them here](https://public.sandbox.exchange.coinbase.com/))
- ✅ Docker Desktop running
- ✅ Python 3.12+ with Poetry installed
- ✅ Available ports: 3000 (Grafana), 9090 (Bot), 9091 (Prometheus), 9093 (Alertmanager)

---

## One-Command Deployment

```bash
# 1. Configure credentials
cp .env.sandbox.example .env
# Edit .env with your Coinbase Sandbox API key/secret

# 2. Deploy everything
./scripts/deploy_sandbox_soak.sh
```

That's it! The script will:
- Validate prerequisites
- Start monitoring stack
- Validate configuration
- Launch the bot

---

## What Gets Deployed

### Bot Configuration (Canary Profile)
- **Environment**: Coinbase Sandbox (no real funds)
- **Symbol**: BTC-USD spot
- **Trading Mode**: Reduce-only
- **Daily Loss Limit**: $10 USD
- **Max Trade Value**: $100 USD per order
- **Position Cap**: 0.01 BTC maximum
- **Streaming**: WebSocket enabled with 5s REST fallback
- **Trading Hours**: 24/7 for continuous testing

### Monitoring Stack
- **Prometheus** (http://localhost:9091) - Metrics collection and alerting
- **Grafana** (http://localhost:3000) - Dashboards and visualization
- **Alertmanager** (http://localhost:9093) - Alert routing and management

### Guardrails Being Tested
1. **Order Caps** - Max $100 per trade, 0.01 BTC position limit
2. **Daily Loss Limit** - Trading halts at $10 daily loss
3. **Circuit Breaker** - Activates after 5 consecutive errors
4. **Streaming Fallback** - Automatic REST polling on WebSocket disconnect

---

## Monitoring During Test

### Quick Status Check
```bash
./scripts/check_soak_status.sh
```

### Start Hourly Metrics Collection
```bash
# Run in background
./scripts/collect_soak_metrics.sh &
```

### View Dashboards
- **Bot Health Overview**: http://localhost:3000/d/bot-health
- **System Overview**: http://localhost:3000/d/system
- **Trading Activity**: http://localhost:3000/d/trading

Login: `admin` / `admin123`

### Check Health Endpoint
```bash
curl http://localhost:9090/health | jq .
```

### Check Metrics
```bash
curl http://localhost:9090/metrics | grep bot_
```

### View Live Logs
```bash
# Find latest log file
ls -t logs/sandbox_soak_*.log | head -1

# Tail it
tail -f logs/sandbox_soak_YYYYMMDD_HHMMSS.log
```

---

## Test Scenarios to Execute

Follow `docs/testing/sandbox_soak_test_plan.md` for detailed scenarios:

### Scenario 1: Normal Operation (First 4 hours)
- Let bot run normally
- Verify streaming connected
- Confirm no guards active
- Monitor baseline metrics

### Scenario 2: Order Cap Guard
- Attempt order >$100
- Verify `bot_guard_trips_total{guard="max_trade_value"}` increments
- Confirm order blocked, not failed

### Scenario 3: Position Cap Guard
- Attempt order >0.01 BTC
- Verify `bot_guard_trips_total{guard="position_limit"}` increments

### Scenario 4: Daily Loss Limit
- Accumulate >$10 realized loss
- Verify `bot_guard_active{guard="daily_loss"}` = 1
- Confirm bot enters reduce-only mode
- Check alert `BotDailyLossLimitBreached` fires

### Scenario 5: Circuit Breaker
- Force 5 consecutive order errors
- Verify `bot_guard_active{guard="circuit_breaker"}` = 1
- Check alert `BotCircuitBreakerTripped` fires
- Confirm trading halts

### Scenario 6: Streaming Disconnect
- Simulate network issue or kill WebSocket
- Verify `bot_streaming_connection_state` = 0
- Confirm `bot_streaming_fallback_active` = 1
- Check REST polling maintains price updates
- Verify alert `BotStreamingDisconnected` fires after 1min

### Scenario 7: Reconnect Storm
- Trigger repeated connection failures
- Verify `bot_streaming_reconnect_total` increments
- Check alert `BotStreamingReconnectStorm` fires after 3 reconnects in 5min

### Scenario 8: Heartbeat Stale
- Monitor during quiet market periods
- Verify `bot_streaming_heartbeat_lag_seconds` metric
- Check alert `BotStreamingHeartbeatStale` fires if lag >15s for 2min

---

## After 24-48 Hours

### 1. Stop Bot
Press `Ctrl+C` in the deployment terminal

### 2. Stop Metrics Collection
```bash
pkill -f collect_soak_metrics
```

### 3. Export Final Metrics
```bash
# Get start time from bot logs (first line timestamp)
START_TIME="2025-10-04T14:23:15Z"

# Get current time
END_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Export
./scripts/export_final_metrics.sh "$START_TIME" "$END_TIME"
```

### 4. Archive Data
```bash
mkdir -p archive
tar -czf archive/soak_test_$(date +%Y%m%d).tar.gz data/ logs/
```

### 5. Analyze Results
Use Prometheus queries from `docs/testing/prometheus_queries.md`:

**Streaming Uptime**:
```promql
avg_over_time(bot_streaming_connection_state[24h]) * 100
```

**Total Guard Trips**:
```promql
sum by (guard) (bot_guard_trips_total)
```

**Order Success Rate**:
```promql
sum(increase(bot_order_attempts_total{status="success"}[24h])) /
sum(increase(bot_order_attempts_total{status="attempted"}[24h]))
```

**Fallback Activations**:
```promql
sum_over_time(bot_streaming_fallback_active[24h])
```

### 6. Generate Report
Document findings in `docs/testing/soak_test_results.md`:
- Streaming uptime percentage
- Fallback activation count and duration
- Guardrail trip summary by type
- Alert firing history
- Order execution statistics
- System stability (crashes, memory leaks, CPU usage)
- Issues encountered and resolutions

---

## Success Criteria

Mark test as **PASS** if all criteria met:

### Guardrails (100% reliability)
- ✅ All order caps blocked correctly (0 bypasses)
- ✅ Daily loss guard activated at exactly $10
- ✅ Circuit breaker engaged after 5 errors
- ✅ Guards auto-cleared after cooldown/reset
- ✅ No false activations

### Streaming (>99% uptime)
- ✅ WebSocket uptime >99%
- ✅ Fallback activated within 30s of disconnect
- ✅ No mark price gaps >10s during fallback
- ✅ Reconnect successful within 3 attempts
- ✅ Fallback deactivated within 10s of recovery

### Observability (100% alert accuracy)
- ✅ All expected alerts fired correctly
- ✅ Zero false positive alerts
- ✅ Metrics reflected actual system state
- ✅ /health endpoint always accurate

### System Stability (Zero crashes)
- ✅ No bot crashes or unexpected exits
- ✅ No deadlocks or hangs
- ✅ Memory usage stable (no leaks)
- ✅ CPU usage <50% average
- ✅ Cycle duration <2s p95

---

## Troubleshooting

### Bot won't start
```bash
# Check credentials
cat .env | grep COINBASE_API

# Test API connection
curl -H "CB-ACCESS-KEY: $COINBASE_API_KEY" \
     https://api-public.sandbox.exchange.coinbase.com/products/BTC-USD

# Check logs
tail -100 logs/sandbox_soak_*.log
```

### Monitoring stack not responding
```bash
cd monitoring
docker-compose down
docker-compose up -d
docker-compose logs
```

### Metrics not appearing in Prometheus
```bash
# Verify bot metrics endpoint
curl http://localhost:9090/metrics | grep bot_uptime

# Check Prometheus targets
open http://localhost:9091/targets

# Verify scrape config
cat monitoring/prometheus/prometheus.yml | grep -A5 gpt-trader
```

### Streaming won't connect
```bash
# Check WebSocket configuration
curl http://localhost:9090/health | jq .streaming

# Verify environment variables
echo $PERPS_ENABLE_STREAMING  # Should be 1
echo $PERPS_STREAM_LEVEL      # Should be 1

# Check logs for WebSocket errors
grep -i "websocket\|streaming" logs/sandbox_soak_*.log
```

---

## Quick Reference

### Commands
```bash
# Deploy
./scripts/deploy_sandbox_soak.sh

# Check status
./scripts/check_soak_status.sh

# Collect metrics (background)
./scripts/collect_soak_metrics.sh &

# Export final data
./scripts/export_final_metrics.sh START_TIME END_TIME

# Stop monitoring stack
cd monitoring && docker-compose down
```

### URLs
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9091
- **Alertmanager**: http://localhost:9093
- **Bot Health**: http://localhost:9090/health
- **Bot Metrics**: http://localhost:9090/metrics

### Key Files
- **Configuration**: `config/profiles/canary.yaml`
- **Test Plan**: `docs/testing/sandbox_soak_test_plan.md`
- **Deployment Checklist**: `docs/testing/sandbox_deployment_checklist.md`
- **Prometheus Queries**: `docs/testing/prometheus_queries.md`
- **Scripts README**: `scripts/README.md`

---

## Need Help?

1. Check `scripts/README.md` for detailed script documentation
2. Review `docs/testing/sandbox_deployment_checklist.md` for manual steps
3. Consult `docs/testing/prometheus_queries.md` for monitoring queries
4. Check bot logs in `logs/sandbox_soak_*.log`
5. Review monitoring stack logs: `cd monitoring && docker-compose logs`

---

**Ready to begin? Run `./scripts/deploy_sandbox_soak.sh`**
