# Sandbox Soak Test - Deployment Checklist

## Pre-Deployment

### Environment Validation
- [ ] Coinbase Sandbox API credentials configured
  ```bash
  export COINBASE_API_KEY="<sandbox_key>"
  export COINBASE_API_SECRET="<sandbox_secret>"
  ```
- [ ] Verify sandbox endpoint connectivity
  ```bash
  curl -H "CB-ACCESS-KEY: $COINBASE_API_KEY" \
       https://api-public.sandbox.exchange.coinbase.com/products/BTC-USD
  ```
- [ ] Confirm Python environment (Python 3.12+)
  ```bash
  python --version
  poetry --version
  ```

### Infrastructure Setup
- [ ] Prometheus running and scraping bot metrics
  ```bash
  docker-compose -f monitoring/docker-compose.yml up -d prometheus
  # Verify: http://localhost:9090/targets
  ```
- [ ] Grafana configured with Prometheus datasource
  ```bash
  docker-compose -f monitoring/docker-compose.yml up -d grafana
  # Verify: http://localhost:3000
  ```
- [ ] Alertmanager configured for notifications
  ```bash
  docker-compose -f monitoring/docker-compose.yml up -d alertmanager
  # Verify: http://localhost:9093
  ```

### Configuration Files
- [ ] `config/profiles/canary.yaml` exists with test settings
- [ ] `.env` file configured for sandbox
- [ ] Backup current production config
  ```bash
  cp config/profiles/production.yaml config/profiles/production.yaml.backup
  ```

---

## Deployment Steps

### 1. Update Dependencies
```bash
cd /Users/rj/PycharmProjects/GPT-Trader
poetry install --sync
```

### 2. Run Unit Tests
```bash
poetry run pytest tests/unit -v --tb=short
# All tests should pass before deployment
```

### 3. Start Monitoring Stack
```bash
cd monitoring
docker-compose up -d
docker-compose ps  # Verify all services running
```

### 4. Configure Environment
```bash
# Copy sandbox environment template
cp .env.sandbox.example .env

# Edit and verify critical settings
cat .env | grep -E "COINBASE_|PERPS_|PROFILE"
```

### 5. Start Bot in Canary Mode
```bash
poetry run python -m bot_v2.cli \
  --profile canary \
  --max-trade-value 100 \
  --streaming-rest-poll-interval 5.0 \
  2>&1 | tee logs/sandbox_soak_$(date +%Y%m%d_%H%M%S).log
```

**Alternative: Background with systemd**
```bash
# Copy service file
sudo cp deploy/systemd/perps-bot-canary.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start perps-bot-canary
sudo systemctl status perps-bot-canary

# Follow logs
journalctl -u perps-bot-canary -f
```

---

## Initial Validation (First 30 Minutes)

### Verify Bot Startup
- [ ] Bot starts without errors
- [ ] Logs show "Started WS streaming thread"
- [ ] No immediate crashes or exceptions

### Check Metrics Endpoint
```bash
# Verify metrics exposed
curl http://localhost:9090/metrics | grep "bot_uptime_seconds"
curl http://localhost:9090/metrics | grep "bot_streaming_connection_state"
curl http://localhost:9090/metrics | grep "bot_guard_active"

# Should see non-zero values
```

### Check Health Endpoint
```bash
curl http://localhost:9090/health | jq .

# Expected output:
# {
#   "status": "ok",
#   "uptime_seconds": <number>,
#   "streaming_connected": true,
#   "streaming": {
#     "connected": true,
#     "heartbeat_lag_seconds": <5,
#     ...
#   },
#   "guards": {}  # Should be empty initially
# }
```

### Verify Prometheus Scraping
1. Open http://localhost:9090
2. Go to Status → Targets
3. Verify bot target is UP
4. Run test query: `bot_uptime_seconds`

### Verify Grafana Dashboards
1. Open http://localhost:3000
2. Navigate to Bot Health Overview dashboard
3. Verify all panels loading data
4. Check streaming panels show connected state

### Check Alertmanager
1. Open http://localhost:9093
2. Verify no alerts firing (should be clean slate)
3. Check alert routing configured correctly

---

## Monitoring Setup

### Create Soak Test Dashboard

Import or create Grafana dashboard with these panels:

**Row 1: System Health**
- Uptime (gauge)
- CPU Usage (graph)
- Memory Usage (graph)
- Cycle Duration p95 (graph)

**Row 2: Guardrails**
- Active Guards (stat)
- Guard Trips by Type (table)
- Error Streak (graph)
- Daily Loss (graph)

**Row 3: Streaming**
- Connection State (stat)
- Heartbeat Lag (graph)
- Reconnect Count (graph)
- Fallback Active (stat)

**Row 4: Trading Activity**
- Order Success Rate (gauge)
- Orders by Status (pie chart)
- Position Size (graph)
- P&L (graph)

### Set Up Alert Logging

Configure Alertmanager to log all alerts:
```yaml
# monitoring/alertmanager/alertmanager.yml
receivers:
  - name: 'soak-test-logger'
    webhook_configs:
      - url: 'http://localhost:8080/webhook'
        send_resolved: true

route:
  receiver: 'soak-test-logger'
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
```

### Start Alert Logger
```bash
# Simple webhook receiver to log alerts
cd monitoring
python -m http.server 8080 &
# Alerts will be logged to terminal
```

---

## Test Scenario Execution

### Scenario Trigger Scripts

**Trigger Order Cap Guard**:
```bash
# Via CLI (if implemented)
poetry run python -m bot_v2.cli.order_test \
  --symbol BTC-USD \
  --side buy \
  --quantity 0.01 \
  --test-guard max_trade_value
```

**Trigger Circuit Breaker**:
```bash
# Force errors by sending invalid orders repeatedly
for i in {1..5}; do
  curl -X POST http://localhost:9090/api/test/force_error
  sleep 1
done
```

**Trigger Streaming Disconnect**:
```bash
# Block WebSocket endpoint temporarily
sudo iptables -A OUTPUT -d wss.exchange.coinbase.com -j DROP
sleep 60
sudo iptables -D OUTPUT -d wss.exchange.coinbase.com -j DROP
```

**Trigger Daily Loss Limit**:
```bash
# Record simulated losses via API
curl -X POST http://localhost:9090/api/test/record_pnl \
  -H "Content-Type: application/json" \
  -d '{"pnl": -12.0}'
```

---

## Data Collection

### Metrics Snapshots

**Hourly Export**:
```bash
#!/bin/bash
# Save as scripts/collect_metrics_hourly.sh

while true; do
  timestamp=$(date +%Y%m%d_%H%M%S)

  # Export instant metrics
  curl -s "http://localhost:9090/api/v1/query?query=bot_uptime_seconds" \
    > "data/soak_${timestamp}_uptime.json"

  curl -s "http://localhost:9090/api/v1/query?query=bot_guard_active" \
    > "data/soak_${timestamp}_guards.json"

  curl -s "http://localhost:9090/api/v1/query?query=bot_streaming_connection_state" \
    > "data/soak_${timestamp}_streaming.json"

  # Health snapshot
  curl -s "http://localhost:9090/health" \
    > "data/soak_${timestamp}_health.json"

  sleep 3600  # Every hour
done
```

**Final Export** (after 24-48hr):
```bash
# Export full time-series data
START_TIME="2025-10-04T00:00:00Z"
END_TIME="2025-10-06T00:00:00Z"

# All bot metrics
curl -G "http://localhost:9090/api/v1/query_range" \
  --data-urlencode "query={__name__=~'bot_.*'}" \
  --data-urlencode "start=${START_TIME}" \
  --data-urlencode "end=${END_TIME}" \
  --data-urlencode "step=15s" \
  > data/soak_full_metrics.json

# Alert history
curl "http://localhost:9090/api/v1/alerts" \
  > data/soak_alerts.json
```

### Log Collection

```bash
# Collect all logs
tar -czf logs/soak_test_$(date +%Y%m%d).tar.gz \
  logs/sandbox_soak_*.log \
  /var/log/perps-bot-canary/*.log

# Extract key events
grep -E "guard|circuit|daily_loss|max_trade" logs/sandbox_soak_*.log > logs/guard_events.log
grep -E "streaming|fallback|reconnect" logs/sandbox_soak_*.log > logs/streaming_events.log
grep -E "ERROR|WARNING|CRITICAL" logs/sandbox_soak_*.log > logs/errors.log
```

---

## Health Checks During Test

### Every 15 Minutes
- [ ] Check /health endpoint shows "ok"
- [ ] Verify bot uptime increasing
- [ ] Confirm no unexpected alerts firing

### Every Hour
- [ ] Review error logs for new issues
- [ ] Check memory usage trend (no leaks)
- [ ] Verify streaming still connected
- [ ] Confirm guard states as expected

### Every 4 Hours
- [ ] Execute manual test scenario
- [ ] Verify alert fired correctly
- [ ] Check recovery behavior
- [ ] Document any anomalies

---

## Emergency Procedures

### Stop Bot
```bash
# Graceful shutdown
sudo systemctl stop perps-bot-canary

# OR if running in terminal
# Send SIGTERM (Ctrl+C)

# Force kill if unresponsive
sudo systemctl kill -s SIGKILL perps-bot-canary
```

### Restart Bot
```bash
sudo systemctl restart perps-bot-canary
sudo systemctl status perps-bot-canary
journalctl -u perps-bot-canary --since "5 minutes ago"
```

### Reset Guards Manually
```bash
# Clear circuit breaker
curl -X POST http://localhost:9090/api/admin/reset_guard \
  -d '{"guard": "circuit_breaker"}'

# Reset daily P&L (for testing only!)
curl -X POST http://localhost:9090/api/admin/reset_daily_pnl
```

### Rollback
```bash
# Stop bot
sudo systemctl stop perps-bot-canary

# Revert to previous version
git checkout <previous_commit>
poetry install --sync

# Restart
sudo systemctl start perps-bot-canary
```

---

## Post-Test Shutdown

### Stop Services
```bash
# Stop bot
sudo systemctl stop perps-bot-canary
sudo systemctl disable perps-bot-canary

# Keep monitoring running for analysis
# docker-compose -f monitoring/docker-compose.yml down
```

### Archive Data
```bash
# Create archive
mkdir -p archive/soak_test_$(date +%Y%m%d)
cp -r data/ archive/soak_test_$(date +%Y%m%d)/
cp -r logs/ archive/soak_test_$(date +%Y%m%d)/

# Compress
tar -czf archive/soak_test_$(date +%Y%m%d).tar.gz \
  archive/soak_test_$(date +%Y%m%d)/
```

### Generate Report
Use analysis scripts to produce final report:
```bash
poetry run python scripts/analysis/soak_test_report.py \
  --metrics data/soak_full_metrics.json \
  --logs logs/sandbox_soak_*.log \
  --output reports/soak_test_$(date +%Y%m%d).md
```

---

## Success Validation

Before marking test complete, verify:
- [ ] Ran for minimum 24 hours
- [ ] All test scenarios executed
- [ ] All expected alerts fired
- [ ] No false positive alerts
- [ ] Data gaps <10s during fallback
- [ ] Guards auto-cleared after conditions met
- [ ] No crashes or deadlocks
- [ ] Memory usage stable
- [ ] Report generated with findings

---

## Common Issues & Solutions

### Bot Won't Start
- Check API credentials
- Verify network connectivity
- Review logs for specific error
- Confirm dependencies installed

### Metrics Not Appearing
- Verify metrics endpoint accessible: `curl localhost:9090/metrics`
- Check Prometheus scrape config
- Confirm firewall not blocking
- Restart Prometheus if needed

### Streaming Won't Connect
- Check WebSocket endpoint reachable
- Verify credentials for authenticated channels
- Review PERPS_ENABLE_STREAMING setting
- Check transport initialization logs

### Guards Not Triggering
- Verify limits configured correctly
- Check guard manager initialization
- Confirm order flow reaching check_order()
- Review guardrails logs

### Alerts Not Firing
- Verify alert rules loaded in Prometheus
- Check alert evaluation interval
- Confirm Alertmanager routing
- Review alert thresholds vs actual metrics

---

## Contact & Support

If critical issues arise during soak test:
1. Document issue thoroughly (logs, metrics, steps to reproduce)
2. Stop test if data integrity at risk
3. Create GitHub issue with `soak-test` label
4. Escalate to team lead if blocking
